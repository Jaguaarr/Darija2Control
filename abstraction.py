"""Symbolic model construction for N-dimensional systems."""
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm

from models import RobotModel
from partition import Partition, Cell
from parallel import ParallelBackend


@dataclass
class SymbolicModel:
    """Symbolic abstraction of robot dynamics."""
    partition: Partition
    model: RobotModel
    inputs: List[np.ndarray]
    input_to_idx: Dict[tuple, int]
    idx_to_input: Dict[int, np.ndarray]

    # Successor relation: cell_idx -> input_idx -> set of cell_idx
    successors: Dict[int, Dict[int, Set[int]]] = field(default_factory=dict)

    # Labelling: cell_idx -> set of observation names
    labelling: Dict[int, Set[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.n_cells = len(self.partition)
        self.n_inputs = len(self.inputs)

    def get_successors(self, cell_idx: int, input_idx: int) -> Set[int]:
        """Get successor cells for given cell and input."""
        return self.successors.get(cell_idx, {}).get(input_idx, set())

    def get_labels(self, cell_idx: int) -> Set[str]:
        """Get observations true in this cell."""
        return self.labelling.get(cell_idx, set())


class AbstractionBuilder:
    """Builds symbolic models using interval over-approximation."""

    def __init__(self, model: RobotModel, partition: Partition,
                 parallel_backend: Optional[ParallelBackend] = None):
        self.model = model
        self.partition = partition
        self.parallel = parallel_backend or ParallelBackend()

        # Prepare inputs
        self.inputs = model.get_inputs()
        self.input_to_idx = {tuple(inp): i for i, inp in enumerate(self.inputs)}
        self.idx_to_input = {i: inp for i, inp in enumerate(self.inputs)}

        # Disturbance bounds
        self.dist_bounds = model.get_disturbance_bounds()

        # For Jacobian-based over-approximation
        self.use_jacobian = True

        # Create a mapping from cell to index for quick lookup
        self._cell_to_idx = {}  # Will be populated lazily

    def _get_cell_index(self, cell: Cell) -> int:
        """
        Get the index of a cell.
        For lazy-loading partitions, we need to compute the linear index.
        """
        # Check if we already have this cell in cache
        if cell in self._cell_to_idx:
            return self._cell_to_idx[cell]

        # Compute linear index from tuple index
        stride = 1
        linear_idx = 0
        for d in range(len(cell.index) - 1, -1, -1):
            linear_idx += cell.index[d] * stride
            stride *= self.partition.resolutions[d]

        # Cache it
        self._cell_to_idx[cell] = linear_idx
        return linear_idx

    def over_approximate_reach(self, cell: Cell, input_vec: np.ndarray,
                               w_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute over-approximation of reachable set from cell under input.
        Returns (min_bounds, max_bounds) for reachable set.
        """
        try:
            center = cell.center()
            half_widths = np.array([(high - low) / 2 for low, high in cell.bounds])

            if self.use_jacobian:
                # Use Jacobian for tighter over-approximation
                Jx, Ju = self.model.jacobian(center, input_vec)

                # Linearized dynamics: x_next ≈ f(center) + Jx*(x - center) + Ju*(u - u) + Dw*w
                f_center = self.model.dynamics(center, input_vec, np.zeros(self.model.state_dim))

                # Ensure arrays have correct dimensions
                if len(half_widths) != Jx.shape[1]:
                    print(f"⚠️ Dimension mismatch: half_widths={len(half_widths)}, Jx.shape={Jx.shape}")
                    # Fallback to simple method
                    return self._over_approximate_reach_simple(cell, input_vec, w_bounds)

                # Effect of state uncertainty
                state_effect = np.abs(Jx) @ half_widths

                # Effect of disturbance (simplified - assumes Dw = I)
                dist_effect = w_bounds

                # Total uncertainty
                uncertainty = state_effect + dist_effect

                min_bounds = f_center - uncertainty
                max_bounds = f_center + uncertainty
            else:
                return self._over_approximate_reach_simple(cell, input_vec, w_bounds)

            return min_bounds, max_bounds
        except Exception as e:
            print(f"⚠️ Error in over_approximate_reach, using simple method: {e}")
            return self._over_approximate_reach_simple(cell, input_vec, w_bounds)

    def _over_approximate_reach_simple(self, cell: Cell, input_vec: np.ndarray,
                                       w_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple corner sampling method as fallback."""
        center = cell.center()
        half_widths = np.array([(high - low) / 2 for low, high in cell.bounds])

        # Simple corner sampling
        corners = []
        for signs in itertools.product([-1, 1], repeat=self.model.state_dim):
            corner = center + np.array(signs) * half_widths
            corners.append(corner)

        # Evaluate dynamics at corners (with zero disturbance)
        next_corners = [self.model.dynamics(c, input_vec, np.zeros(self.model.state_dim))
                        for c in corners]

        # Add center evaluation
        next_center = self.model.dynamics(center, input_vec, np.zeros(self.model.state_dim))
        next_corners.append(next_center)

        # Take min/max and add disturbance bounds
        next_corners = np.array(next_corners)
        min_bounds = np.min(next_corners, axis=0) - w_bounds
        max_bounds = np.max(next_corners, axis=0) + w_bounds

        return min_bounds, max_bounds

    def _compute_successor(self, task):
        """
        Helper method for parallel computation.
        """
        cell_idx, cell, input_idx, input_vec = task
        try:
            min_bounds, max_bounds = self.over_approximate_reach(
                cell, input_vec, self.dist_bounds
            )

            # SAFETY CHECK: Clip bounds to state space limits
            state_bounds = self.model.get_state_bounds()
            for d in range(len(min_bounds)):
                if d < len(state_bounds):
                    min_bounds[d] = max(min_bounds[d], state_bounds[d][0])
                    max_bounds[d] = min(max_bounds[d], state_bounds[d][1])

            # Find intersecting cells
            successor_cells = self.partition.box_to_cells(min_bounds, max_bounds)

            # Convert cells to indices
            successor_indices = set()
            for succ_cell in successor_cells:
                try:
                    succ_idx = self._get_cell_index(succ_cell)
                    # SAFETY CHECK: Ensure index is valid
                    if succ_idx < len(self.partition):
                        successor_indices.add(succ_idx)
                except Exception:
                    continue

            return cell_idx, input_idx, successor_indices
        except Exception as e:
            return cell_idx, input_idx, set()



    def build_successors(self, progress_bar: bool = True) -> SymbolicModel:
        """Build successor relation for all cells and inputs."""
        print(f"🔨 Building symbolic model with {len(self.partition)} cells and {len(self.inputs)} inputs...")

        model = SymbolicModel(
            partition=self.partition,
            model=self.model,
            inputs=self.inputs,
            input_to_idx=self.input_to_idx,
            idx_to_input=self.idx_to_input
        )

        # Prepare all (cell_idx, input_idx) pairs
        tasks = []

        # Instead of enumerating self.partition.cells directly (which might be lazy),
        # we iterate over all possible indices
        ranges = [range(r) for r in self.partition.resolutions]

        # Create a progress bar for cell enumeration if needed
        cell_iterator = itertools.product(*ranges)
        if progress_bar:
            total_cells = len(self.partition)
            cell_iterator = tqdm(cell_iterator, desc="Preparing tasks", total=total_cells)

        for idx_tuple in cell_iterator:
            cell = self.partition.get_cell(idx_tuple)
            cell_idx = self._get_cell_index(cell)
            for input_idx, input_vec in enumerate(self.inputs):
                tasks.append((cell_idx, cell, input_idx, input_vec))

        print(f"📊 Created {len(tasks)} tasks ({len(self.partition)} cells × {len(self.inputs)} inputs)")

        # Run parallel computation using the helper method
        results = self.parallel.map(self._compute_successor, tasks,
                                    desc="Building successors" if progress_bar else None)

        # Aggregate results
        success_count = 0
        for cell_idx, input_idx, succ_indices in results:
            if succ_indices:
                success_count += 1
                if cell_idx not in model.successors:
                    model.successors[cell_idx] = {}
                model.successors[cell_idx][input_idx] = succ_indices

        print(f"✅ Successor relation built: {success_count} transitions found")
        return model

    def add_labelling(self, model: SymbolicModel,
                      region_definitions: Dict[str, List[Tuple[float, float]]]):
        """
        Add observation labels to symbolic model based on region definitions.

        Args:
            model: SymbolicModel to label
            region_definitions: Dict mapping region names to list of (min, max) bounds
                               for each dimension (can be partial)
        """
        print(f"🏷️  Adding labelling for regions: {list(region_definitions.keys())}")

        # Iterate over all cells using the lazy iterator
        ranges = [range(r) for r in self.partition.resolutions]
        cell_iterator = itertools.product(*ranges)

        for idx_tuple in cell_iterator:
            cell = self.partition.get_cell(idx_tuple)
            cell_idx = self._get_cell_index(cell)

            labels = set()
            center = cell.center()

            for region_name, bounds_list in region_definitions.items():
                in_region = True
                # Ensure bounds_list has at least as many dimensions as we need
                for d, (low, high) in enumerate(bounds_list):
                    if d >= len(center):
                        continue
                    if center[d] < low or center[d] > high:
                        in_region = False
                        break

                if in_region:
                    labels.add(region_name)

            if labels:
                model.labelling[cell_idx] = labels

        # Count labelled cells
        labelled_count = len(model.labelling)
        print(f"✅ Labelling complete: {labelled_count} cells have labels")

        return model

    def get_cell_by_index(self, idx: int) -> Cell:
        """
        Get a cell by its linear index.
        Useful for converting between index and cell.
        """
        # Convert linear index to tuple index
        remaining = idx
        tuple_idx = []
        for d in range(len(self.partition.resolutions) - 1, -1, -1):
            tuple_idx.insert(0, remaining % self.partition.resolutions[d])
            remaining //= self.partition.resolutions[d]

        return self.partition.get_cell(tuple(tuple_idx))