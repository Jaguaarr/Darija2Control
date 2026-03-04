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

    successors: Dict[int, Dict[int, Set[int]]] = field(default_factory=dict)
    labelling: Dict[int, Set[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.n_cells = len(self.partition)
        self.n_inputs = len(self.inputs)

    def get_successors(self, cell_idx: int, input_idx: int) -> Set[int]:
        return self.successors.get(cell_idx, {}).get(input_idx, set())

    def get_labels(self, cell_idx: int) -> Set[str]:
        return self.labelling.get(cell_idx, set())


# Top-level function for computing successors (must be at module level for pickling)
def _compute_successor_task(task, model, partition, dist_bounds, use_jacobian, cell_to_idx_cache):
    """Compute successors for a single (cell, input) pair."""
    cell_idx, cell, input_idx, input_vec = task
    try:
        # Simple corner sampling (works everywhere)
        center = cell.center()
        half_widths = np.array([(high - low) / 2 for low, high in cell.bounds])

        # Generate corners
        corners = []
        for signs in itertools.product([-1, 1], repeat=model.state_dim):
            corner = center + np.array(signs) * half_widths
            corners.append(corner)

        # Evaluate dynamics at corners
        next_corners = [model.dynamics(c, input_vec, np.zeros(model.state_dim)) for c in corners]
        next_center = model.dynamics(center, input_vec, np.zeros(model.state_dim))
        next_corners.append(next_center)

        next_corners = np.array(next_corners)
        min_bounds = np.min(next_corners, axis=0) - dist_bounds
        max_bounds = np.max(next_corners, axis=0) + dist_bounds

        # Clip to state bounds
        state_bounds = model.get_state_bounds()
        for d in range(len(min_bounds)):
            if d < len(state_bounds):
                min_bounds[d] = max(min_bounds[d], state_bounds[d][0])
                max_bounds[d] = min(max_bounds[d], state_bounds[d][1])

        # Find intersecting cells
        successor_cells = partition.box_to_cells(min_bounds, max_bounds)

        # Convert cells to indices
        successor_indices = set()
        for succ_cell in successor_cells:
            # Compute linear index
            stride = 1
            linear_idx = 0
            for d in range(len(succ_cell.index) - 1, -1, -1):
                linear_idx += succ_cell.index[d] * stride
                stride *= partition.resolutions[d]
            successor_indices.add(linear_idx)

        return cell_idx, input_idx, successor_indices

    except Exception as e:
        return cell_idx, input_idx, set()


# Top-level function for labelling (must be at module level for pickling)
def _process_labelling_task(task):
    """Process a single labelling task."""
    cell_idx, cell, region_definitions = task
    labels = set()
    center = cell.center()

    for region_name, bounds_list in region_definitions.items():
        in_region = True
        for d, (low, high) in enumerate(bounds_list):
            if d >= len(center):
                continue
            if center[d] < low or center[d] > high:
                in_region = False
                break
        if in_region:
            labels.add(region_name)

    return cell_idx, labels


class AbstractionBuilder:
    """Builds symbolic models using interval over-approximation."""

    def __init__(self, model: RobotModel, partition: Partition,
                 parallel_backend: Optional[ParallelBackend] = None):
        self.model = model
        self.partition = partition
        self.parallel = parallel_backend or ParallelBackend()

        self.inputs = model.get_inputs()
        self.input_to_idx = {tuple(inp): i for i, inp in enumerate(self.inputs)}
        self.idx_to_input = {i: inp for i, inp in enumerate(self.inputs)}
        self.dist_bounds = model.get_disturbance_bounds()
        self.use_jacobian = True

        # Check if GPU is available
        self._gpu_enabled = hasattr(self.parallel, 'gpu') and self.parallel.gpu and self.parallel.gpu.available

    def _get_cell_index(self, cell: Cell) -> int:
        """Get linear index of a cell."""
        stride = 1
        linear_idx = 0
        for d in range(len(cell.index) - 1, -1, -1):
            linear_idx += cell.index[d] * stride
            stride *= self.partition.resolutions[d]
        return linear_idx

    def build_successors(self, progress_bar: bool = True) -> SymbolicModel:
        """Build successor relation for all cells and inputs."""
        print(f"🔨 Building abstraction: {len(self.partition)} cells × {len(self.inputs)} inputs")

        model = SymbolicModel(
            partition=self.partition,
            model=self.model,
            inputs=self.inputs,
            input_to_idx=self.input_to_idx,
            idx_to_input=self.idx_to_input
        )

        # Create all tasks
        tasks = []
        ranges = [range(r) for r in self.partition.resolutions]

        cell_iter = itertools.product(*ranges)
        if progress_bar:
            cell_iter = tqdm(cell_iter, desc="Creating tasks", total=len(self.partition))

        for idx_tuple in cell_iter:
            cell = self.partition.get_cell(idx_tuple)
            cell_idx = self._get_cell_index(cell)
            for input_idx, input_vec in enumerate(self.inputs):
                tasks.append((cell_idx, cell, input_idx, input_vec))

        print(f"📊 {len(tasks)} tasks created")

        # Create a wrapper function that passes the necessary data
        from functools import partial
        worker_func = partial(
            _compute_successor_task,
            model=self.model,
            partition=self.partition,
            dist_bounds=self.dist_bounds,
            use_jacobian=self.use_jacobian,
            cell_to_idx_cache={}
        )

        # Process tasks in parallel
        results = self.parallel.map(worker_func, tasks,
                                    desc="Building successors" if progress_bar else None)

        # Aggregate results
        for cell_idx, input_idx, succ_indices in results:
            if succ_indices:
                if cell_idx not in model.successors:
                    model.successors[cell_idx] = {}
                model.successors[cell_idx][input_idx] = succ_indices

        # Count transitions
        total = sum(len(inputs) for inputs in model.successors.values())
        print(f"✅ Built {total} transitions")

        return model

    def add_labelling(self, model: SymbolicModel,
                      region_definitions: Dict[str, List[Tuple[float, float]]]):
        """Add region labels to cells."""
        print(f"🏷️  Adding labels for regions: {list(region_definitions.keys())}")

        # Create labelling tasks
        tasks = []
        ranges = [range(r) for r in self.partition.resolutions]

        for idx_tuple in itertools.product(*ranges):
            cell = self.partition.get_cell(idx_tuple)
            cell_idx = self._get_cell_index(cell)
            tasks.append((cell_idx, cell, region_definitions))

        # Process in parallel if many cells
        if len(tasks) > 10000 and hasattr(self.parallel, 'map'):
            print(f"🚀 Parallel labelling for {len(tasks)} cells")
            results = self.parallel.map(_process_labelling_task, tasks,
                                       desc="Labelling cells")
        else:
            # Sequential for small numbers
            results = [_process_labelling_task(task) for task in tqdm(tasks, desc="Labelling cells")]

        # Store results
        for cell_idx, labels in results:
            if labels:
                model.labelling[cell_idx] = labels

        print(f"✅ Labelled {len(model.labelling)} cells")
        return model