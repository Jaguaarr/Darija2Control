"""N-dimensional state space partitioning."""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from itertools import product
import time


@dataclass
class Cell:
    """Representation of an N-dimensional cell."""
    index: Tuple[int, ...]
    bounds: List[Tuple[float, float]]  # (min, max) for each dimension

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return self.index == other.index

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside this cell."""
        for i, (low, high) in enumerate(self.bounds):
            if point[i] < low or point[i] >= high:
                return False
        return True

    def center(self) -> np.ndarray:
        """Return center point of cell."""
        return np.array([(low + high) / 2 for low, high in self.bounds])

    def volume(self) -> float:
        """Return volume of cell."""
        vol = 1.0
        for low, high in self.bounds:
            vol *= (high - low)
        return vol


class Partition:
    """N-dimensional partition manager (optimized with lazy loading)."""

    def __init__(self, bounds: List[Tuple[float, float]],
                 resolutions: Optional[List[int]] = None,
                 custom_intervals: Optional[List[List[float]]] = None):
        """
        Initialize partition (does NOT generate all cells upfront).

        Args:
            bounds: List of (min, max) for each dimension
            resolutions: Number of cells per dimension (if uniform grid)
            custom_intervals: List of interval boundaries per dimension
        """
        start = time.time()

        self.bounds = bounds
        self.dim = len(bounds)
        print(f"📐 Creating {self.dim}D partition...")

        # Validate resolutions match dimension
        if resolutions and len(resolutions) != self.dim:
            print(f"⚠️ Resolutions length {len(resolutions)} doesn't match dimension {self.dim}")
            # Adjust resolutions
            if len(resolutions) < self.dim:
                resolutions = resolutions + [10] * (self.dim - len(resolutions))
            else:
                resolutions = resolutions[:self.dim]
            print(f"   Adjusted to {resolutions}")

        # Create intervals
        if custom_intervals:
            self.intervals = custom_intervals
            self.resolutions = [len(interv) - 1 for interv in custom_intervals]
        else:
            if not resolutions:
                resolutions = [10] * self.dim  # Default to 10 instead of 100
            self.resolutions = resolutions
            self.intervals = []
            for d, (low, high) in enumerate(bounds):
                n = resolutions[d]
                if n <= 0:
                    raise ValueError(f"Resolution must be positive, got {n}")
                # Add small epsilon to ensure high bound is included
                self.intervals.append(np.linspace(low, high, n + 1))

        # Calculate total cells without generating them
        self._n_cells = int(np.prod(self.resolutions))

        # Create lookup dictionary ONLY for existing cells (will be populated on-demand)
        self.index_to_cell = {}

        # For backward compatibility - create cells list ONLY if total cells is small
        self.cells = []
        self.cell_indices = []

        # Only pre-generate cells if total is less than 1000 (for small test cases)
        if self._n_cells < 1000:
            print(f"📦 Pre-generating {self._n_cells} cells for backward compatibility...")
            ranges = [range(self.resolutions[d]) for d in range(self.dim)]
            for idx_tuple in product(*ranges):
                cell = self._create_cell(idx_tuple)
                self.cells.append(cell)
                self.cell_indices.append(idx_tuple)
                self.index_to_cell[idx_tuple] = cell
        else:
            # For large partitions, we don't pre-generate
            print(f"📊 Large partition: {self._n_cells} cells (lazy loading enabled)")

        print(f"✅ Partition initialized: {self._n_cells} total cells")
        print(f"⏱️  Init time: {time.time() - start:.3f}s")

    def _create_cell(self, idx_tuple: Tuple[int, ...]) -> Cell:
        """Create a cell for the given index (internal method)."""
        if len(idx_tuple) != self.dim:
            raise ValueError(f"Index tuple length {len(idx_tuple)} != dimension {self.dim}")

        cell_bounds = []
        for d, idx in enumerate(idx_tuple):
            if idx < 0 or idx >= self.resolutions[d]:
                raise IndexError(f"Index {idx} out of bounds for dimension {d} (0-{self.resolutions[d] - 1})")

            low = self.intervals[d][idx]
            high = self.intervals[d][idx + 1]
            cell_bounds.append((low, high))
        return Cell(idx_tuple, cell_bounds)

    def get_cell(self, idx_tuple: Tuple[int, ...]) -> Cell:
        """Get cell by index (creates and caches on-demand)."""
        # Validate index tuple
        if len(idx_tuple) != self.dim:
            raise ValueError(f"Index tuple length {len(idx_tuple)} must match dimension {self.dim}")

        for d, idx in enumerate(idx_tuple):
            if idx < 0 or idx >= self.resolutions[d]:
                raise IndexError(f"Index {idx} out of bounds for dimension {d} (0-{self.resolutions[d] - 1})")

        if idx_tuple not in self.index_to_cell:
            self.index_to_cell[idx_tuple] = self._create_cell(idx_tuple)
        return self.index_to_cell[idx_tuple]

    def point_to_cell(self, point: np.ndarray) -> Optional[Cell]:
        """Find cell containing a point."""
        if len(point) != self.dim:
            print(f"Warning: Point dimension {len(point)} != partition dimension {self.dim}")
            return None

        idx = []
        for d, (low, high) in enumerate(self.bounds):
            # Check if point is within bounds (with small epsilon)
            eps = 1e-10
            if point[d] < low - eps or point[d] > high + eps:
                print(f"Point {point[d]} outside dimension {d} bounds [{low}, {high}]")
                return None

            # Clamp to bounds
            p = max(low, min(point[d], high - eps))

            # Find interval using linear search (simpler and more reliable)
            intervals = self.intervals[d]
            found = False
            for i in range(len(intervals) - 1):
                if intervals[i] <= p < intervals[i + 1]:
                    idx.append(i)
                    found = True
                    break

            if not found:
                # If at the very end, use last interval
                if abs(p - intervals[-1]) < eps:
                    idx.append(len(intervals) - 2)
                else:
                    print(f"Could not find interval for point {p} in dimension {d}")
                    return None

        try:
            return self.get_cell(tuple(idx))
        except (IndexError, ValueError) as e:
            print(f"Error getting cell for indices {idx}: {e}")
            return None

    def idx_to_linear(self, idx_tuple: Tuple[int, ...]) -> int:
        """Convert tuple index to linear index."""
        linear = 0
        stride = 1
        for d in range(self.dim - 1, -1, -1):
            linear += idx_tuple[d] * stride
            stride *= self.resolutions[d]
        return linear

    def linear_to_idx(self, linear: int) -> Tuple[int, ...]:
        """Convert linear index to tuple index."""
        if linear < 0 or linear >= self._n_cells:
            raise IndexError(f"Linear index {linear} out of bounds (0-{self._n_cells - 1})")

        idx = []
        remaining = linear
        for d in range(self.dim - 1, -1, -1):
            stride = 1
            for _ in range(d):
                stride *= self.resolutions[_]
            idx.insert(0, remaining // stride)
            remaining %= stride
        return tuple(idx)

    def box_to_cells(self, box_min: np.ndarray, box_max: np.ndarray) -> List[Cell]:
        """Find all cells intersecting a bounding box."""
        if np.any(box_min >= box_max):
            return []

        # Clip to bounds
        box_min = np.maximum(box_min, [b[0] for b in self.bounds])
        box_max = np.minimum(box_max, [b[1] for b in self.bounds])

        # Find index ranges
        idx_ranges = []
        for d in range(self.dim):
            intervals = self.intervals[d]

            # Find first interval containing box_min[d]
            start = 0
            while start < len(intervals) - 1 and intervals[start + 1] <= box_min[d]:
                start += 1

            # Find last interval containing box_max[d]
            end = start
            while end < len(intervals) - 1 and intervals[end] < box_max[d]:
                end += 1

            # Ensure indices are within bounds
            start = max(0, min(start, self.resolutions[d] - 1))
            end = max(start, min(end, self.resolutions[d] - 1))

            idx_ranges.append(range(start, end + 1))

        # Generate all index tuples and get/create cells
        cells = []
        for idx_tuple in product(*idx_ranges):
            try:
                cells.append(self.get_cell(idx_tuple))
            except IndexError:
                continue

        return cells

    def get_neighbors(self, cell: Cell, radius: int = 1) -> List[Cell]:
        """Get neighboring cells within Manhattan distance radius."""
        idx = cell.index
        neighbors = []

        ranges = []
        for d in range(self.dim):
            low = max(0, idx[d] - radius)
            high = min(self.resolutions[d] - 1, idx[d] + radius)
            ranges.append(range(low, high + 1))

        for neighbor_idx in product(*ranges):
            if neighbor_idx != idx:
                try:
                    neighbors.append(self.get_cell(neighbor_idx))
                except IndexError:
                    continue

        return neighbors

    def __len__(self) -> int:
        return self._n_cells

    def __iter__(self) -> Iterator[Cell]:
        """Lazy iteration over all cells (generates on-the-fly)."""
        ranges = [range(self.resolutions[d]) for d in range(self.dim)]
        for idx_tuple in product(*ranges):
            yield self.get_cell(idx_tuple)

    def get_cells_batch(self, start: int, end: int) -> List[Cell]:
        """Get a batch of cells (useful for parallel processing)."""
        ranges = [range(self.resolutions[d]) for d in range(self.dim)]
        cells = []
        count = 0
        for idx_tuple in product(*ranges):
            if count >= start and count < end:
                cells.append(self.get_cell(idx_tuple))
            count += 1
            if count >= end:
                break
        return cells


class _LazyCellList:
    """Helper class for backward compatibility - acts like a list but loads cells lazily."""

    def __init__(self, partition):
        self.partition = partition
        self._cache = {}

    def __getitem__(self, idx):
        """Support indexing like cells[i]."""
        if isinstance(idx, int):
            if idx in self._cache:
                return self._cache[idx]

            # Convert linear index to tuple index
            try:
                tuple_idx = self.partition.linear_to_idx(idx)
                cell = self.partition.get_cell(tuple_idx)
                self._cache[idx] = cell
                return cell
            except (IndexError, ValueError) as e:
                raise IndexError(f"Cell index {idx} out of range: {e}")
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self):
        return self.partition._n_cells

    def __iter__(self):
        """Iterate over all cells."""
        return iter(self.partition)

    def __contains__(self, item):
        """Check if cell exists."""
        if isinstance(item, Cell):
            return item.index in self.partition.index_to_cell
        return False

    def index(self, cell: Cell) -> int:
        """Find index of a cell."""
        if not isinstance(cell, Cell):
            raise ValueError("Can only find index of Cell objects")

        try:
            return self.partition.idx_to_linear(cell.index)
        except (AttributeError, KeyError):
            # Fallback: calculate manually
            stride = 1
            linear_idx = 0
            for d in range(len(cell.index) - 1, -1, -1):
                linear_idx += cell.index[d] * stride
                stride *= self.partition.resolutions[d]
            return linear_idx