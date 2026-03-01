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

        # Create intervals
        if custom_intervals:
            self.intervals = custom_intervals
            self.resolutions = [len(interv) - 1 for interv in custom_intervals]
        else:
            if not resolutions:
                resolutions = [100] * self.dim
            self.resolutions = resolutions
            self.intervals = []
            for d, (low, high) in enumerate(bounds):
                n = resolutions[d]
                self.intervals.append(np.linspace(low, high, n + 1))

        # Calculate total cells without generating them
        self._n_cells = int(np.prod(self.resolutions))

        # Create lookup dictionary ONLY for existing cells (will be populated on-demand)
        self.index_to_cell = {}

        # For backward compatibility - create cells list ONLY if total cells is small
        # This prevents breaking existing code that expects self.cells
        self.cells = []
        self.cell_indices = []

        # Only pre-generate cells if total is less than 1000 (for small test cases)
        # This maintains backward compatibility while keeping performance
        if self._n_cells < 1000:
            print(f"📦 Pre-generating {self._n_cells} cells for backward compatibility...")
            ranges = [range(len(self.intervals[d]) - 1) for d in range(self.dim)]
            for idx_tuple in product(*ranges):
                cell = self._create_cell(idx_tuple)
                self.cells.append(cell)
                self.cell_indices.append(idx_tuple)
                self.index_to_cell[idx_tuple] = cell
        else:
            # For large partitions, we don't pre-generate but create a property that warns
            self.cells = _LazyCellList(self)
            print(f"⚠️  Large partition detected ({self._n_cells} cells). Using lazy loading.")

        print(f"✅ Partition initialized: {self._n_cells} total cells")
        print(f"⏱️  Init time: {time.time()-start:.3f}s")

    def _create_cell(self, idx_tuple: Tuple[int, ...]) -> Cell:
        """Create a cell for the given index (internal method)."""
        cell_bounds = []
        for d, idx in enumerate(idx_tuple):
            low = self.intervals[d][idx]
            high = self.intervals[d][idx + 1]
            cell_bounds.append((low, high))
        return Cell(idx_tuple, cell_bounds)

    def get_cell(self, idx_tuple: Tuple[int, ...]) -> Cell:
        """Get cell by index (creates and caches on-demand)."""
        if idx_tuple not in self.index_to_cell:
            self.index_to_cell[idx_tuple] = self._create_cell(idx_tuple)
        return self.index_to_cell[idx_tuple]

    def point_to_cell(self, point: np.ndarray) -> Optional[Cell]:
        """Find cell containing a point."""
        if len(point) != self.dim:
            raise ValueError(f"Point dimension {len(point)} != partition dimension {self.dim}")

        idx = []
        for d, (low, high) in enumerate(self.bounds):
            if point[d] < low or point[d] >= high:
                return None

            # Binary search for interval
            intervals = self.intervals[d]
            left, right = 0, len(intervals) - 1
            while left < right:
                mid = (left + right) // 2
                if point[d] < intervals[mid]:
                    right = mid
                elif point[d] >= intervals[mid + 1]:
                    left = mid + 1
                else:
                    left = mid
                    break
            idx.append(left)

        return self.get_cell(tuple(idx))

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
            idx_ranges.append(range(start, end + 1))

        # Generate all index tuples and get/create cells
        cells = []
        for idx_tuple in product(*idx_ranges):
            cells.append(self.get_cell(idx_tuple))

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
                neighbors.append(self.get_cell(neighbor_idx))

        return neighbors

    def __len__(self) -> int:
        return self._n_cells

    def __iter__(self) -> Iterator[Cell]:
        """Lazy iteration over all cells (generates on-the-fly)."""
        ranges = [range(r) for r in self.resolutions]
        for idx_tuple in product(*ranges):
            yield self.get_cell(idx_tuple)

    def get_cells_batch(self, start: int, end: int) -> List[Cell]:
        """Get a batch of cells (useful for parallel processing)."""
        ranges = [range(r) for r in self.resolutions]
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
            # Convert integer index to tuple index
            if idx in self._cache:
                return self._cache[idx]

            # Find the idx-th cell
            count = 0
            ranges = [range(r) for r in self.partition.resolutions]
            for idx_tuple in product(*ranges):
                if count == idx:
                    cell = self.partition.get_cell(idx_tuple)
                    self._cache[idx] = cell
                    return cell
                count += 1
            raise IndexError("Cell index out of range")
        else:
            raise TypeError("Invalid index type")

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
        """Find index of a cell (slow but works)."""
        if not isinstance(cell, Cell):
            raise ValueError("Can only find index of Cell objects")

        # Calculate linear index from tuple index
        stride = 1
        linear_idx = 0
        for d in range(len(cell.index)-1, -1, -1):
            linear_idx += cell.index[d] * stride
            stride *= self.partition.resolutions[d]
        return linear_idx