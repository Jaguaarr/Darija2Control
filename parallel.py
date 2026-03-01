"""Parallelization backends for CPU and GPU."""
import numpy as np
from typing import Callable, List, Any, Optional, Set, Dict
import multiprocessing as mp
from tqdm import tqdm


class ParallelBackend:
    """Abstract base for parallelization backends."""

    def map(self, func: Callable, items: List[Any],
            desc: Optional[str] = None) -> List[Any]:
        """Apply func to each item in parallel."""
        raise NotImplementedError


class CPUBackend(ParallelBackend):
    """Multi-core CPU parallelization."""

    def __init__(self, num_cores: int = None):
        self.num_cores = num_cores or mp.cpu_count()

    def map(self, func: Callable, items: List[Any],
            desc: Optional[str] = None) -> List[Any]:
        if len(items) == 0:
            return []

        if self.num_cores == 1:
            # Sequential
            if desc:
                items = tqdm(items, desc=desc)
            return [func(item) for item in items]

        # Use spawn context for better compatibility
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.num_cores) as pool:
            if desc:
                results = []
                with tqdm(total=len(items), desc=desc) as pbar:
                    for result in pool.imap(func, items):
                        results.append(result)
                        pbar.update(1)
                return results
            else:
                return pool.map(func, items)


class GPUBackend(ParallelBackend):
    """GPU parallelization using CuPy."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        try:
            import cupy as cp
            self.cp = cp
            self.cp.cuda.Device(device_id).use()
            self.available = True
        except ImportError:
            print("CuPy not available, falling back to CPU")
            self.available = False

    def map(self, func: Callable, items: List[Any],
            desc: Optional[str] = None) -> List[Any]:
        # Simple fallback to CPU for now
        cpu = CPUBackend()
        return cpu.map(func, items, desc)


class HybridBackend(ParallelBackend):
    """Hybrid backend that chooses best available."""

    def __init__(self, config):
        self.config = config
        self.cpu = CPUBackend(config.num_cpu_cores)

        if config.use_gpu:
            self.gpu = GPUBackend(config.gpu_device_id)
        else:
            self.gpu = None

    def map(self, func: Callable, items: List[Any],
            desc: Optional[str] = None) -> List[Any]:
        # Simple: always use CPU for reliability
        return self.cpu.map(func, items, desc)