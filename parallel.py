"""Simple but efficient parallelization backends for CPU and GPU."""
import numpy as np
from typing import Callable, List, Any, Optional, Set, Dict
import multiprocessing as mp
from tqdm import tqdm
import time


class ParallelBackend:
    """Base class for parallelization backends."""

    def map(self, func: Callable, items: List[Any], desc: Optional[str] = None) -> List[Any]:
        """Apply function to all items in parallel."""
        raise NotImplementedError


class CPUBackend(ParallelBackend):
    """Multi-core CPU parallelization using multiprocessing."""

    def __init__(self, num_cores: int = None):
        self.num_cores = num_cores or mp.cpu_count()
        print(f"🖥️  CPU Backend: {self.num_cores} cores")

    def map(self, func: Callable, items: List[Any], desc: Optional[str] = None) -> List[Any]:
        if not items:
            return []

        # For small tasks, do sequential
        if len(items) < 1000 or self.num_cores == 1:
            if desc:
                items = tqdm(items, desc=desc)
            return [func(item) for item in items]

        # Parallel processing
        start_time = time.time()

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

        elapsed = time.time() - start_time
        print(f"⚡ CPU Speed: {len(items)/elapsed:.0f} items/sec ({elapsed*1000:.1f}ms)")
        return results


class GPUBackend(ParallelBackend):
    """GPU parallelization using CuPy - only for vectorized operations."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.available = False

        try:
            import cupy as cp
            self.cp = cp
            self.available = True
            # Get GPU info
            gpu_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()
            print(f"🎮 GPU Backend: {gpu_name}")
        except ImportError:
            print("⚠️ CuPy not installed. GPU acceleration disabled.")
        except Exception as e:
            print(f"⚠️ GPU init failed: {e}")

    def map(self, func: Callable, items: List[Any], desc: Optional[str] = None) -> List[Any]:
        """GPU-accelerated map - only works with vectorized functions."""
        if not self.available:
            print("⚠️ GPU not available, falling back to CPU")
            return CPUBackend().map(func, items, desc)

        # Check if function is vectorized for GPU
        if not hasattr(func, 'vectorized') or not func.vectorized:
            print("⚠️ Function not vectorized, falling back to CPU")
            return CPUBackend().map(func, items, desc)

        start_time = time.time()

        try:
            # Convert to GPU array
            gpu_items = self.cp.array(items)

            # Apply function on GPU
            gpu_results = func(gpu_items)

            # Get back to CPU
            results = self.cp.asnumpy(gpu_results).tolist()

            elapsed = time.time() - start_time
            print(f"⚡ GPU Speed: {len(items)/elapsed:.0f} items/sec ({elapsed*1000:.1f}ms)")

            return results

        except Exception as e:
            print(f"⚠️ GPU error: {e}, falling back to CPU")
            return CPUBackend().map(func, items, desc)


class HybridBackend(ParallelBackend):
    """Smart backend that chooses CPU or GPU based on task size."""

    def __init__(self, config):
        self.cpu = CPUBackend(config.num_cpu_cores)
        self.gpu = GPUBackend(config.gpu_device_id) if config.use_gpu else None
        self.gpu_threshold = 10000  # Use GPU for >10k items

        print(f"🔀 Hybrid Backend: CPU ({config.num_cpu_cores} cores)" +
              (f" + GPU" if config.use_gpu else ""))

    def map(self, func: Callable, items: List[Any], desc: Optional[str] = None) -> List[Any]:
        if not items:
            return []

        # Decide which backend to use
        use_gpu = (self.gpu and self.gpu.available and
                  len(items) > self.gpu_threshold and
                  hasattr(func, 'vectorized') and func.vectorized)

        if use_gpu:
            return self.gpu.map(func, items, desc)
        else:
            return self.cpu.map(func, items, desc)