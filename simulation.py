"""Simulation engine for closed-loop system."""
import numpy as np
from typing import List, Tuple, Optional, Dict
import random

from models import RobotModel
from abstraction import SymbolicModel
from synthesis import SymbolicController, ProductState
from automaton import Automaton


class Simulator:
    """Simulator for closed-loop system with symbolic controller."""

    def __init__(self, model: RobotModel, symbolic: SymbolicModel,
                 controller: SymbolicController, automaton: Optional[Automaton] = None):
        self.model = model
        self.symbolic = symbolic
        self.controller = controller
        self.automaton = automaton

        # For tracking product state
        self.current_auto_state = automaton.initial if automaton else None
        self.trajectory = []
        self.cell_trajectory = []
        self.auto_trajectory = []

    def _get_cell_index(self, cell) -> int:
        """Get index of a cell."""
        try:
            # Try to use the partition's cells list if available
            return self.symbolic.partition.cells.index(cell)
        except (AttributeError, ValueError):
            # Fallback: compute linear index from tuple index
            stride = 1
            idx = 0
            for d in range(len(cell.index) - 1, -1, -1):
                idx += cell.index[d] * stride
                stride *= self.symbolic.partition.resolutions[d]
            return idx

    def _find_closest_cell(self, point: np.ndarray):
        """Find closest cell when point is outside workspace."""
        best_cell = None
        best_dist = float('inf')

        # Iterate through all cells (may be slow for large partitions)
        # In practice, you might want a spatial index
        for cell in self.symbolic.partition:
            center = cell.center()
            # Only compare relevant dimensions
            dist = np.linalg.norm(point[:len(center)] - center[:len(point)])
            if dist < best_dist:
                best_dist = dist
                best_cell = cell

        return best_cell

    def reset(self, initial_continuous_state: np.ndarray):
        """Reset simulator to initial state."""
        self.continuous_state = initial_continuous_state.copy()
        self.trajectory = [initial_continuous_state.copy()]

        # Find initial cell
        cell = self.symbolic.partition.point_to_cell(initial_continuous_state)
        if cell is None:
            print(f"⚠️ Initial state {initial_continuous_state} outside workspace, finding closest cell")
            cell = self._find_closest_cell(initial_continuous_state)
            print(f"   Using cell with center: {cell.center()}")

        self.current_cell_idx = self._get_cell_index(cell)
        self.cell_trajectory = [self.current_cell_idx]

        if self.automaton:
            self.current_auto_state = self.automaton.initial
            self.auto_trajectory = [self.current_auto_state]

        print(f"  Reset complete: cell={self.current_cell_idx}, auto={self.current_auto_state}")

    def step(self, noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Take one simulation step.

        Returns:
            next_state, done (True if no valid control found)
        """
        # Get current product state
        if self.automaton:
            product_state = ProductState(self.current_auto_state, self.current_cell_idx)
        else:
            product_state = ProductState("", self.current_cell_idx)

        # Get allowed inputs
        allowed_inputs = self.controller.get_allowed_inputs(product_state)

        if not allowed_inputs:
            print(f"⚠️ No allowed inputs for state {product_state}")
            return self.continuous_state, True  # deadlock

        # Choose input (random for exploration)
        input_vec = random.choice(allowed_inputs)

        # Apply dynamics with noise
        if noise is None:
            noise = np.zeros(self.model.state_dim)

        next_state = self.model.dynamics(self.continuous_state, input_vec, noise)

        # Clip to bounds
        bounds = self.model.get_state_bounds()
        for d in range(len(next_state)):
            if d < len(bounds):
                next_state[d] = np.clip(next_state[d], bounds[d][0], bounds[d][1])

        # Update state
        self.continuous_state = next_state

        # Find new cell
        cell = self.symbolic.partition.point_to_cell(next_state)
        if cell is None:
            print(f"⚠️ State {next_state} outside workspace, finding closest cell")
            cell = self._find_closest_cell(next_state)

        self.current_cell_idx = self._get_cell_index(cell)

        # Update automaton state if needed
        if self.automaton:
            # Get observations from new cell
            labels = self.symbolic.get_labels(self.current_cell_idx)

            # Update automaton
            if labels:
                for label in labels:
                    next_auto = self.automaton.next(self.current_auto_state, label)
                    if next_auto:
                        self.current_auto_state = next_auto
                        break
            else:
                # Try empty observation
                next_auto = self.automaton.next(self.current_auto_state, "")
                if next_auto:
                    self.current_auto_state = next_auto

        # Record trajectory
        self.trajectory.append(next_state.copy())
        self.cell_trajectory.append(self.current_cell_idx)
        if self.automaton:
            self.auto_trajectory.append(self.current_auto_state)

        return next_state, False

    def simulate(self, num_steps: int, initial_state: np.ndarray,
                 noise_scale: float = 0.0) -> List[np.ndarray]:
        """
        Simulate for given number of steps.
        """
        self.reset(initial_state)
        print(f"🚀 Starting simulation from {initial_state} for {num_steps} steps")
        print(f"  Initial cell: {self.current_cell_idx}, auto: {self.current_auto_state}")

        for step in range(num_steps):
            # Generate noise if needed
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, self.model.state_dim)
            else:
                noise = None

            next_state, done = self.step(noise)

            # Print progress every 20 steps
            if step % 20 == 0 or step == num_steps - 1:
                print(f"  Step {step+1:3d}: pos=({next_state[0]:.2f}, {next_state[1]:.2f}), "
                      f"cell={self.current_cell_idx:4d}, auto={self.current_auto_state}")

            if done:
                print(f"⛔ Simulation stopped at step {step+1}: no valid control")
                break

        print(f"✅ Simulation complete: {len(self.trajectory)} points generated")
        if len(self.trajectory) > 0:
            print(f"  Start: {self.trajectory[0]}")
            print(f"  End:   {self.trajectory[-1]}")

        return self.trajectory

    def simulate_until_target(self, initial_state: np.ndarray,
                              target_auto_state: Optional[str] = None,
                              max_steps: int = 1000) -> Tuple[List[np.ndarray], bool]:
        """
        Simulate until reaching target automaton state or max steps.
        """
        self.reset(initial_state)
        print(f"🎯 Simulating until target state '{target_auto_state}' or {max_steps} steps")

        for step in range(max_steps):
            _, done = self.step()

            if step % 50 == 0:
                print(f"  Step {step}: auto={self.current_auto_state}")

            if done:
                print(f"⛔ Simulation stopped: no valid control")
                return self.trajectory, False

            if target_auto_state and self.current_auto_state == target_auto_state:
                print(f"✅ Target reached at step {step}!")
                return self.trajectory, True

        print(f"⏱️  Max steps ({max_steps}) reached without reaching target")
        return self.trajectory, False