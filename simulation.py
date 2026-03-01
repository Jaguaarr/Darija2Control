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

    def reset(self, initial_continuous_state: np.ndarray):
        """Reset simulator to initial state."""
        self.continuous_state = initial_continuous_state.copy()

        # Find initial cell
        cell = self.symbolic.partition.point_to_cell(initial_continuous_state)
        if cell is None:
            raise ValueError(f"Initial state {initial_continuous_state} outside workspace")

        self.current_cell_idx = self.symbolic.partition.cells.index(cell)

        if self.automaton:
            self.current_auto_state = self.automaton.initial

        self.trajectory = [initial_continuous_state.copy()]
        self.cell_trajectory = [self.current_cell_idx]
        self.auto_trajectory = [self.current_auto_state] if self.automaton else []

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
            # If no automaton, use only cell index
            product_state = ProductState("", self.current_cell_idx)

        # Get allowed inputs
        allowed_inputs = self.controller.get_allowed_inputs(product_state)
        if not allowed_inputs:
            return self.continuous_state, True  # deadlock

        # Choose input (random for exploration)
        input_vec = random.choice(allowed_inputs)

        # Apply dynamics with noise
        if noise is None:
            noise = np.zeros(self.model.state_dim)

        next_state = self.model.dynamics(self.continuous_state, input_vec, noise)

        # Update state
        self.continuous_state = next_state

        # Find new cell
        cell = self.symbolic.partition.point_to_cell(next_state)
        if cell is None:
            # Out of bounds
            return next_state, True

        self.current_cell_idx = self.symbolic.partition.cells.index(cell)

        # Update automaton state if needed
        if self.automaton:
            # Get observations from new cell
            labels = self.symbolic.get_labels(self.current_cell_idx)

            # Update automaton (nondeterministic - choose one)
            if labels:
                # Try each label until we find a valid transition
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
        print(f"Simulating for {num_steps} steps...")

        for step in range(num_steps):
            noise = np.random.normal(0, noise_scale, self.model.state_dim)
            _, done = self.step(noise)

            if step % 50 == 0:
                print(f"  Step {step}/{num_steps}")

            if done:
                print(f"Simulation stopped early at step {step}")
                break

        print(f"Simulation complete: {len(self.trajectory)} points")
        return self.trajectory

    def simulate_until_target(self, initial_state: np.ndarray,
                              target_auto_state: Optional[str] = None,
                              max_steps: int = 1000) -> Tuple[List[np.ndarray], bool]:
        """
        Simulate until reaching target automaton state or max steps.
        """
        self.reset(initial_state)

        for _ in range(max_steps):
            _, done = self.step()
            if done:
                return self.trajectory, False

            if target_auto_state and self.current_auto_state == target_auto_state:
                return self.trajectory, True

        return self.trajectory, False