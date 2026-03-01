"""Controller synthesis algorithms for symbolic control."""
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import deque

from automaton import Automaton, ProductSystem, ProductState
from abstraction import SymbolicModel
from parallel import ParallelBackend


class SymbolicController:
    """Symbolic controller mapping states to allowed inputs."""

    def __init__(self):
        self.controller: Dict[ProductState, List[np.ndarray]] = {}
        self.winning_states: Set[ProductState] = set()

    def get_allowed_inputs(self, state: ProductState) -> List[np.ndarray]:
        """Get allowed inputs for given product state."""
        return self.controller.get(state, [])

    def add_control(self, state: ProductState, inputs: List[np.ndarray]):
        """Add control mapping."""
        self.controller[state] = inputs
        self.winning_states.add(state)


class SynthesisEngine:
    """Synthesis algorithms for symbolic control."""

    def __init__(self, product: ProductSystem, symbolic: SymbolicModel,
                 parallel: Optional[ParallelBackend] = None):
        self.product = product
        self.symbolic = symbolic
        self.parallel = parallel or ParallelBackend()

        # Precompute successor relation for product (SERIAL - no pickling issues)
        print("🔄 Precomputing product successors...")
        self.product_succ = self._precompute_product_successors_serial()
        print(f"✅ Product successors precomputed for {len(self.product_succ)} states")

    def _precompute_product_successors_serial(self) -> Dict[int, Dict[int, Set[int]]]:
        """
        Serial version of product successor precomputation.
        Avoids pickling issues entirely.
        """
        n_states = self.product.n_states
        n_inputs = self.symbolic.n_inputs

        succ = {s_idx: {u_idx: set() for u_idx in range(n_inputs)}
                for s_idx in range(n_states)}

        for s_idx in range(n_states):
            if s_idx % 100 == 0:  # Progress indicator
                print(f"  Computing successors for state {s_idx}/{n_states}")

            for u_idx in range(n_inputs):
                succ[s_idx][u_idx] = self.product.get_successors(s_idx, u_idx)

        return succ

    def synthesize_safety(self, safe_set: Set[ProductState]) -> SymbolicController:
        """
        Synthesize controller for safety specification.

        Args:
            safe_set: Set of product states that are safe

        Returns:
            Controller that keeps system within safe_set
        """
        print(f"🛡️ Synthesizing safety controller for {len(safe_set)} safe states...")

        # Convert to indices
        safe_indices = {self.product.state_to_idx[s] for s in safe_set}

        # Start with all safe states
        R = set(safe_indices)
        iteration = 0

        while True:
            iteration += 1
            # Pre(R) = states that can force staying in R for one step
            pre = set()

            for s_idx in range(self.product.n_states):
                # Skip if not in safe set
                if s_idx not in safe_indices:
                    continue

                # Check each input
                for u_idx in range(self.symbolic.n_inputs):
                    succs = self.product_succ.get(s_idx, {}).get(u_idx, set())
                    if succs and all(s in R for s in succs):
                        pre.add(s_idx)
                        break

            # New R = previous R ∩ Pre(R)
            R_next = R.intersection(pre)

            print(f"  Iteration {iteration}: |R| = {len(R_next)}")

            if R_next == R:
                break
            R = R_next

            if len(R) == 0:
                print("  ⚠️ No winning states found!")
                break

        # Build controller
        controller = SymbolicController()
        winning_states = {self.product.states[idx] for idx in R}

        print(f"✅ Safety synthesis complete: {len(winning_states)} winning states")

        for state in winning_states:
            s_idx = self.product.state_to_idx[state]
            allowed = []

            for u_idx, inp in enumerate(self.symbolic.inputs):
                succs = self.product_succ[s_idx][u_idx]
                if succs and all(s in R for s in succs):
                    allowed.append(inp)

            if allowed:
                controller.add_control(state, allowed)

        controller.winning_states = winning_states
        return controller

    def synthesize_reachability(self, target_set: Set[ProductState]) -> SymbolicController:
        """
        Synthesize controller for reachability specification.

        Args:
            target_set: Set of product states to reach

        Returns:
            Controller that forces reaching target_set
        """
        print(f"🎯 Synthesizing reachability controller for {len(target_set)} target states...")

        # Convert to indices
        target_indices = {self.product.state_to_idx[s] for s in target_set}

        # R = set of states that can reach target
        R = set(target_indices)
        controller_dict = {}
        iteration = 0

        # Initialize with target states
        for idx in target_indices:
            state = self.product.states[idx]
            controller_dict[state] = []

        # Fixed point: add states that can force reaching R
        while True:
            iteration += 1
            new_states = set()

            # Find predecessors that can force reaching current R
            for s_idx in range(self.product.n_states):
                if s_idx in R:
                    continue

                for u_idx in range(self.symbolic.n_inputs):
                    succs = self.product_succ.get(s_idx, {}).get(u_idx, set())
                    if succs and all(s in R for s in succs):
                        new_states.add(s_idx)
                        # Store controller for this state
                        state = self.product.states[s_idx]
                        if state not in controller_dict:
                            controller_dict[state] = []
                        controller_dict[state].append(self.symbolic.inputs[u_idx])
                        break

            if not new_states:
                break

            R.update(new_states)
            print(f"  Iteration {iteration}: added {len(new_states)} states, total |R| = {len(R)}")

        # Build controller object
        controller = SymbolicController()
        for state, inputs in controller_dict.items():
            if inputs:  # Only add if there are allowed inputs
                controller.add_control(state, inputs)

        controller.winning_states = {self.product.states[idx] for idx in R}
        print(f"✅ Reachability synthesis complete: {len(controller.winning_states)} winning states")

        return controller

    def _pre_serial(self, state_indices: Set[int]) -> Set[int]:
        """
        Serial version of Pre computation.
        No pickling issues, guaranteed to work.
        """
        if not state_indices:
            return set()

        pre = set()
        n_states = self.product.n_states

        for s_idx in range(n_states):
            for u_idx in range(self.symbolic.n_inputs):
                succs = self.product_succ.get(s_idx, {}).get(u_idx, set())
                if succs and all(s in state_indices for s in succs):
                    pre.add(s_idx)
                    break

        return pre

    def synthesize_automaton(self) -> SymbolicController:
        """
        Synthesize controller for general automaton specification.
        """
        print("🤖 Synthesizing controller for automaton specification...")

        # Step 1: Find winning region for Buchi condition
        accepting_indices = {
            idx for idx, state in enumerate(self.product.states)
            if self.product.is_accepting(idx)
        }

        if not accepting_indices:
            print("⚠️ No accepting states found, returning empty controller")
            return SymbolicController()

        # Compute attractor of accepting states (serial version)
        attractor = self._attractor_serial(accepting_indices)

        # Build controller for attractor
        controller = SymbolicController()

        for idx in attractor:
            state = self.product.states[idx]
            allowed = []

            for u_idx, inp in enumerate(self.symbolic.inputs):
                succs = self.product_succ[idx][u_idx]
                if succs and all(s in attractor for s in succs):
                    allowed.append(inp)

            if allowed:
                controller.add_control(state, allowed)

        controller.winning_states = {self.product.states[idx] for idx in attractor}
        print(f"✅ Automaton synthesis complete: {len(controller.winning_states)} winning states")

        return controller

    def _attractor_serial(self, target_indices: Set[int]) -> Set[int]:
        """
        Serial version of attractor computation.
        """
        attractor = set(target_indices)
        queue = deque(target_indices)
        iteration = 0

        while queue:
            iteration += 1
            idx = queue.popleft()

            # Find predecessors that can force reaching attractor
            for pred_idx in range(self.product.n_states):
                if pred_idx in attractor:
                    continue

                for u_idx in range(self.symbolic.n_inputs):
                    succs = self.product_succ.get(pred_idx, {}).get(u_idx, set())
                    if succs and all(s in attractor for s in succs):
                        attractor.add(pred_idx)
                        queue.append(pred_idx)
                        break

            if iteration % 100 == 0:
                print(f"  Attractor iteration {iteration}: |attractor| = {len(attractor)}")

        return attractor