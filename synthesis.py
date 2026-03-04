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

        # Precompute successor relation for product (using your working approach)
        print("🔄 Precomputing product successors...")
        self.product_succ = self._precompute_product_successors()
        print(f"✅ Product successors precomputed")

    def _precompute_product_successors(self) -> Dict[int, Dict[int, Set[int]]]:
        """Precompute successors for all product states."""
        n_states = self.product.n_states
        n_inputs = self.symbolic.n_inputs

        succ = {s_idx: {u_idx: set() for u_idx in range(n_inputs)}
                for s_idx in range(n_states)}

        # Your working approach: iterate through all states and inputs
        for s_idx in range(n_states):
            if s_idx % 100 == 0:  # Progress indicator
                print(f"  Computing successors for state {s_idx}/{n_states}")

            for u_idx in range(n_inputs):
                succ[s_idx][u_idx] = self.product.get_successors(s_idx, u_idx)

        return succ

    def Pre_fast(self, R_idx_set: Set[int]) -> List[int]:
        """
        YOUR working Pre_fast function adapted.
        R_idx_set: set of state indices that are in R
        returns: list of state indices that are predecessors of R
        """
        res = []
        for s_idx in range(self.product.n_states):
            # Try each input; if at least ONE input has all successors in R, s_idx is in Pre(R)
            for u_idx in range(self.symbolic.n_inputs):
                succ_set = self.product_succ.get(s_idx, {}).get(u_idx, set())
                if not succ_set:
                    continue  # no successors for this input
                # Check if ALL successors are inside R
                if succ_set.issubset(R_idx_set):
                    res.append(s_idx)
                    break  # no need to check other inputs for this state
        return res

    def pointFixe_fast(self, Q_states: Set[ProductState]) -> List[ProductState]:
        """
        YOUR working pointFixe_fast function adapted for safety.
        Q_states: set of product states that are safe
        returns: list of product states belonging to the fixed point.
        """
        # Convert to indices
        Q_idx = {self.product.state_to_idx[s] for s in Q_states}

        # Fixed point: R_{k+1} = Pre(R_k) ∩ Q
        R0 = set(Q_idx)
        iteration = 0

        while True:
            iteration += 1
            pre = self.Pre_fast(R0)
            R1 = {i for i in pre if i in Q_idx}
            print(f"  Iteration {iteration}: |R| = {len(R1)}")

            if R1 == R0:
                break
            R0 = R1

        # Decode back to ProductState objects
        return [self.product.states[i] for i in R0]

    def synthesize_safety(self, safe_set: Set[ProductState]) -> SymbolicController:
        """
        Synthesize controller for safety specification using your working method.
        """
        print(f"🛡️ Synthesizing safety controller for {len(safe_set)} safe states...")

        # Get fixed point using your algorithm
        winning_states = self.pointFixe_fast(safe_set)
        winning_indices = {self.product.state_to_idx[s] for s in winning_states}

        print(f"✅ Safety synthesis complete: {len(winning_states)} winning states")

        # Build controller
        controller = SymbolicController()

        for state in winning_states:
            s_idx = self.product.state_to_idx[state]
            allowed_inputs = []

            for u_idx, inp in enumerate(self.symbolic.inputs):
                succ_set = self.product_succ.get(s_idx, {}).get(u_idx, set())
                if not succ_set:
                    continue
                # Check condition: all successors in winning set
                if succ_set.issubset(winning_indices):
                    allowed_inputs.append(inp)

            if allowed_inputs:
                controller.add_control(state, allowed_inputs)

        controller.winning_states = set(winning_states)
        return controller

    def pointFixeAtteignabilité_fast(self, Q_states: Set[ProductState]) -> Dict[ProductState, List[np.ndarray]]:
        """
        YOUR working pointFixeAtteignabilité_fast function adapted for reachability.
        Q_states: set of product states that we want to reach (target set)
        Returns: controller dict {state: [allowed_inputs]}
        """
        # Convert to indices
        Q_idx = {self.product.state_to_idx[s] for s in Q_states}

        # controller indexed by state index internally
        controlleur_idx = {}

        # INITIALIZATION
        R0 = set(Q_idx)

        # Pre(R0)
        pre0 = self.Pre_fast(R0)

        # R1 = Q ∪ Pre(Q)
        R1 = R0.union(pre0)

        # Initialize controller for all states in R1 \ Q
        for s_idx in R1:
            if s_idx not in Q_idx:  # only states that are not targets
                if s_idx not in controlleur_idx:
                    controlleur_idx[s_idx] = []
                for u_idx, inp in enumerate(self.symbolic.inputs):
                    succ_set = self.product_succ.get(s_idx, {}).get(u_idx, set())
                    if succ_set and succ_set.issubset(R0):
                        controlleur_idx[s_idx].append(inp)

        # FIXED-POINT LOOP
        iteration = 1
        while R1 != R0:
            iteration += 1
            R0 = set(R1)

            # Pre(R0)
            pre0 = self.Pre_fast(R0)

            # R1 = Q ∪ Pre(prev)
            R1 = R0.union(pre0)

            # For new states in R1 \ Q, add controller entries
            for s_idx in R1:
                if s_idx not in Q_idx and s_idx not in controlleur_idx:
                    controlleur_idx[s_idx] = []
                    for u_idx, inp in enumerate(self.symbolic.inputs):
                        succ_set = self.product_succ.get(s_idx, {}).get(u_idx, set())
                        if succ_set and succ_set.issubset(R0):
                            controlleur_idx[s_idx].append(inp)

            print(f"  Iteration {iteration}: |R| = {len(R1)}")

        # Convert state indices back to ProductState objects
        controlleur = {
            self.product.states[s_idx]: inputs_list
            for s_idx, inputs_list in controlleur_idx.items()
        }

        return controlleur

    def synthesize_reachability(self, target_set: Set[ProductState]) -> SymbolicController:
        """
        Synthesize controller for reachability specification using your working method.
        """
        print(f"🎯 Synthesizing reachability controller for {len(target_set)} target states...")

        # Get controller using your algorithm
        controlleur_dict = self.pointFixeAtteignabilité_fast(target_set)

        # Build controller object
        controller = SymbolicController()
        for state, inputs in controlleur_dict.items():
            if inputs:
                controller.add_control(state, inputs)

        # Winning states are all states in the controller
        controller.winning_states = set(controlleur_dict.keys())

        print(f"✅ Reachability synthesis complete: {len(controller.winning_states)} winning states")
        return controller

    def synthesize_automaton(self) -> SymbolicController:
        """
        Synthesize controller for general automaton specification.
        This uses reachability to accepting states.
        """
        print("🤖 Synthesizing controller for automaton specification...")

        # Find accepting states in the product
        accepting_states = set()
        for state in self.product.states:
            if state.auto_state in self.product.automaton.accepting:
                accepting_states.add(state)

        if not accepting_states:
            print("⚠️ No accepting states found, returning empty controller")
            return SymbolicController()

        print(f"🎯 Found {len(accepting_states)} accepting states")

        # Synthesize reachability controller to accepting states
        return self.synthesize_reachability(accepting_states)

    def _pre_serial(self, state_indices: Set[int]) -> Set[int]:
        """Fallback serial Pre computation."""
        return set(self.Pre_fast(state_indices))