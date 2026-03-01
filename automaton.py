"""Automaton representation for specifications."""
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class Automaton:
    """Finite state automaton for specifications."""
    states: Set[str]
    initial: str
    accepting: Set[str]
    transitions: Dict[str, Dict[str, str]]  # state -> observation -> next_state

    def __post_init__(self):
        if self.initial not in self.states:
            raise ValueError(f"Initial state {self.initial} not in states")
        if not self.accepting.issubset(self.states):
            raise ValueError(f"Accepting states {self.accepting} not subset of states")

    def next(self, state: str, observation: str) -> Optional[str]:
        """Get next state given current state and observation."""
        if state not in self.transitions:
            return None
        return self.transitions[state].get(observation)

    def get_enabled_observations(self, state: str) -> Set[str]:
        """Get observations that enable a transition from given state."""
        if state not in self.transitions:
            return set()
        return set(self.transitions[state].keys())

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "states": list(self.states),
            "initial": self.initial,
            "accepting": list(self.accepting),
            "transitions": self.transitions
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Automaton':
        """Create from JSON dict."""
        return cls(
            states=set(data["states"]),
            initial=data["initial"],
            accepting=set(data["accepting"]),
            transitions=data["transitions"]
        )

    @classmethod
    def from_prompt(cls, prompt: str, region_names: List[str],
                    llm_config: dict = None) -> 'Automaton':
        """
        Generate automaton from natural language prompt using LLM.
        This is a placeholder - actual implementation in llm_integration.py
        """
        from llm_integration import prompt_to_automaton
        return prompt_to_automaton(prompt, region_names, llm_config)


class ProductState:
    """Product of automaton state and cell index."""

    def __init__(self, auto_state: str, cell_idx: int):
        self.auto_state = auto_state
        self.cell_idx = cell_idx

    def __hash__(self):
        return hash((self.auto_state, self.cell_idx))

    def __eq__(self, other):
        return (self.auto_state == other.auto_state and
                self.cell_idx == other.cell_idx)

    def __repr__(self):
        return f"({self.auto_state}, {self.cell_idx})"


class ProductSystem:
    """Product of automaton and symbolic model."""

    def __init__(self, automaton: Automaton, symbolic_model):
        self.automaton = automaton
        self.symbolic = symbolic_model

        # Precompute all product states
        self.states = []
        self.state_to_idx = {}

        for auto_state in automaton.states:
            for cell_idx in range(symbolic_model.n_cells):
                ps = ProductState(auto_state, cell_idx)
                idx = len(self.states)
                self.states.append(ps)
                self.state_to_idx[ps] = idx

        self.n_states = len(self.states)

    def get_successors(self, state_idx: int, input_idx: int) -> Set[int]:
        """
        Get successor product states for given state and input.
        Considers all possible observations from successor cells.
        """
        state = self.states[state_idx]

        # Get concrete successors of this cell
        cell_succs = self.symbolic.get_successors(state.cell_idx, input_idx)
        if not cell_succs:
            return set()

        # For each successor cell, get its labels and follow automaton transitions
        successors = set()
        for succ_cell in cell_succs:
            labels = self.symbolic.get_labels(succ_cell)

            # If cell has multiple labels, we need to consider all possible
            # observations (nondeterministic)
            if not labels:
                # No labels: check if automaton has transition on "true" or empty
                next_auto = self.automaton.next(state.auto_state, "")
                if next_auto:
                    successors.add(self.state_to_idx[ProductState(next_auto, succ_cell)])
            else:
                for label in labels:
                    next_auto = self.automaton.next(state.auto_state, label)
                    if next_auto:
                        successors.add(self.state_to_idx[ProductState(next_auto, succ_cell)])

        return successors

    def get_labels(self, state_idx: int) -> Set[str]:
        """Get observations true in this product state."""
        state = self.states[state_idx]
        return self.symbolic.get_labels(state.cell_idx)

    def is_accepting(self, state_idx: int) -> bool:
        """Check if product state is accepting."""
        state = self.states[state_idx]
        return state.auto_state in self.automaton.accepting