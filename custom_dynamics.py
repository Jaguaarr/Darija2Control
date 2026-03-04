"""Custom dynamics functions for user-defined models."""
import numpy as np
from typing import List, Callable
import math


class CustomDynamics:
    """Container for custom dynamics functions that can be pickled."""

    def __init__(self, equations: List[str], state_dim: int, input_dim: int):
        self.equations = equations
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.functions = self._compile_equations()

    def _compile_equations(self):
        """Compile equations into callable functions."""
        functions = []
        for eq in self.equations:
            # Create a function that evaluates the expression
            func = self._create_function(eq)
            functions.append(func)
        return functions

    def _create_function(self, equation: str):
        """Create a callable function from an equation string."""
        # Safe namespace with math functions
        namespace = {
            'np': np, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'pi': np.pi, 'abs': abs, 'math': math
        }

        # Compile the expression
        code = compile(equation, '<string>', 'eval')

        def func(x, u):
            # Update namespace with current values
            local_namespace = namespace.copy()
            for i in range(len(x)):
                local_namespace[f'x{i}'] = x[i]
            for i in range(len(u)):
                local_namespace[f'u{i}'] = u[i]

            return eval(code, {"__builtins__": {}}, local_namespace)

        return func

    def __call__(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Evaluate all equations."""
        x_next = np.zeros(self.state_dim)
        for i, func in enumerate(self.functions):
            x_next[i] = func(x, u) + w[i]
        return x_next

    def __getstate__(self):
        """For pickling - store only the equations, not the functions."""
        return {
            'equations': self.equations,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim
        }

    def __setstate__(self, state):
        """For unpickling - recompile functions."""
        self.equations = state['equations']
        self.state_dim = state['state_dim']
        self.input_dim = state['input_dim']
        self.functions = self._compile_equations()