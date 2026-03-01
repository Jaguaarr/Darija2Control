"""Robot models library for N-dimensional dynamics."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, List, Tuple, Dict, Any


class RobotModel(ABC):
    """Abstract base class for robot models."""

    def __init__(self, name: str, state_dim: int, input_dim: int):
        self.name = name
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.state_names = [f"x{i + 1}" for i in range(state_dim)]
        self.input_names = [f"u{i + 1}" for i in range(input_dim)]

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Continuous dynamics: x_next = f(x, u, w)"""
        pass

    @abstractmethod
    def get_state_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for each state dimension."""
        pass

    @abstractmethod
    def get_inputs(self) -> List[np.ndarray]:
        """Return list of discrete input vectors."""
        pass

    @abstractmethod
    def get_disturbance_bounds(self) -> np.ndarray:
        """Return bounds for disturbance (symmetric) per dimension."""
        pass

    def jacobian(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optional: analytical Jacobians for better over-approximation."""
        # Default: numerical approximation
        eps = 1e-6
        n = self.state_dim
        m = self.input_dim

        Jx = np.zeros((n, n))
        Ju = np.zeros((n, m))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            Jx[:, i] = (self.dynamics(x_plus, u, np.zeros(n)) -
                        self.dynamics(x_minus, u, np.zeros(n))) / (2 * eps)

        for i in range(m):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            Ju[:, i] = (self.dynamics(x, u_plus, np.zeros(n)) -
                        self.dynamics(x, u_minus, np.zeros(n))) / (2 * eps)

        return Jx, Ju


class DifferentialDrive(RobotModel):
    """2D differential drive robot."""

    def __init__(self):
        super().__init__("differential_drive", state_dim=3, input_dim=2)

    def dynamics(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """x = [x, y, theta], u = [v, omega]"""
        x_next = np.array([
            x[0] + u[0] * np.cos(x[2]) + w[0],
            x[1] + u[0] * np.sin(x[2]) + w[1],
            (x[2] + u[1] + w[2]) % (2 * np.pi)
        ])
        return x_next

    def get_state_bounds(self) -> List[Tuple[float, float]]:
        return [(0, 10), (0, 10), (0, 2 * np.pi)]

    def get_inputs(self) -> List[np.ndarray]:
        v_vals = [0.25, 0.625, 1.0]
        omega_vals = [-1.0, -0.5, 0, 0.5, 1.0]
        return [np.array([v, omega]) for v in v_vals for omega in omega_vals]

    def get_disturbance_bounds(self) -> np.ndarray:
        return np.array([0.05, 0.05, 0.05])


class ArmRobot(RobotModel):
    """N-DOF robotic arm."""

    def __init__(self, num_joints: int = 2):
        super().__init__(f"{num_joints}DOF_arm", state_dim=num_joints * 2, input_dim=num_joints)
        self.num_joints = num_joints

    def dynamics(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """x = [q1, q2, ..., qN, q1_dot, q2_dot, ..., qN_dot]"""
        dt = 0.1  # time step
        n = self.num_joints

        # Simple double integrator dynamics
        q = x[:n]
        q_dot = x[n:]

        q_next = q + q_dot * dt + w[:n]
        q_dot_next = q_dot + u * dt + w[n:]

        return np.concatenate([q_next, q_dot_next])

    def get_state_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        # Joint angle bounds
        for i in range(self.num_joints):
            bounds.append((-np.pi, np.pi))
        # Joint velocity bounds
        for i in range(self.num_joints):
            bounds.append((-2.0, 2.0))
        return bounds

    def get_inputs(self) -> List[np.ndarray]:
        # Discretized acceleration commands
        acc_vals = [-1.0, -0.5, 0, 0.5, 1.0]
        inputs = []
        for i in range(len(acc_vals) ** self.num_joints):
            idx = i
            u = np.zeros(self.num_joints)
            for j in range(self.num_joints):
                u[j] = acc_vals[idx % len(acc_vals)]
                idx //= len(acc_vals)
            inputs.append(u)
        return inputs

    def get_disturbance_bounds(self) -> np.ndarray:
        return np.array([0.01] * (2 * self.num_joints))


# Model registry
MODEL_REGISTRY = {
    "differential_drive": DifferentialDrive,
    "2DOF_arm": lambda: ArmRobot(2),
    "3DOF_arm": lambda: ArmRobot(3),
}