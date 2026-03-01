"""Configuration management for the symbolic control framework."""
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class HardwareConfig:
    """Hardware configuration for parallelization."""
    num_cpu_cores: int = 4
    use_gpu: bool = False
    gpu_device_id: int = 0


@dataclass
class PartitionConfig:
    """Configuration for state space partitioning."""
    resolution: Dict[str, int] = None  # dimension -> number of cells
    custom_intervals: Dict[str, list] = None  # dimension -> list of intervals


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    provider: str = "openai"  # or "anthropic", "local"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.1


class Config:
    """Main configuration class."""

    def __init__(self, config_file: Optional[str] = None):
        self.hardware = HardwareConfig()
        self.partition = PartitionConfig()
        self.llm = LLMConfig()
        self.workspace_bounds: Dict[str, tuple] = {}  # dimension -> (min, max)

        if config_file and os.path.exists(config_file):
            self.load(config_file)

    def load(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            data = json.load(f)

        if 'hardware' in data:
            self.hardware = HardwareConfig(**data['hardware'])
        if 'partition' in data:
            self.partition = PartitionConfig(**data['partition'])
        if 'llm' in data:
            self.llm = LLMConfig(**data['llm'])
        if 'workspace_bounds' in data:
            self.workspace_bounds = data['workspace_bounds']

    def save(self, config_file: str):
        """Save configuration to JSON file."""
        data = {
            'hardware': {
                'num_cpu_cores': self.hardware.num_cpu_cores,
                'use_gpu': self.hardware.use_gpu,
                'gpu_device_id': self.hardware.gpu_device_id
            },
            'partition': {
                'resolution': self.partition.resolution,
                'custom_intervals': self.partition.custom_intervals
            },
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'temperature': self.llm.temperature
            },
            'workspace_bounds': self.workspace_bounds
        }

        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)