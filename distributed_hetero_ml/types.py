from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from torch import nn


class DeviceType(Enum):
    """Supported device types for training."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class DeviceInfo:
    """Information about a training device."""

    device_type: DeviceType
    device_name: str
    hostname: str
    device_id: int | None = None
    memory_gb: int | None = None


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""

    batch_size: int = 64
    learning_rate: float = 0.01
    num_epochs: int = 10
    gradient_clipping: float | None = None
    use_mixed_precision: bool = False
    checkpoint_interval: int = 100

    num_gpu_workers: int = 2
    num_cpu_workers: int = 1
    cpus_per_worker: int = 2
    gpus_per_worker: int = 1


@dataclass
class TrainingResult:
    """Results from a training step."""

    worker_id: int
    device_info: DeviceInfo
    loss: float
    gradients: dict[str, torch.Tensor]
    metrics: dict[str, float]
    step_time: float


class ParameterServer(Protocol):
    """Protocol for parameter server coordination."""

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Get current model weights."""
        ...

    def update_weights(self, gradients_list: list[dict[str, torch.Tensor]]) -> int:
        """Update model weights with averaged gradients."""
        ...

    def get_iteration(self) -> int:
        """Get current training iteration."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        ...


class TrainingWorker(Protocol):
    """Protocol for training workers."""

    def train_step(self, parameter_server: ParameterServer) -> TrainingResult:
        """Execute a single training step."""
        ...

    def get_device_info(self) -> DeviceInfo:
        """Get device information."""
        ...


class ModelFactory(Protocol):
    """Protocol for model creation."""

    def create_model(self) -> nn.Module:
        """Create a new model instance."""
        ...

    def create_optimizer(self, model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        """Create optimizer for the model."""
        ...

    def create_criterion(self) -> nn.Module:
        """Create loss function."""
        ...


class DataLoader(Protocol):
    """Protocol for data loading."""

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of training data."""
        ...

    def get_dataset_size(self) -> int:
        """Get total dataset size."""
        ...
