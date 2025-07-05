import logging
import sys

import torch
from torch import nn, optim

from distributed_hetero_ml.types import TrainingConfig


class SimpleModelFactory:
    """
    Simple model factory for testing and examples.

    Creates a basic feedforward neural network with configurable dimensions.

    Args:
        input_size: Number of input features (default: 784, typical for MNIST)
        hidden_size: Number of hidden units (default: 128)
        output_size: Number of output classes (default: 10)

    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def create_model(self) -> nn.Module:
        """Create a simple feedforward neural network."""
        return nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.output_size))

    def create_optimizer(self, model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        """Create SGD optimizer with configurable learning rate."""
        return optim.SGD(model.parameters(), lr=config.learning_rate)

    def create_criterion(self) -> nn.Module:
        """Create cross-entropy loss function for classification."""
        return nn.CrossEntropyLoss()


class SyntheticDataLoader:
    """
    Synthetic data loader for testing and examples.

    Generates random data for training. Useful for testing the framework
    without needing real datasets.

    Args:
        input_size: Number of input features (default: 784)
        num_classes: Number of output classes (default: 10)
        dataset_size: Total size of synthetic dataset (default: 1000)

    """

    def __init__(self, input_size: int = 784, num_classes: int = 10, dataset_size: int = 1000):
        self.input_size = input_size
        self.num_classes = num_classes
        self.dataset_size = dataset_size

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random batch of data.

        Args:
            batch_size: Number of samples in the batch
            device: Device to place the tensors on

        Returns:
            Tuple of (features, labels) tensors

        """
        X = torch.randn(batch_size, self.input_size, device=device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=device)
        return X, y

    def get_dataset_size(self) -> int:
        """Return the total dataset size."""
        return self.dataset_size


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Set up logging configuration for distributed training.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs to stdout only.

    """
    # Configure logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for distributed_hetero_ml
    distributed_logger = logging.getLogger("distributed_hetero_ml")
    distributed_logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance

    """
    return logging.getLogger(name)
