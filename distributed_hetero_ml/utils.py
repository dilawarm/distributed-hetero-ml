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
