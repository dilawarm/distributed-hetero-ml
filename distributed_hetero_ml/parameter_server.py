import time
from typing import Any

import ray
import torch

from distributed_hetero_ml.types import ModelFactory, TrainingConfig


@ray.remote(num_cpus=1)
class RayParameterServer:
    """
    Ray-based parameter server for coordinating distributed training.

    Manages model weights, gradient aggregation, and optimization.
    """

    def __init__(self, model_factory: ModelFactory, config: TrainingConfig):
        """
        Initialize the parameter server.

        Args:
            model_factory: Factory for creating model components
            config: Training configuration

        """
        self.model = model_factory.create_model()
        self.optimizer = model_factory.create_optimizer(self.model, config)
        self.config = config
        self.iteration = 0
        self.training_history: list[dict[str, Any]] = []
        self.start_time = time.time()

        self.model = self.model.cpu()

        self.gradient_clipping = config.gradient_clipping

    def get_weights(self) -> dict[str, torch.Tensor]:
        """
        Get current model weights.

        Returns:
            dict[str, torch.Tensor]: Current model parameters

        """
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def update_weights(self, gradients_list: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """
        Update model weights with averaged gradients.

        Args:
            gradients_list: List of gradients from workers

        Returns:
            dict[str, float]: Training metrics

        """
        if not gradients_list:
            return {"iteration": self.iteration, "num_workers": 0}

        avg_gradients = self._average_gradients(gradients_list)

        self._apply_gradients(avg_gradients)
        self.iteration += 1
        metrics = self._calculate_metrics(gradients_list)

        self.training_history.append({"iteration": self.iteration, "timestamp": time.time() - self.start_time, "num_workers": len(gradients_list), **metrics})

        return metrics

    def _average_gradients(self, gradients_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Average gradients from multiple workers."""
        if not gradients_list:
            return {}

        avg_gradients = {}

        param_names = gradients_list[0].keys()

        for name in param_names:
            gradients = torch.stack([grads[name] for grads in gradients_list])
            avg_gradients[name] = gradients.mean(dim=0)

        return avg_gradients

    def _apply_gradients(self, gradients: dict[str, torch.Tensor]) -> None:
        """Apply averaged gradients to model."""
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name]

        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def _calculate_metrics(self, gradients_list: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Calculate training metrics."""
        num_workers = len(gradients_list)

        total_grad_norm = 0.0
        param_count = 0

        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm**2
                param_count += 1

        avg_grad_norm = (total_grad_norm / param_count) ** 0.5 if param_count > 0 else 0.0

        return {
            "iteration": self.iteration,
            "num_workers": num_workers,
            "avg_grad_norm": avg_grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "elapsed_time": time.time() - self.start_time,
        }

    def get_iteration(self) -> int:
        """Get current training iteration."""
        return self.iteration

    def get_training_history(self) -> list[dict[str, Any]]:
        return self.training_history

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.

        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "config": self.config,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint from

        """
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.training_history = checkpoint.get("training_history", [])

    def get_model_summary(self) -> dict[str, float]:
        """Get model summary statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": float(total_params),
            "trainable_parameters": float(trainable_params),
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
