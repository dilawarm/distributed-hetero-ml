import os
import tempfile
import time
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn, optim

from distributed_hetero_ml.types import TrainingConfig


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[no-any-return]


class MockModelFactory:
    """Mock model factory for testing."""

    def create_model(self) -> nn.Module:
        return MockModel()

    def create_optimizer(self, model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        return optim.SGD(model.parameters(), lr=config.learning_rate)

    def create_criterion(self) -> nn.Module:
        return nn.MSELoss()


def create_parameter_server_for_testing(model_factory: MockModelFactory, config: TrainingConfig) -> Any:
    """Create a parameter server instance for testing without Ray remote decorator."""

    class TestParameterServer:
        def __init__(self, model_factory: MockModelFactory, config: TrainingConfig):
            self.model = model_factory.create_model()
            self.optimizer = model_factory.create_optimizer(self.model, config)
            self.config = config
            self.iteration = 0
            self.training_history: list[dict[str, Any]] = []
            self.start_time = time.time()
            self.model = self.model.cpu()
            self.gradient_clipping = config.gradient_clipping

        def get_weights(self) -> dict[str, torch.Tensor]:
            return {name: param.clone().detach() for name, param in self.model.named_parameters()}

        def update_weights(self, gradients_list: list[dict[str, torch.Tensor]]) -> dict[str, float]:
            if not gradients_list:
                return {"iteration": self.iteration, "num_workers": 0}

            avg_gradients = self._average_gradients(gradients_list)
            self._apply_gradients(avg_gradients)
            self.iteration += 1
            metrics = self._calculate_metrics(gradients_list)

            self.training_history.append(
                {"iteration": self.iteration, "timestamp": time.time() - self.start_time, "num_workers": len(gradients_list), **metrics}
            )

            return metrics

        def _average_gradients(self, gradients_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            if not gradients_list:
                return {}

            avg_gradients = {}
            param_names = gradients_list[0].keys()

            for name in param_names:
                gradients = torch.stack([grads[name] for grads in gradients_list])
                avg_gradients[name] = gradients.mean(dim=0)

            return avg_gradients

        def _apply_gradients(self, gradients: dict[str, torch.Tensor]) -> None:
            for name, param in self.model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name]

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()
            self.optimizer.zero_grad()

        def _calculate_metrics(self, gradients_list: list[dict[str, torch.Tensor]]) -> dict[str, float]:
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
            return self.iteration

        def get_training_history(self) -> list[dict[str, Any]]:
            return self.training_history

        def save_checkpoint(self, path: str) -> None:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "config": self.config,
                "training_history": self.training_history,
            }
            torch.save(checkpoint, path)

        def load_checkpoint(self, path: str) -> None:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.iteration = checkpoint["iteration"]
            self.training_history = checkpoint.get("training_history", [])

        def get_model_summary(self) -> dict[str, float]:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            return {
                "total_parameters": float(total_params),
                "trainable_parameters": float(trainable_params),
                "model_size_mb": total_params * 4 / (1024 * 1024),
            }

    return TestParameterServer(model_factory, config)


class TestRayParameterServer:
    """Test cases for RayParameterServer."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TrainingConfig(learning_rate=0.01, gradient_clipping=1.0, batch_size=32)

    @pytest.fixture
    def model_factory(self):
        """Create test model factory."""
        return MockModelFactory()

    @pytest.fixture
    def mock_ray_remote(self):
        """Mock ray.remote decorator."""

        def decorator(cls):
            return cls

        return decorator

    def test_parameter_server_initialization(self, config, model_factory):
        """Test parameter server initialization."""
        ps = create_parameter_server_for_testing(model_factory, config)

        assert ps.model is not None
        assert ps.optimizer is not None
        assert ps.config == config
        assert ps.iteration == 0
        assert ps.gradient_clipping == config.gradient_clipping
        assert isinstance(ps.training_history, list)

    def test_get_weights(self, config, model_factory):
        """Test getting model weights."""
        ps = create_parameter_server_for_testing(model_factory, config)

        weights = ps.get_weights()

        assert isinstance(weights, dict)
        assert "linear.weight" in weights
        assert "linear.bias" in weights
        assert isinstance(weights["linear.weight"], torch.Tensor)
        assert isinstance(weights["linear.bias"], torch.Tensor)

    def test_update_weights_single_worker(self, config, model_factory):
        """Test updating weights with single worker gradients."""
        ps = create_parameter_server_for_testing(model_factory, config)

        gradients = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}

        metrics = ps.update_weights([gradients])

        assert ps.iteration == 1
        assert isinstance(metrics, dict)
        assert "iteration" in metrics
        assert "num_workers" in metrics
        assert "avg_grad_norm" in metrics
        assert metrics["num_workers"] == 1
        assert metrics["iteration"] == 1

    def test_update_weights_multiple_workers(self, config, model_factory):
        """Test updating weights with multiple worker gradients."""
        ps = create_parameter_server_for_testing(model_factory, config)

        gradients_list = [
            {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)},
            {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)},
            {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)},
        ]

        metrics = ps.update_weights(gradients_list)

        assert ps.iteration == 1
        assert metrics["num_workers"] == 3
        assert metrics["iteration"] == 1

    def test_update_weights_empty_gradients(self, config, model_factory):
        """Test updating weights with empty gradients list."""
        ps = create_parameter_server_for_testing(model_factory, config)

        metrics = ps.update_weights([])

        assert ps.iteration == 0
        assert metrics["num_workers"] == 0
        assert metrics["iteration"] == 0

    def test_average_gradients(self, config, model_factory):
        """Test gradient averaging functionality."""
        ps = create_parameter_server_for_testing(model_factory, config)

        grad1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        grad2 = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]])

        gradients_list = [{"linear.weight": grad1, "linear.bias": torch.tensor([1.0])}, {"linear.weight": grad2, "linear.bias": torch.tensor([2.0])}]

        avg_gradients = ps._average_gradients(gradients_list)

        expected_weight = torch.tensor([[1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]])
        expected_bias = torch.tensor([1.5])

        assert torch.allclose(avg_gradients["linear.weight"], expected_weight)
        assert torch.allclose(avg_gradients["linear.bias"], expected_bias)

    def test_gradient_clipping(self, model_factory):
        """Test gradient clipping functionality."""
        config = TrainingConfig(learning_rate=0.01, gradient_clipping=0.5)
        ps = create_parameter_server_for_testing(model_factory, config)

        large_gradients = {"linear.weight": torch.ones(1, 10) * 10.0, "linear.bias": torch.ones(1) * 10.0}

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            ps.update_weights([large_gradients])

            mock_clip.assert_called_once()
            args, kwargs = mock_clip.call_args
            assert args[1] == 0.5

    def test_training_history_tracking(self, config, model_factory):
        """Test training history tracking."""
        ps = create_parameter_server_for_testing(model_factory, config)

        gradients = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}

        ps.update_weights([gradients])
        ps.update_weights([gradients])
        ps.update_weights([gradients])

        history = ps.get_training_history()

        assert len(history) == 3
        assert all("iteration" in entry for entry in history)
        assert all("num_workers" in entry for entry in history)
        assert all("timestamp" in entry for entry in history)
        assert history[0]["iteration"] == 1
        assert history[1]["iteration"] == 2
        assert history[2]["iteration"] == 3

    def test_save_and_load_checkpoint(self, config, model_factory):
        """Test checkpoint saving and loading."""
        ps = create_parameter_server_for_testing(model_factory, config)

        gradients = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}
        ps.update_weights([gradients])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            checkpoint_path = f.name

        try:
            ps.save_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)

            ps2 = create_parameter_server_for_testing(model_factory, config)

            assert ps2.iteration == 0

            ps2.load_checkpoint(checkpoint_path)

            assert ps2.iteration == ps.iteration
            assert len(ps2.training_history) == len(ps.training_history)

        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)

    def test_model_summary(self, config, model_factory):
        """Test model summary generation."""
        ps = create_parameter_server_for_testing(model_factory, config)

        summary = ps.get_model_summary()

        assert isinstance(summary, dict)
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "model_size_mb" in summary

        assert summary["total_parameters"] == 11.0
        assert summary["trainable_parameters"] == 11.0
        assert summary["model_size_mb"] > 0

    def test_calculate_metrics(self, config, model_factory):
        """Test metrics calculation."""
        ps = create_parameter_server_for_testing(model_factory, config)

        gradients_list = [
            {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)},
            {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)},
        ]

        for name, param in ps.model.named_parameters():
            param.grad = torch.randn_like(param)

        metrics = ps._calculate_metrics(gradients_list)

        assert "iteration" in metrics
        assert "num_workers" in metrics
        assert "avg_grad_norm" in metrics
        assert "learning_rate" in metrics
        assert "elapsed_time" in metrics

        assert metrics["num_workers"] == 2
        assert metrics["learning_rate"] == config.learning_rate
        assert isinstance(metrics["avg_grad_norm"], float)
        assert isinstance(metrics["elapsed_time"], float)


class TestParameterServerIntegration:
    """Integration tests for parameter server."""

    def test_multiple_training_steps(self):
        """Test multiple training steps in sequence."""
        config = TrainingConfig(learning_rate=0.01, batch_size=32)
        model_factory = MockModelFactory()

        ps = create_parameter_server_for_testing(model_factory, config)

        for i in range(5):
            gradients = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}

            metrics = ps.update_weights([gradients])

            assert metrics["iteration"] == i + 1
            assert metrics["num_workers"] == 1

        assert ps.iteration == 5
        assert len(ps.training_history) == 5

    def test_parameter_server_with_different_configs(self):
        """Test parameter server with different configurations."""
        configs = [
            TrainingConfig(learning_rate=0.01, gradient_clipping=None),
            TrainingConfig(learning_rate=0.1, gradient_clipping=1.0),
            TrainingConfig(learning_rate=0.001, gradient_clipping=0.5),
        ]

        for config in configs:
            model_factory = MockModelFactory()

            ps = create_parameter_server_for_testing(model_factory, config)

            gradients = {"linear.weight": torch.randn(1, 10), "linear.bias": torch.randn(1)}

            metrics = ps.update_weights([gradients])

            assert metrics["learning_rate"] == config.learning_rate
            assert ps.gradient_clipping == config.gradient_clipping
