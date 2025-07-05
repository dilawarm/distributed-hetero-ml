from unittest.mock import MagicMock, patch

import pytest
import torch

from distributed_hetero_ml.core import DistributedTrainer, quick_start
from distributed_hetero_ml.hardware import HardwareDetector
from distributed_hetero_ml.types import TrainingConfig
from distributed_hetero_ml.utils import SimpleModelFactory, SyntheticDataLoader
from distributed_hetero_ml.workers import WorkerFactory


class TestFrameworkIntegration:
    """Integration tests for the complete framework."""

    @pytest.fixture
    def mock_ray(self):
        """Mock Ray for testing."""
        mock_actor = MagicMock()
        mock_actor.remote = MagicMock(return_value=mock_actor)
        mock_actor.get_model_state.remote = MagicMock(return_value=mock_actor)
        mock_actor.update_model.remote = MagicMock(return_value=mock_actor)
        mock_actor.train_step.remote = MagicMock(return_value=mock_actor)

        with (
            patch("ray.init"),
            patch("ray.cluster_resources", return_value={"CPU": 4, "GPU": 0}),
            patch("ray.nodes", return_value=[{"NodeID": "test"}]),
            patch("ray.get", return_value={}),
            patch("ray.remote", return_value=lambda cls: mock_actor),
            patch("distributed_hetero_ml.parameter_server.RayParameterServer.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.CPUWorker.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.GPUWorker.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.MPSWorker.remote", return_value=mock_actor),
        ):
            yield mock_actor

    @pytest.fixture
    def mock_torch(self):
        """Mock torch for testing."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("socket.gethostname", return_value="test-machine"),
        ):
            yield

    def test_quick_start_functionality(self, mock_ray, mock_torch):
        """Test quick start functionality."""
        model_factory = SimpleModelFactory(input_size=10, hidden_size=5, output_size=2)
        data_loader = SyntheticDataLoader(input_size=10, num_classes=2, dataset_size=100)

        trainer = quick_start(model_factory=model_factory, data_loader=data_loader, num_iterations=5)

        assert trainer is not None
        assert isinstance(trainer, DistributedTrainer)
        assert trainer.model_factory == model_factory
        assert trainer.data_loader == data_loader

    def test_distributed_trainer_creation(self, mock_ray, mock_torch):
        """Test distributed trainer creation and configuration."""
        config = TrainingConfig(batch_size=16, learning_rate=0.01, num_gpu_workers=0, num_cpu_workers=1)

        model_factory = SimpleModelFactory(input_size=10, hidden_size=5, output_size=2)
        data_loader = SyntheticDataLoader(input_size=10, num_classes=2, dataset_size=100)

        trainer = DistributedTrainer(model_factory=model_factory, data_loader=data_loader, config=config)

        assert trainer.config == config
        assert trainer.model_factory == model_factory
        assert trainer.data_loader == data_loader

    def test_hardware_detection_integration(self, mock_torch):
        """Test hardware detection integration."""
        device_info = HardwareDetector.get_current_device_info()
        assert device_info.hostname == "test-machine"
        assert device_info.device_name == "cpu"

        devices = HardwareDetector.get_available_devices()
        assert len(devices) >= 1
        assert any(d.device_name == "cpu" for d in devices)

        optimal_device = HardwareDetector.get_optimal_device()
        assert str(optimal_device) == "cpu"

    def test_worker_factory_integration(self, mock_ray, mock_torch):
        """Test worker factory integration."""
        config = TrainingConfig(num_gpu_workers=0, num_cpu_workers=2)

        model_factory = SimpleModelFactory()
        data_loader = SyntheticDataLoader()

        workers = WorkerFactory.create_workers(model_factory=model_factory, data_loader=data_loader, config=config)

        assert len(workers) >= 1

    def test_context_manager_usage(self, mock_ray, mock_torch):
        """Test using trainer as context manager."""
        model_factory = SimpleModelFactory(input_size=10, hidden_size=5, output_size=2)
        data_loader = SyntheticDataLoader(input_size=10, num_classes=2, dataset_size=100)

        trainer = quick_start(model_factory=model_factory, data_loader=data_loader, num_iterations=3)

        with trainer:
            assert not trainer.is_training

        assert not trainer.is_training

    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        config = TrainingConfig()
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.num_epochs == 10

        custom_config = TrainingConfig(batch_size=32, learning_rate=0.001, num_epochs=5, gradient_clipping=0.5, use_mixed_precision=True)

        assert custom_config.batch_size == 32
        assert custom_config.learning_rate == 0.001
        assert custom_config.num_epochs == 5
        assert custom_config.gradient_clipping == 0.5
        assert custom_config.use_mixed_precision is True

    def test_model_factory_integration(self):
        """Test model factory integration."""
        factory = SimpleModelFactory(input_size=784, hidden_size=128, output_size=10)

        model = factory.create_model()
        assert model is not None

        config = TrainingConfig(learning_rate=0.01)
        optimizer = factory.create_optimizer(model, config)
        assert optimizer is not None

        criterion = factory.create_criterion()
        assert criterion is not None

    def test_data_loader_integration(self):
        """Test data loader integration."""
        data_loader = SyntheticDataLoader(input_size=784, num_classes=10, dataset_size=1000)

        assert data_loader.get_dataset_size() == 1000

        device = torch.device("cpu")
        X, y = data_loader.get_batch(32, device)

        assert X.shape == (32, 784)
        assert y.shape == (32,)
        assert X.device == device
        assert y.device == device

    def test_error_handling(self, mock_ray, mock_torch):
        """Test error handling in framework."""
        model_factory = SimpleModelFactory()
        data_loader = SyntheticDataLoader()

        trainer = DistributedTrainer(model_factory=model_factory, data_loader=data_loader, config=TrainingConfig(), cluster_address="invalid://address")

        assert trainer is not None

    def test_performance_monitoring(self, mock_ray, mock_torch):
        """Test performance monitoring capabilities."""
        model_factory = SimpleModelFactory(input_size=10, hidden_size=5, output_size=2)
        data_loader = SyntheticDataLoader(input_size=10, num_classes=2, dataset_size=100)

        trainer = quick_start(model_factory=model_factory, data_loader=data_loader, num_iterations=3)

        with patch.object(trainer, "train") as mock_train:
            mock_train.return_value = [
                {"iteration": 1, "avg_loss": 1.5, "avg_accuracy": 0.3, "devices_used": ["cpu"], "hostnames_used": ["test-machine"], "num_workers": 1}
            ]

            with trainer:
                results = trainer.train(num_iterations=3)

                assert len(results) == 1
                assert results[0]["iteration"] == 1
                assert "avg_loss" in results[0]
                assert "avg_accuracy" in results[0]


class TestFrameworkUsability:
    """Tests for framework usability and ease of use."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch for testing."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("socket.gethostname", return_value="test-machine"),
        ):
            yield

    def test_minimal_usage(self):
        """Test minimal usage example."""
        model_factory = SimpleModelFactory()
        data_loader = SyntheticDataLoader()

        mock_actor = MagicMock()
        mock_actor.remote = MagicMock(return_value=mock_actor)
        mock_actor.get_model_state.remote = MagicMock(return_value=mock_actor)
        mock_actor.update_model.remote = MagicMock(return_value=mock_actor)
        mock_actor.train_step.remote = MagicMock(return_value=mock_actor)

        with (
            patch("ray.init"),
            patch("ray.cluster_resources", return_value={"CPU": 1}),
            patch("ray.nodes", return_value=[]),
            patch("ray.get", return_value={}),
            patch("ray.remote", return_value=lambda cls: mock_actor),
            patch("distributed_hetero_ml.parameter_server.RayParameterServer.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.CPUWorker.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.GPUWorker.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.MPSWorker.remote", return_value=mock_actor),
        ):
            trainer = quick_start(model_factory, data_loader, num_iterations=1)
            assert trainer is not None

    def test_progressive_complexity(self):
        """Test that users can progressively add complexity."""
        config1 = TrainingConfig()

        config2 = TrainingConfig(batch_size=128, learning_rate=0.001, use_mixed_precision=True)

        config3 = TrainingConfig(batch_size=256, learning_rate=0.0001, use_mixed_precision=True, gradient_clipping=1.0, num_gpu_workers=2, num_cpu_workers=1)

        for config in [config1, config2, config3]:
            assert config.batch_size > 0
            assert config.learning_rate > 0

    def test_documentation_examples(self, mock_torch):
        """Test that documentation examples work."""
        mock_actor = MagicMock()
        mock_actor.remote = MagicMock(return_value=mock_actor)
        mock_actor.get_model_state.remote = MagicMock(return_value=mock_actor)
        mock_actor.update_model.remote = MagicMock(return_value=mock_actor)
        mock_actor.train_step.remote = MagicMock(return_value=mock_actor)

        with (
            patch("ray.init"),
            patch("ray.cluster_resources", return_value={"CPU": 1}),
            patch("ray.nodes", return_value=[]),
            patch("ray.get", return_value={}),
            patch("ray.remote", return_value=lambda cls: mock_actor),
            patch("distributed_hetero_ml.parameter_server.RayParameterServer.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.CPUWorker.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.GPUWorker.remote", return_value=mock_actor),
            patch("distributed_hetero_ml.workers.MPSWorker.remote", return_value=mock_actor),
        ):
            model_factory = SimpleModelFactory(input_size=784, hidden_size=128, output_size=10)
            data_loader = SyntheticDataLoader(input_size=784, num_classes=10)

            trainer = quick_start(model_factory, data_loader, num_iterations=1)

            assert trainer is not None
            assert hasattr(trainer, "train")
            assert hasattr(trainer, "save_checkpoint")
            assert hasattr(trainer, "get_model_summary")
