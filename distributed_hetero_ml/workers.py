import logging
import time
from typing import Any

import ray
import torch

from distributed_hetero_ml.hardware import HardwareDetector
from distributed_hetero_ml.types import DataLoader, DeviceInfo, ModelFactory, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


class BaseWorker:
    """Base class for training workers."""

    def __init__(self, worker_id: int, model_factory: ModelFactory, data_loader: DataLoader, config: TrainingConfig):
        """
        Initialize the worker.

        Args:
            worker_id: Unique worker identifier
            model_factory: Factory for creating model components
            data_loader: Data loader for training data
            config: Training configuration

        """
        self.worker_id = worker_id
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.config = config

        self.model = model_factory.create_model()
        self.criterion = model_factory.create_criterion()

        self.device_info = HardwareDetector.get_current_device_info()
        self.device = self._get_torch_device()

        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info(f"Worker {worker_id} initialized on {self.device_info.hostname}: {self.device_info.device_name}")

    def _get_torch_device(self) -> torch.device:
        """Get PyTorch device from device info."""
        if self.device_info.device_type.value == "cuda":
            return torch.device(f"cuda:{self.device_info.device_id}")
        elif self.device_info.device_type.value == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def train_step(self, parameter_server: Any) -> TrainingResult:
        """
        Execute a single training step.

        Args:
            parameter_server: Ray parameter server actor

        Returns:
            TrainingResult: Results from the training step

        """
        start_time = time.time()

        weights: dict[str, torch.Tensor] = ray.get(parameter_server.get_weights.remote())
        self._update_model_weights(weights)

        X, y = self.data_loader.get_batch(self.config.batch_size, self.device)

        self.model.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)

        loss.backward()

        gradients = self._extract_gradients()

        metrics = self._calculate_metrics(outputs, y, loss)

        step_time = time.time() - start_time

        return TrainingResult(
            worker_id=self.worker_id, device_info=self.device_info, loss=loss.item(), gradients=gradients, metrics=metrics, step_time=step_time
        )

    def _update_model_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Update model weights from parameter server."""
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = weights[name].to(self.device)

    def _extract_gradients(self) -> dict[str, torch.Tensor]:
        """Extract gradients from model."""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()
        return gradients

    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
        """Calculate training metrics."""
        with torch.no_grad():
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == targets).float().mean().item()
            else:
                accuracy = 0.0

            return {"accuracy": accuracy, "loss": loss.item(), "batch_size": outputs.shape[0]}

    def get_device_info(self) -> DeviceInfo:
        """Get device information."""
        return self.device_info


@ray.remote(num_gpus=1)
class GPUWorker(BaseWorker):
    """Worker specialized for GPU training."""

    def __init__(self, worker_id: int, model_factory: ModelFactory, data_loader: DataLoader, config: TrainingConfig):
        super().__init__(worker_id, model_factory, data_loader, config)

        self.scaler: torch.cuda.amp.GradScaler | None
        if config.use_mixed_precision and self.device_info.device_type.value == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False

    def train_step(self, parameter_server: Any) -> TrainingResult:
        """Execute training step with GPU optimizations."""
        if self.use_amp:
            return self._train_step_with_amp(parameter_server)
        else:
            return super().train_step(parameter_server)

    def _train_step_with_amp(self, parameter_server: Any) -> TrainingResult:
        """Training step with automatic mixed precision."""
        start_time = time.time()

        weights: dict[str, torch.Tensor] = ray.get(parameter_server.get_weights.remote())
        self._update_model_weights(weights)

        X, y = self.data_loader.get_batch(self.config.batch_size, self.device)

        self.model.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(torch.optim.SGD(self.model.parameters(), lr=0.01))
            self.scaler.update()
        else:
            loss.backward()

        gradients = self._extract_gradients()

        metrics = self._calculate_metrics(outputs, y, loss)

        step_time = time.time() - start_time

        return TrainingResult(
            worker_id=self.worker_id, device_info=self.device_info, loss=loss.item(), gradients=gradients, metrics=metrics, step_time=step_time
        )


@ray.remote(num_cpus=2)
class CPUWorker(BaseWorker):
    """Worker specialized for CPU training."""

    def __init__(self, worker_id: int, model_factory: ModelFactory, data_loader: DataLoader, config: TrainingConfig):
        super().__init__(worker_id, model_factory, data_loader, config)

        torch.set_num_threads(2)


@ray.remote(num_cpus=2)
class MPSWorker(BaseWorker):
    """Worker specialized for Apple Silicon MPS training."""

    def __init__(self, worker_id: int, model_factory: ModelFactory, data_loader: DataLoader, config: TrainingConfig):
        super().__init__(worker_id, model_factory, data_loader, config)

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


class WorkerFactory:
    """Factory for creating appropriate workers based on available hardware."""

    @staticmethod
    def create_workers(
        model_factory: ModelFactory, data_loader: DataLoader, config: TrainingConfig, num_gpu_workers: int | None = None, num_cpu_workers: int | None = None
    ) -> list[Any]:
        """
        Create workers based on available hardware and configuration.

        Args:
            model_factory: Factory for creating model components
            data_loader: Data loader for training data
            config: Training configuration
            num_gpu_workers: Number of GPU workers (overrides config)
            num_cpu_workers: Number of CPU workers (overrides config)

        Returns:
            List of Ray worker actors

        """
        workers = []
        worker_id = 0

        num_gpus = num_gpu_workers if num_gpu_workers is not None else config.num_gpu_workers
        num_cpus = num_cpu_workers if num_cpu_workers is not None else config.num_cpu_workers

        available_devices = HardwareDetector.get_available_devices()
        gpu_devices = [d for d in available_devices if d.device_type.value == "cuda"]
        mps_devices = [d for d in available_devices if d.device_type.value == "mps"]

        actual_gpu_workers = min(num_gpus, len(gpu_devices))
        for i in range(actual_gpu_workers):
            worker = GPUWorker.remote(worker_id, model_factory, data_loader, config)  # type: ignore
            workers.append(worker)
            worker_id += 1

        if actual_gpu_workers == 0 and mps_devices:
            worker = MPSWorker.remote(worker_id, model_factory, data_loader, config)  # type: ignore
            workers.append(worker)
            worker_id += 1

        for i in range(num_cpus):
            worker = CPUWorker.remote(worker_id, model_factory, data_loader, config)  # type: ignore
            workers.append(worker)
            worker_id += 1

        if not workers:
            worker = CPUWorker.remote(worker_id, model_factory, data_loader, config)  # type: ignore
            workers.append(worker)

        return workers

    @staticmethod
    def get_optimal_worker_count(device_info: DeviceInfo) -> dict[str, int]:
        """
        Get optimal worker count based on device capabilities.

        Args:
            device_info: Device information

        Returns:
            Dict with recommended worker counts

        """
        if device_info.device_type.value == "cuda" and device_info.memory_gb:
            gpu_workers = 2 if device_info.memory_gb > 8 else 1
            cpu_workers = 1
        elif device_info.device_type.value == "mps":
            gpu_workers = 0
            cpu_workers = 2
        else:
            gpu_workers = 0
            cpu_workers = 4

        return {"num_gpu_workers": gpu_workers, "num_cpu_workers": cpu_workers}
