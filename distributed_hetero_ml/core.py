import logging
import time
from collections.abc import Callable
from typing import Any

import ray

from distributed_hetero_ml.hardware import HardwareDetector
from distributed_hetero_ml.parameter_server import RayParameterServer
from distributed_hetero_ml.types import DataLoader, ModelFactory, TrainingConfig, TrainingResult
from distributed_hetero_ml.workers import WorkerFactory

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Main distributed training orchestrator.

    Coordinates parameter server and workers for distributed training
    across heterogeneous hardware.
    """

    def __init__(self, model_factory: ModelFactory, data_loader: DataLoader, config: TrainingConfig, cluster_address: str | None = None):
        """
        Initialize the distributed trainer.

        Args:
            model_factory: Factory for creating model components
            data_loader: Data loader for training data
            config: Training configuration
            cluster_address: Ray cluster address (None for local)

        """
        self.model_factory = model_factory
        self.data_loader = data_loader
        self.config = config
        self.cluster_address = cluster_address

        self._initialize_ray()

        self.parameter_server = RayParameterServer.remote(model_factory, config)  # type: ignore

        self.workers = WorkerFactory.create_workers(model_factory, data_loader, config)

        self.training_results: list[dict[str, Any]] = []
        self.is_training = False

        logger.info(f"🚀 Distributed trainer initialized with {len(self.workers)} workers")
        self._print_cluster_info()

    def _initialize_ray(self) -> None:
        """Initialize Ray cluster connection."""
        try:
            if self.cluster_address:
                ray.init(address=self.cluster_address, ignore_reinit_error=True)
                logger.info(f"✅ Connected to Ray cluster at {self.cluster_address}")
            else:
                ray.init(ignore_reinit_error=True)
                logger.info("🔧 Started local Ray cluster")
        except Exception as e:
            logger.warning(f"⚠️ Ray initialization failed: {e}")
            logger.info("🔧 Starting local Ray cluster as fallback")
            ray.init(ignore_reinit_error=True)

    def _print_cluster_info(self) -> None:
        """Print cluster information."""
        resources = ray.cluster_resources()
        nodes = len(ray.nodes())

        logger.info("📊 Cluster info:")
        logger.info(f"  - Nodes: {nodes}")
        logger.info(f"  - CPUs: {resources.get('CPU', 0)}")
        logger.info(f"  - GPUs: {resources.get('GPU', 0)}")
        logger.info(f"  - Memory: {resources.get('memory', 0) / 1e9:.1f}GB")

    def train(
        self, num_iterations: int, progress_callback: Callable[[int, dict[str, float]], None] | None = None, checkpoint_path: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Run distributed training.

        Args:
            num_iterations: Number of training iterations
            progress_callback: Optional callback for progress updates
            checkpoint_path: Optional path to save checkpoints

        Returns:
            List of training results

        """
        logger.info(f"🎯 Starting distributed training for {num_iterations} iterations")

        self.is_training = True
        training_history = []

        try:
            for iteration in range(num_iterations):
                logger.info(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

                step_start = time.time()
                futures = [worker.train_step.remote(self.parameter_server) for worker in self.workers]

                results = ray.get(futures)

                gradients_list = [result.gradients for result in results]

                metrics: dict[str, float] = ray.get(self.parameter_server.update_weights.remote(gradients_list))

                iteration_metrics = self._calculate_iteration_metrics(results, metrics, step_start)
                training_history.append(iteration_metrics)

                self._print_progress(iteration + 1, results, iteration_metrics)

                if progress_callback:
                    progress_callback(iteration + 1, iteration_metrics)

                if checkpoint_path and (iteration + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_file = f"{checkpoint_path}_iter_{iteration + 1}.pt"
                    ray.get(self.parameter_server.save_checkpoint.remote(checkpoint_file))
                    logger.info(f"💾 Checkpoint saved: {checkpoint_file}")

        except KeyboardInterrupt:
            logger.info("\n⏹️ Training interrupted by user")

        finally:
            self.is_training = False

        logger.info("\n🎉 Training complete!")
        return training_history

    def _calculate_iteration_metrics(self, results: list[TrainingResult], ps_metrics: dict[str, float], step_start: float) -> dict[str, Any]:
        """Calculate metrics for current iteration."""
        total_step_time = time.time() - step_start

        total_loss = sum(r.loss for r in results)
        avg_loss = total_loss / len(results)

        total_accuracy = sum(r.metrics.get("accuracy", 0) for r in results)
        avg_accuracy = total_accuracy / len(results)

        devices_used = list(set([r.device_info.device_name for r in results]))
        hostnames_used = list(set([r.device_info.hostname for r in results]))

        worker_times = [r.step_time for r in results]

        return {
            **ps_metrics,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "total_step_time": total_step_time,
            "avg_worker_time": sum(worker_times) / len(worker_times),
            "max_worker_time": max(worker_times),
            "devices_used": devices_used,
            "hostnames_used": hostnames_used,
            "num_workers": len(results),
        }

    def _print_progress(self, iteration: int, results: list[TrainingResult], metrics: dict[str, Any]) -> None:
        """Print training progress."""
        logger.info("  📈 Results:")
        logger.info(f"    - Avg Loss: {metrics['avg_loss']:.4f}")
        logger.info(f"    - Avg Accuracy: {metrics['avg_accuracy']:.4f}")
        logger.info(f"    - Step Time: {metrics['total_step_time']:.2f}s")
        logger.info(f"    - Learning Rate: {metrics['learning_rate']:.6f}")

        logger.info("  🖥️  Workers:")
        for result in results:
            logger.info(f"    - Worker {result.worker_id}: {result.device_info.device_name} | Loss: {result.loss:.4f} | Time: {result.step_time:.2f}s")

        logger.info(f"  🌐 Distributed on: {', '.join(metrics['hostnames_used'])}")
        logger.info(f"  ⚡ Devices: {', '.join(metrics['devices_used'])}")

    def evaluate(self, num_batches: int = 10) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            num_batches: Number of batches to evaluate

        Returns:
            Evaluation metrics

        """
        logger.info(f"📊 Evaluating model on {num_batches} batches...")

        eval_results = []

        for _ in range(num_batches):
            if self.workers:
                future = self.workers[0].train_step.remote(self.parameter_server)
                result = ray.get(future)
                eval_results.append(result)

        avg_loss = sum(r.loss for r in eval_results) / len(eval_results)
        avg_accuracy = sum(r.metrics.get("accuracy", 0) for r in eval_results) / len(eval_results)

        eval_metrics = {"eval_loss": avg_loss, "eval_accuracy": avg_accuracy, "num_batches": num_batches}

        logger.info("📋 Evaluation complete:")
        logger.info(f"  - Loss: {avg_loss:.4f}")
        logger.info(f"  - Accuracy: {avg_accuracy:.4f}")

        return eval_metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint

        """
        ray.get(self.parameter_server.save_checkpoint.remote(path))
        logger.info(f"💾 Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to load checkpoint from

        """
        ray.get(self.parameter_server.load_checkpoint.remote(path))
        logger.info(f"📂 Checkpoint loaded from {path}")

    def get_model_summary(self) -> dict[str, float]:
        """Get model summary."""
        return ray.get(self.parameter_server.get_model_summary.remote())  # type: ignore

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get training history from parameter server."""
        return ray.get(self.parameter_server.get_training_history.remote())  # type: ignore

    def shutdown(self) -> None:
        """Shutdown the distributed trainer."""
        logger.info("🛑 Shutting down distributed trainer...")

        self.is_training = False

        logger.info("✅ Shutdown complete")

    def __enter__(self) -> "DistributedTrainer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()


def quick_start(
    model_factory: ModelFactory, data_loader: DataLoader, num_iterations: int = 10, cluster_address: str | None = None, config: TrainingConfig | None = None
) -> DistributedTrainer:
    """
    Quick start function for simple distributed training.

    Args:
        model_factory: Factory for creating model components
        data_loader: Data loader for training data
        num_iterations: Number of training iterations
        cluster_address: Ray cluster address (None for local)
        config: Training configuration (uses defaults if None)

    Returns:
        Configured DistributedTrainer

    """
    if config is None:
        config = TrainingConfig()

    device_info = HardwareDetector.get_current_device_info()
    optimal_workers = WorkerFactory.get_optimal_worker_count(device_info)

    config.num_gpu_workers = optimal_workers["num_gpu_workers"]
    config.num_cpu_workers = optimal_workers["num_cpu_workers"]

    trainer = DistributedTrainer(model_factory, data_loader, config, cluster_address)

    logger.info(f"🚀 Quick start: Auto-configured for {device_info.device_name}")
    logger.info(f"   - GPU workers: {config.num_gpu_workers}")
    logger.info(f"   - CPU workers: {config.num_cpu_workers}")

    return trainer
