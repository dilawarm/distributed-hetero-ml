import socket

import torch

from distributed_hetero_ml.types import DeviceInfo, DeviceType


class HardwareDetector:
    """Detects available hardware and provides device information."""

    @staticmethod
    def get_current_device_info() -> DeviceInfo:
        """
        Get information about the current device.

        Returns:
            DeviceInfo: Information about the current device.

        """
        hostname = socket.gethostname()

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_id)
            memory_gb = torch.cuda.get_device_properties(device_id).total_memory // (1024**3)

            return DeviceInfo(
                device_type=DeviceType.CUDA, device_name=f"cuda:{device_id} ({device_name})", hostname=hostname, device_id=device_id, memory_gb=memory_gb
            )

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(
                device_type=DeviceType.MPS,
                device_name="mps (Apple Silicon)",
                hostname=hostname,
                device_id=None,
                memory_gb=None,  # MPS doesn't expose memory info easily
            )

        else:
            return DeviceInfo(device_type=DeviceType.CPU, device_name="cpu", hostname=hostname, device_id=None, memory_gb=None)

    @staticmethod
    def get_available_devices() -> list[DeviceInfo]:
        """
        Get all available devices on the current machine.

        Returns:
            list[DeviceInfo]: List of available devices

        """
        devices = []
        hostname = socket.gethostname()

        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(device_id)
                memory_gb = torch.cuda.get_device_properties(device_id).total_memory // (1024**3)

                devices.append(
                    DeviceInfo(
                        device_type=DeviceType.CUDA,
                        device_name=f"cuda:{device_id} ({device_name})",
                        hostname=hostname,
                        device_id=device_id,
                        memory_gb=memory_gb,
                    )
                )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(DeviceInfo(device_type=DeviceType.MPS, device_name="mps (Apple Silicon)", hostname=hostname, device_id=None, memory_gb=None))

        devices.append(DeviceInfo(device_type=DeviceType.CPU, device_name="cpu", hostname=hostname, device_id=None, memory_gb=None))

        return devices

    @staticmethod
    def get_optimal_device() -> torch.device:
        """
        Get the optimal device for training.

        Returns:
            torch.device: The optimal device

        """
        device_info = HardwareDetector.get_current_device_info()

        if device_info.device_type == DeviceType.CUDA:
            return torch.device(f"cuda:{device_info.device_id}")
        elif device_info.device_type == DeviceType.MPS:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @staticmethod
    def estimate_batch_size(device_info: DeviceInfo, model_size_mb: float) -> int:
        """
        Estimate optimal batch size for given device and model.

        Args:
            device_info: Device information
            model_size_mb: Model size in MB

        Returns:
            int: Estimated optimal batch size

        """
        if device_info.device_type == DeviceType.CUDA and device_info.memory_gb:
            # Use 30% of GPU memory for batch data (more conservative)
            available_memory_mb = device_info.memory_gb * 1024 * 0.3
            # Rough estimate: each sample uses model_size/5 memory (more conservative)
            estimated_batch_size = int(available_memory_mb / (model_size_mb / 5))
            return max(1, min(128, estimated_batch_size))
        elif device_info.device_type == DeviceType.MPS:
            # Conservative estimate for MPS
            return 32
        else:
            # CPU - very conservative
            return 16
