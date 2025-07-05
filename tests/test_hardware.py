from unittest.mock import MagicMock, patch

import pytest

from distributed_hetero_ml.hardware import HardwareDetector
from distributed_hetero_ml.types import DeviceInfo, DeviceType


class TestHardwareDetector:
    """Test cases for HardwareDetector class."""

    def test_get_current_device_info_cuda(self):
        """Test device info detection for CUDA devices."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_name", return_value="NVIDIA GTX 1080"),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("socket.gethostname", return_value="test-machine"),
        ):
            mock_props.return_value = MagicMock(total_memory=11811160064)  # ~11GB

            device_info = HardwareDetector.get_current_device_info()

            assert device_info.device_type == DeviceType.CUDA
            assert device_info.device_name == "cuda:0 (NVIDIA GTX 1080)"
            assert device_info.hostname == "test-machine"
            assert device_info.device_id == 0
            assert device_info.memory_gb == pytest.approx(11.0, rel=0.1)

    def test_get_current_device_info_mps(self):
        """Test device info detection for MPS devices."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("socket.gethostname", return_value="macbook"),
        ):
            device_info = HardwareDetector.get_current_device_info()

            assert device_info.device_type == DeviceType.MPS
            assert device_info.device_name == "mps (Apple Silicon)"
            assert device_info.hostname == "macbook"
            assert device_info.device_id is None
            assert device_info.memory_gb is None

    def test_get_current_device_info_cpu(self):
        """Test device info detection for CPU devices."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
            patch("socket.gethostname", return_value="cpu-machine"),
        ):
            device_info = HardwareDetector.get_current_device_info()

            assert device_info.device_type == DeviceType.CPU
            assert device_info.device_name == "cpu"
            assert device_info.hostname == "cpu-machine"
            assert device_info.device_id is None
            assert device_info.memory_gb is None

    def test_get_available_devices_multi_gpu(self):
        """Test getting all available devices on a multi-GPU system."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_name") as mock_name,
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.backends.mps.is_available", return_value=False),
            patch("socket.gethostname", return_value="gpu-server"),
        ):
            mock_name.side_effect = lambda i: f"NVIDIA GTX 108{i}"
            mock_props.side_effect = lambda i: MagicMock(total_memory=11811160064)

            devices = HardwareDetector.get_available_devices()

            assert len(devices) == 3

            gpu_devices = [d for d in devices if d.device_type == DeviceType.CUDA]
            assert len(gpu_devices) == 2
            assert gpu_devices[0].device_name == "cuda:0 (NVIDIA GTX 1080)"
            assert gpu_devices[1].device_name == "cuda:1 (NVIDIA GTX 1081)"

            cpu_devices = [d for d in devices if d.device_type == DeviceType.CPU]
            assert len(cpu_devices) == 1
            assert cpu_devices[0].device_name == "cpu"

    def test_get_optimal_device_cuda(self):
        """Test optimal device selection for CUDA."""
        with patch("distributed_hetero_ml.hardware.HardwareDetector.get_current_device_info") as mock_info:
            mock_info.return_value = DeviceInfo(device_type=DeviceType.CUDA, device_name="cuda:0 (NVIDIA GTX 1080)", hostname="test", device_id=0)

            device = HardwareDetector.get_optimal_device()
            assert str(device) == "cuda:0"

    def test_get_optimal_device_mps(self):
        """Test optimal device selection for MPS."""
        with patch("distributed_hetero_ml.hardware.HardwareDetector.get_current_device_info") as mock_info:
            mock_info.return_value = DeviceInfo(device_type=DeviceType.MPS, device_name="mps (Apple Silicon)", hostname="macbook")

            device = HardwareDetector.get_optimal_device()
            assert str(device) == "mps"

    def test_get_optimal_device_cpu(self):
        """Test optimal device selection for CPU."""
        with patch("distributed_hetero_ml.hardware.HardwareDetector.get_current_device_info") as mock_info:
            mock_info.return_value = DeviceInfo(device_type=DeviceType.CPU, device_name="cpu", hostname="cpu-machine")

            device = HardwareDetector.get_optimal_device()
            assert str(device) == "cpu"

    def test_estimate_batch_size_high_memory_gpu(self):
        """Test batch size estimation for high-memory GPU."""
        device_info = DeviceInfo(device_type=DeviceType.CUDA, device_name="cuda:0 (NVIDIA RTX 3090)", hostname="test", device_id=0, memory_gb=24)

        batch_size = HardwareDetector.estimate_batch_size(device_info, 100.0)
        assert batch_size > 50
        assert batch_size <= 128

    def test_estimate_batch_size_low_memory_gpu(self):
        """Test batch size estimation for low-memory GPU."""
        device_info = DeviceInfo(device_type=DeviceType.CUDA, device_name="cuda:0 (NVIDIA GTX 1050)", hostname="test", device_id=0, memory_gb=2)

        batch_size = HardwareDetector.estimate_batch_size(device_info, 100.0)
        assert batch_size >= 1
        assert batch_size < 50

    def test_estimate_batch_size_mps(self):
        """Test batch size estimation for MPS."""
        device_info = DeviceInfo(device_type=DeviceType.MPS, device_name="mps (Apple Silicon)", hostname="macbook")

        batch_size = HardwareDetector.estimate_batch_size(device_info, 100.0)
        assert batch_size == 32

    def test_estimate_batch_size_cpu(self):
        """Test batch size estimation for CPU."""
        device_info = DeviceInfo(device_type=DeviceType.CPU, device_name="cpu", hostname="cpu-machine")

        batch_size = HardwareDetector.estimate_batch_size(device_info, 100.0)
        assert batch_size == 16

    def test_estimate_batch_size_no_memory_info(self):
        """Test batch size estimation when memory info is not available."""
        device_info = DeviceInfo(device_type=DeviceType.CUDA, device_name="cuda:0 (Unknown GPU)", hostname="test", device_id=0, memory_gb=None)

        batch_size = HardwareDetector.estimate_batch_size(device_info, 100.0)
        assert batch_size >= 1


@pytest.fixture
def mock_torch():
    """Mock torch imports for testing."""
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        yield


class TestDeviceInfo:
    """Test cases for DeviceInfo dataclass."""

    def test_device_info_creation(self):
        """Test DeviceInfo creation with all fields."""
        device_info = DeviceInfo(device_type=DeviceType.CUDA, device_name="cuda:0 (NVIDIA GTX 1080)", hostname="test-machine", device_id=0, memory_gb=11)

        assert device_info.device_type == DeviceType.CUDA
        assert device_info.device_name == "cuda:0 (NVIDIA GTX 1080)"
        assert device_info.hostname == "test-machine"
        assert device_info.device_id == 0
        assert device_info.memory_gb == 11

    def test_device_info_optional_fields(self):
        """Test DeviceInfo creation with optional fields."""
        device_info = DeviceInfo(device_type=DeviceType.CPU, device_name="cpu", hostname="cpu-machine")

        assert device_info.device_type == DeviceType.CPU
        assert device_info.device_name == "cpu"
        assert device_info.hostname == "cpu-machine"
        assert device_info.device_id is None
        assert device_info.memory_gb is None


class TestDeviceType:
    """Test cases for DeviceType enum."""

    def test_device_type_values(self):
        """Test DeviceType enum values."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"

    def test_device_type_comparison(self):
        """Test DeviceType comparison."""
        assert DeviceType.CPU == DeviceType.CPU
        assert DeviceType.CUDA == DeviceType.CUDA
        assert DeviceType.MPS == DeviceType.MPS

        assert DeviceType.CPU != DeviceType.CUDA  # type: ignore[comparison-overlap]
        assert DeviceType.CPU != DeviceType.MPS  # type: ignore[comparison-overlap]
        assert DeviceType.CUDA != DeviceType.MPS  # type: ignore[comparison-overlap]
