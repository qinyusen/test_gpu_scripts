"""GPU specifications database for NVIDIA datacenter GPUs."""

import shutil
import subprocess

# GPU name patterns -> internal key mapping
GPU_NAME_PATTERNS = {
    "H100": "h100",
    "H200": "h200",
    "B200": "b200",
    "B300": "b300",
}

# Specs database — ALL values are DENSE (non-sparse) TFLOPS
GPU_SPECS = {
    "h100": {
        "full_name": "NVIDIA H100 SXM5",
        "architecture": "Hopper",
        "compute_capability": 9.0,
        "hbm_capacity_gb": 80,
        "hbm_type": "HBM3",
        "memory_bandwidth_gbps": 3400,      # GB/s (3.4 TB/s)
        "fp32_tflops": 67,
        "tf32_tflops": 495,                 # dense (989 sparse)
        "fp16_tflops": 990,                 # dense (1979 sparse w/ 2:4)
        "bf16_tflops": 990,                 # dense
        "fp8_tflops": 1979,                 # dense
        "tdp_watts": 700,
        "nvlink_gen": 4,
        "nvlink_bandwidth_gbps": 900,       # bidirectional
        "pcie_gen": 5,
        "min_driver_version": "535",
        "min_cuda_version": "12.1",
    },
    "h200": {
        "full_name": "NVIDIA H200 SXM",
        "architecture": "Hopper",
        "compute_capability": 9.0,
        "hbm_capacity_gb": 141,
        "hbm_type": "HBM3e",
        "memory_bandwidth_gbps": 4800,      # GB/s (4.8 TB/s) — THIS IS THE CORRECT VALUE, NOT 989!
        "fp32_tflops": 67,
        "tf32_tflops": 495,                 # dense
        "fp16_tflops": 990,                 # dense
        "bf16_tflops": 990,                 # dense
        "fp8_tflops": 1979,                 # dense
        "tdp_watts": 700,
        "nvlink_gen": 4,
        "nvlink_bandwidth_gbps": 900,
        "pcie_gen": 5,
        "min_driver_version": "535",
        "min_cuda_version": "12.1",
    },
    "b200": {
        "full_name": "NVIDIA B200 SXM",
        "architecture": "Blackwell",
        "compute_capability": 10.0,
        "hbm_capacity_gb": 180,
        "hbm_type": "HBM3e",
        "memory_bandwidth_gbps": 8000,      # GB/s (8 TB/s)
        "fp32_tflops": 90,
        "tf32_tflops": 1125,                # dense
        "fp16_tflops": 2250,                # dense
        "bf16_tflops": 2250,                # dense
        "fp8_tflops": 4500,                 # dense
        "tdp_watts": 1000,
        "nvlink_gen": 5,
        "nvlink_bandwidth_gbps": 1800,
        "pcie_gen": 5,
        "min_driver_version": "550",
        "min_cuda_version": "12.4",
    },
    "b300": {
        "full_name": "NVIDIA B300 SXM (Blackwell Ultra)",
        "architecture": "Blackwell Ultra",
        "compute_capability": 10.0,
        "hbm_capacity_gb": 288,
        "hbm_type": "HBM3e",
        "memory_bandwidth_gbps": 8000,      # GB/s (8 TB/s)
        "fp32_tflops": 125,
        "tf32_tflops": 1750,                # dense (estimated)
        "fp16_tflops": 3500,                # dense
        "bf16_tflops": 3500,                # dense
        "fp8_tflops": 7000,                 # dense
        "tdp_watts": 1200,
        "nvlink_gen": 5,
        "nvlink_bandwidth_gbps": 1800,
        "pcie_gen": 5,
        "min_driver_version": "550",
        "min_cuda_version": "12.4",
    },
}

# Fallback for unknown / unsupported GPUs
_UNKNOWN_SPECS = {
    "full_name": "Unknown GPU",
    "architecture": "unknown",
    "compute_capability": 0.0,
    "hbm_capacity_gb": 0,
    "hbm_type": "unknown",
    "memory_bandwidth_gbps": 0,
    "fp32_tflops": 0,
    "tf32_tflops": 0,
    "fp16_tflops": 0,
    "bf16_tflops": 0,
    "fp8_tflops": 0,
    "tdp_watts": 700,
    "nvlink_gen": 0,
    "nvlink_bandwidth_gbps": 0,
    "pcie_gen": 0,
    "min_driver_version": "",
    "min_cuda_version": "",
}


def detect_gpu_type() -> str:
    """Detect GPU type via nvidia-smi and return the internal key (e.g. 'h200').

    Returns 'unknown' if nvidia-smi is unavailable or the GPU is not recognized.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "unknown"

    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return "unknown"

        first_line = r.stdout.strip().splitlines()[0].strip()
        for pattern, key in GPU_NAME_PATTERNS.items():
            if pattern in first_line.upper():
                return key
        return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"


def get_gpu_specs(gpu_type: str = None) -> dict:
    """Return specs dict for the given gpu_type, auto-detecting if None.

    Returns a minimal 'unknown' fallback dict with zero peaks for unsupported GPUs.
    """
    if gpu_type is None:
        gpu_type = detect_gpu_type()
    return GPU_SPECS.get(gpu_type, dict(_UNKNOWN_SPECS))


def get_supported_gpus() -> list:
    """Return list of supported GPU type keys."""
    return list(GPU_SPECS.keys())


def get_gpu_label(gpu_type: str) -> str:
    """Return a short human-readable label like 'H200 SXM' for display in tables."""
    specs = GPU_SPECS.get(gpu_type)
    if specs:
        full = specs["full_name"]
        # Strip the "NVIDIA " prefix for display
        return full.replace("NVIDIA ", "")
    return "Unknown GPU"
