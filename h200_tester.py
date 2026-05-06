#!/usr/bin/env python3
"""GPU Training Server Test Suite (H100/H200/B200/B300) - Main CLI Entry Point."""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich import box

from modules.gpu_info import GPUInfo
from modules.health_check import HealthCheck
from modules.benchmark import Benchmark
from modules.nccl_test import NCCLTest
from modules.training_sim import TrainingSim
from modules.stress_test import StressTest
from modules.rdma_test import RDMATest
from modules.report import ReportGenerator
from modules.gpu_specs import detect_gpu_type, get_gpu_specs, get_gpu_label, get_supported_gpus

DEFAULT_CONFIG = {
    "benchmark": {
        "memory": {"size_mb": 4096, "iterations": 10, "nvbandwidth_buffer_mb": 512, "nvbandwidth_samples": 3},
        "compute": {
            "dtypes": ["fp32", "tf32", "fp16", "bf16", "fp8"],
            "matrix_size": 4096,
            "warmup": 10,
            "iterations": 100,
        },
    },
    "health": {"temp_warning": 80, "temp_critical": 90, "power_limit": None},
    "nccl": {
        "min_bandwidth_gbps": None,
        "test_allreduce": True,
        "test_alltoall": True,
        "test_broadcast": True,
        "test_reduce_scatter": False,
        "test_allgather": False,
        "test_sendrecv": False,
    },
    "stress": {
        "duration_sec": 60,
        "use_doubles": False,
        "use_tensor_cores": True,
        "memory_pct": 90,
        "gpus": "all",
    },
    "rdma": {
        "min_bandwidth_gbps": 50,
        "max_latency_us": 10,
        "ib_iterations": 1000,
        "msg_size": 65536,
        "ib_device": None,
        "ib_port": 1,
    },
    "training": {
        "model": "gpt2",
        "batch_size": 8,
        "seq_length": 2048,
        "num_steps": 50,
        "dtype": "bf16",
    },
    "report": {"output_dir": "./reports", "format": "json"},
    "tools": {"install_dir": "/opt/h200-test-tools"},
}

BANNER = r"""
[bold cyan]
╔══════════════════════════════════════════════════╗
║                                                  ║
║       GPU Training Server Test Suite             ║
║       Diagnostics & Benchmarking Tool            ║
║       Supports: H100 / H200 / B200 / B300        ║
║                                                  ║
╚══════════════════════════════════════════════════╝
[/bold cyan]
"""


def load_config() -> dict:
    """Load config from yaml file, fallback to defaults."""
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    return DEFAULT_CONFIG.copy()


def check_prerequisites(console: Console) -> bool:
    """Check if required tools are available."""
    import shutil

    ok = True
    if not shutil.which("nvidia-smi"):
        console.print("[bold red]ERROR: nvidia-smi not found![/bold red]")
        console.print("  Please install NVIDIA drivers first.")
        ok = False
    return ok


def interactive_menu(config: dict):
    """Run interactive menu loop."""
    console = Console()

    console.print(BANNER)

    gpu_type = detect_gpu_type()
    gpu_label = get_gpu_label(gpu_type)
    if gpu_type != "unknown":
        console.print(f"[bold green]Detected GPU: {gpu_label} ({gpu_type.upper()})[/bold green]\n")
    else:
        console.print("[yellow]GPU type could not be auto-detected. Using default thresholds.[/yellow]\n")

    if not check_prerequisites(console):
        return

    results_store: dict = {"timestamp": datetime.now().isoformat(), "tests": {}}

    menu_items = [
        ("1", "GPU Information", "gpu_info"),
        ("2", "Health Check", "health"),
        ("3", "Memory Benchmark (nvbandwidth)", "memory_bench"),
        ("4", "Compute Benchmark", "compute_bench"),
        ("5", "NCCL Multi-GPU Test", "nccl"),
        ("6", "GPU Stress Test (gpu-burn)", "stress"),
        ("7", "RDMA/IB Test", "rdma"),
        ("8", "Training Simulation", "training"),
        ("9", "Full Test Suite (All Tests)", "all"),
        ("0", "Generate Report", "report"),
    ]

    while True:
        console.print()
        table = Table(
            title="[bold cyan]Select a Test[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            show_header=False,
            padding=(0, 2),
        )
        table.add_column("Key", style="bold yellow", width=5)
        table.add_column("Test Name")
        table.add_column("Description", style="dim")
        descriptions = {
            "gpu_info": "Detect GPUs, show specs & NVLink topology",
            "health": "Temperature, power, ECC errors, PCIe, DCGM",
            "memory_bench": "HBM bandwidth via nvbandwidth",
            "compute_bench": "GEMM TFLOPS across FP32/TF32/FP16/BF16/FP8",
            "nccl": "AllReduce, AllToAll, Broadcast via nccl-tests",
            "stress": "Long-running GPU stress via gpu-burn",
            "rdma": "InfiniBand bandwidth & latency (ib_write_bw)",
            "training": "Simulate LLM training with PyTorch",
            "all": "Run all tests sequentially",
            "report": "Export results to JSON/HTML",
        }
        for key, name, action in menu_items:
            table.add_row(f"[{key}]", name, descriptions.get(action, ""))
        table.add_row("[q]", Text("Quit", style="bold red"), "Exit the program")

        console.print(table)
        choice = console.input("\n[bold green]Enter choice > [/bold green]").strip().lower()

        if choice == "q":
            if results_store.get("tests"):
                _save_results_prompt(results_store, config, console)
            console.print("[dim]Goodbye![/dim]")
            break

        action_map = {item[0]: item[2] for item in menu_items}
        action = action_map.get(choice)
        if action is None:
            console.print(f"[yellow]Invalid choice: {choice}[/yellow]")
            continue

        result = _run_test(action, config, console)
        if result:
            if result.get("__report__"):
                if results_store.get("tests"):
                    rg = ReportGenerator(config)
                    rg.generate(results_store)
                else:
                    console.print("[yellow]No test results to export. Run tests first.[/yellow]")
            else:
                results_store["tests"][action] = result

    return results_store


def _save_results_prompt(results_store: dict, config: dict, console: Console):
    if not results_store.get("tests"):
        return
    save = console.input("[bold green]Save results before quitting? [y/N]: [/bold green]").strip().lower()
    if save == "y":
        rg = ReportGenerator(config)
        rg.generate(results_store)


def _run_test(test_name: str, config: dict, console: Console) -> dict:
    """Execute a single test by name."""
    try:
        if test_name == "gpu_info":
            m = GPUInfo(config)
            result = m.run()
            m.print_results(result)
            return result

        elif test_name == "health":
            m = HealthCheck(config)
            result = m.run()
            m.print_results(result)
            return result

        elif test_name == "memory_bench":
            m = Benchmark(config)
            result = m.run_memory_benchmark()
            Benchmark.print_results(result)
            return result

        elif test_name == "compute_bench":
            m = Benchmark(config)
            result = m.run_compute_benchmark()
            Benchmark.print_results(result)
            return result

        elif test_name == "nccl":
            m = NCCLTest(config)
            result = m.run()
            m.print_results(result)
            return result

        elif test_name == "stress":
            m = StressTest(config)
            result = m.run()
            m.print_results(result)
            return result

        elif test_name == "rdma":
            m = RDMATest(config)
            result = m.run()
            m.print_results(result)
            return result

        elif test_name == "training":
            m = TrainingSim(config)
            result = m.run()
            m.print_results(result)
            return result

        elif test_name == "all":
            return _run_full_suite(config, console)

        elif test_name == "report":
            return {"__report__": True}

    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user.[/yellow]")
        return {"error": "interrupted"}
    except Exception as e:
        console.print(f"[bold red]Test failed: {e}[/bold red]")
        return {"error": str(e)}


def _run_full_suite(config: dict, console: Console) -> dict:
    """Run all tests sequentially."""
    console.print(Panel("[bold cyan]Running Full Test Suite[/bold cyan]", box=box.DOUBLE))
    all_results: dict = {"timestamp": datetime.now().isoformat()}
    tests = [
        ("gpu_info", "GPU Information", GPUInfo),
        ("health", "Health Check", HealthCheck),
        ("memory_bench", "Memory Benchmark", lambda c: Benchmark(c)),
        ("compute_bench", "Compute Benchmark", lambda c: Benchmark(c)),
        ("nccl", "NCCL Test", NCCLTest),
        ("stress", "GPU Stress Test", StressTest),
        ("rdma", "RDMA/IB Test", RDMATest),
        ("training", "Training Simulation", TrainingSim),
    ]

    for i, (key, name, mod_cls) in enumerate(tests, 1):
        console.print(f"\n[bold cyan][{i}/{len(tests)}] {name}[/bold cyan]")
        try:
            mod = mod_cls(config)
            if key == "memory_bench":
                result = mod.run_memory_benchmark()
                mod.print_results(result)
            elif key == "compute_bench":
                result = mod.run_compute_benchmark()
                mod.print_results(result)
            else:
                result = mod.run()
                mod.print_results(result)
            all_results[key] = result
        except Exception as e:
            console.print(f"[bold red]{name} FAILED: {e}[/bold red]")
            all_results[key] = {"error": str(e)}

    # Summary
    console.print("\n" + "=" * 60)
    passed = sum(1 for v in all_results.values() if not isinstance(v, dict) or "error" not in v)
    total = len(tests)
    color = "green" if passed == total else ("yellow" if passed > 0 else "red")
    console.print(f"[bold {color}]Suite complete: {passed}/{total} tests passed[/bold {color}]")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="GPU Training Server Test Suite (H100/H200/B200/B300)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
   python h200_tester.py                        # Interactive menu
   python h200_tester.py --gpu-type h200        # Override GPU type
   python h200_tester.py --test gpu-info        # GPU info
   python h200_tester.py --test health          # Health check
   python h200_tester.py --test benchmark --type memory
   python h200_tester.py --test benchmark --type compute --dtype fp16
   python h200_tester.py --test nccl            # NCCL test
   python h200_tester.py --test training        # Training sim
   python h200_tester.py --test all             # Full suite
   python h200_tester.py --report --format json --output report.json
        """,
    )
    parser.add_argument("--test", choices=["gpu-info", "health", "benchmark", "nccl", "stress", "rdma", "training", "all"],
                        help="Run a specific test")
    parser.add_argument("--type", choices=["memory", "compute"], help="Benchmark type (with --test benchmark)")
    parser.add_argument("--dtype", choices=["fp32", "tf32", "fp16", "bf16", "fp8"],
                        help="Compute benchmark dtype (with --test benchmark --type compute)")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode")
    parser.add_argument("--report", action="store_true", help="Generate report from last results")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="Report format")
    parser.add_argument("--output", default=None, help="Report output file path")
    parser.add_argument("--config", default=None, help="Path to config YAML file")
    parser.add_argument("--gpu-type", choices=["auto", "h100", "h200", "b200", "b300"],
                        default="auto", help="Override GPU type detection")

    args = parser.parse_args()
    config = load_config()

    # Override config with CLI args (load before gpu_type so custom configs work)
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Set GPU type after config is finalized
    if args.gpu_type and args.gpu_type != "auto":
        config["gpu_type"] = args.gpu_type
    else:
        config["gpu_type"] = detect_gpu_type()

    console = Console()

    # Handle --report standalone
    if args.report and not args.test:
        console.print("[yellow]Run tests first to generate a report.[/yellow]")
        return

    # Interactive mode
    if args.interactive or not args.test:
        interactive_menu(config)
        return

    # CLI mode
    if not check_prerequisites(console):
        sys.exit(1)

    test_map = {
        "gpu-info": "gpu_info",
        "health": "health",
        "benchmark": None,
        "nccl": "nccl",
        "stress": "stress",
        "rdma": "rdma",
        "training": "training",
        "all": "all",
    }

    if args.test == "benchmark":
        bench = Benchmark(config)
        if args.type == "memory":
            result = bench.run_memory_benchmark()
            Benchmark.print_results(result)
        elif args.type == "compute":
            result = bench.run_compute_benchmark(dtypes=[args.dtype] if args.dtype else None)
            Benchmark.print_results(result)
        else:
            result = bench.run()
            Benchmark.print_results(result)
        if args.report:
            ReportGenerator(config).generate({"benchmark": result, "timestamp": datetime.now().isoformat()})
    elif args.test == "all":
        results = _run_full_suite(config, console)
        if args.report:
            ReportGenerator(config).generate(results)
        has_errors = any("error" in v for v in results.values() if isinstance(v, dict))
        sys.exit(1 if has_errors else 0)
    else:
        result = _run_test(test_map[args.test], config, console)
        if args.report and result:
            ReportGenerator(config).generate({args.test: result, "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    main()
