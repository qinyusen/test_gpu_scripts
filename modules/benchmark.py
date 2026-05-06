"""GPU benchmark module — nvbandwidth + PyTorch compute throughput."""

import json
import shutil
import subprocess
import time
from datetime import datetime
from typing import Optional, List

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from modules.gpu_specs import detect_gpu_type, get_gpu_specs, get_gpu_label

TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
except ImportError:
    pass


class Benchmark:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.bench_cfg = config.get("benchmark", {})
        self.tools_dir = config.get("tools", {}).get("install_dir", "/opt/h200-test-tools")
        self.gpu_type = detect_gpu_type()
        self.specs = get_gpu_specs(self.gpu_type)
        self.gpu_label = get_gpu_label(self.gpu_type)

    def run(self) -> dict:
        results = {}
        results.update(self.run_memory_benchmark())
        results.update(self.run_compute_benchmark())
        return results

    def _find_nvbandwidth(self) -> Optional[str]:
        p = shutil.which("nvbandwidth")
        if p:
            return p
        local = shutil.os.path.join(self.tools_dir, "nvbandwidth", "nvbandwidth")
        if shutil.os.path.isfile(local) and shutil.os.access(local, shutil.os.X_OK):
            return local
        return None

    def run_memory_benchmark(self) -> dict:
        nvbw = self._find_nvbandwidth()

        if nvbw:
            return self._run_nvbandwidth(nvbw)

        self.console.print("[yellow]nvbandwidth not found, falling back to PyTorch[/yellow]")
        return self._run_memory_pytorch()

    def _run_nvbandwidth(self, nvbw_path: str) -> dict:
        mem_cfg = self.bench_cfg.get("memory", {})
        buffer_mb = mem_cfg.get("nvbandwidth_buffer_mb", 512)
        samples = mem_cfg.get("nvbandwidth_samples", 3)

        self.console.print(f"[cyan]Memory Benchmark via nvbandwidth ({nvbw_path})[/cyan]")

        results_by_test = {}
        per_gpu_d2d = []

        testcases = [
            "host_to_device_memcpy_read_ce",
            "device_to_host_memcpy_write_ce",
            "device_to_device_memcpy_write_ce",
            "device_to_device_memcpy_read_ce",
            "device_to_device_bidirectional_sm",
        ]

        try:
            list_r = subprocess.run(
                [nvbw_path, "-l", "-j"],
                capture_output=True, text=True, timeout=15,
            )
            available = []
            if list_r.returncode == 0:
                try:
                    avail_list = json.loads(list_r.stdout)
                    available = [t.get("name", "") for t in avail_list if isinstance(t, dict)]
                except json.JSONDecodeError:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            available = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=self.console,
        ) as progress:
            task = progress.add_task("nvbandwidth tests...", total=len(testcases))

            for tc in testcases:
                if available and tc not in available:
                    progress.advance(task)
                    continue

                try:
                    cmd = [
                        nvbw_path,
                        f"-b{buffer_mb}",
                        f"-i{samples}",
                        "-j",
                        f"-t{tc}",
                    ]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                    if r.returncode == 0 and r.stdout.strip():
                        try:
                            data = json.loads(r.stdout)
                            bw_values = []
                            for entry in data if isinstance(data, list) else [data]:
                                if isinstance(entry, dict):
                                    for row in entry.get("results", []):
                                        val = row.get("value", 0)
                                        if isinstance(val, (int, float)):
                                            bw_values.append(val)
                            avg_bw = sum(bw_values) / len(bw_values) if bw_values else 0
                            results_by_test[tc] = round(avg_bw, 1)
                        except json.JSONDecodeError:
                            results_by_test[tc] = 0
                    else:
                        results_by_test[tc] = 0
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    results_by_test[tc] = 0

                progress.advance(task)

        d2d_bw = max(
            results_by_test.get("device_to_device_memcpy_write_ce", 0),
            results_by_test.get("device_to_device_memcpy_read_ce", 0),
            results_by_test.get("device_to_device_bidirectional_sm", 0),
        )
        h2d_bw = results_by_test.get("host_to_device_memcpy_read_ce", 0)
        d2h_bw = results_by_test.get("device_to_host_memcpy_write_ce", 0)
        efficiency = (d2d_bw / self.specs["memory_bandwidth_gbps"]) * 100 if d2d_bw else 0

        return {
            "memory": {
                "source": "nvbandwidth",
                "h2d_bandwidth_gbps": round(h2d_bw, 1),
                "d2h_bandwidth_gbps": round(d2h_bw, 1),
                "d2d_bandwidth_gbps": round(d2d_bw, 1),
                "peak_bandwidth_gbps": self.specs["memory_bandwidth_gbps"],
                "efficiency_pct": round(efficiency, 1),
                "results_by_test": results_by_test,
                "per_gpu": per_gpu_d2d,
            }
        }

    def _run_memory_pytorch(self) -> dict:
        mem_cfg = self.bench_cfg.get("memory", {})
        test_sizes_mb = [1, 4, 16, 64, 256, 1024, 4096]
        iterations = mem_cfg.get("iterations", 10)

        if not TORCH_AVAILABLE:
            self.console.print("[yellow]PyTorch not available - skipping memory benchmark[/yellow]")
            return {"memory": {"error": "pytorch_not_available"}}

        gpu_count = torch.cuda.device_count()
        self.console.print(f"[cyan]Memory Benchmark (PyTorch fallback) - {gpu_count} GPU(s)[/cyan]")

        bandwidth_by_size = {}

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=self.console,
        ) as progress:
            task = progress.add_task("Testing sizes...", total=len(test_sizes_mb))

            for size_mb in test_sizes_mb:
                size_bytes = size_mb * 1024 * 1024
                h2d_times, d2h_times, d2d_times = [], [], []
                x_cpu = torch.randn(size_bytes // 4, dtype=torch.float32)

                for _ in range(iterations):
                    t0 = time.perf_counter()
                    x_gpu = x_cpu.cuda()
                    torch.cuda.synchronize()
                    h2d_times.append(time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    x_gpu.cpu()
                    torch.cuda.synchronize()
                    d2h_times.append(time.perf_counter() - t0)

                    x_gpu2 = torch.randn_like(x_gpu)
                    t0 = time.perf_counter()
                    x_gpu2.copy_(x_gpu)
                    torch.cuda.synchronize()
                    d2d_times.append(time.perf_counter() - t0)

                    del x_gpu, x_gpu2
                    torch.cuda.empty_cache()

                def median(lst):
                    s = sorted(lst)
                    return s[len(s) // 2]

                def bw_gb(t, sz):
                    return (sz / t) / 1e9

                bandwidth_by_size[str(size_mb)] = {
                    "h2d_gbps": round(bw_gb(median(h2d_times), size_bytes), 1),
                    "d2h_gbps": round(bw_gb(median(d2h_times), size_bytes), 1),
                    "d2d_gbps": round(bw_gb(median(d2d_times), size_bytes), 1),
                }
                progress.advance(task)

        best_d2d = max(v["d2d_gbps"] for v in bandwidth_by_size.values())
        efficiency = (best_d2d / self.specs["memory_bandwidth_gbps"]) * 100

        return {
            "memory": {
                "source": "pytorch",
                "h2d_bandwidth_gbps": round(max(v["h2d_gbps"] for v in bandwidth_by_size.values()), 1),
                "d2h_bandwidth_gbps": round(max(v["d2h_gbps"] for v in bandwidth_by_size.values()), 1),
                "d2d_bandwidth_gbps": round(best_d2d, 1),
                "peak_bandwidth_gbps": self.specs["memory_bandwidth_gbps"],
                "efficiency_pct": round(efficiency, 1),
                "test_sizes_mb": test_sizes_mb,
                "bandwidth_by_size": bandwidth_by_size,
                "per_gpu": [],
            }
        }

    def run_compute_benchmark(self, dtypes: Optional[List[str]] = None) -> dict:
        comp_cfg = self.bench_cfg.get("compute", {})
        configured_dtypes = dtypes or comp_cfg.get("dtypes", ["fp32", "tf32", "fp16", "bf16", "fp8"])
        matrix_size = comp_cfg.get("matrix_size", 4096)
        warmup = comp_cfg.get("warmup", 10)
        iterations = comp_cfg.get("iterations", 100)

        if not TORCH_AVAILABLE:
            self.console.print("[yellow]PyTorch not available - skipping compute benchmark[/yellow]")
            return {"compute": {"error": "pytorch_not_available"}}

        gpu_count = torch.cuda.device_count()
        self.console.print(f"[cyan]Compute Benchmark - {gpu_count} GPU(s)[/cyan]")

        dtype_map = {
            "fp32": (torch.float32, self.specs["fp32_tflops"]),
            "tf32": ("tf32", self.specs["tf32_tflops"]),
            "fp16": (torch.float16, self.specs["fp16_tflops"]),
            "bf16": (torch.bfloat16, self.specs["bf16_tflops"]),
            "fp8": (torch.float8_e4m3fn, self.specs["fp8_tflops"]),
        }

        results_by_dtype = {}
        per_gpu_results = [{"index": i} for i in range(gpu_count)]

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=self.console,
        ) as progress:
            task = progress.add_task("Testing dtypes...", total=len(configured_dtypes))

            for dtype_name in configured_dtypes:
                if dtype_name not in dtype_map:
                    progress.advance(task)
                    continue

                dtype_val, peak_tflops = dtype_map[dtype_name]

                try:
                    if dtype_name == "tf32":
                        old_tf32 = torch.backends.cuda.matmul.allow_tf32
                        torch.backends.cuda.matmul.allow_tf32 = True
                        dtype_val = torch.float32

                    M = N = K = matrix_size

                    if dtype_name == "fp8":
                        a = torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
                        b = torch.randn(K, N, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
                    else:
                        a = torch.randn(M, K, device="cuda", dtype=dtype_val)
                        b = torch.randn(K, N, device="cuda", dtype=dtype_val)

                    for _ in range(warmup):
                        torch.matmul(a, b)
                    torch.cuda.synchronize()

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    for _ in range(iterations):
                        c = torch.matmul(a, b)
                    end_event.record()
                    torch.cuda.synchronize()

                    elapsed_ms = start_event.elapsed_time(end_event)
                    flops = 2 * M * N * K * iterations
                    tflops = flops / (elapsed_ms / 1000) / 1e12
                    results_by_dtype[dtype_name] = round(tflops, 1)

                    for pg in per_gpu_results:
                        pg[dtype_name] = round(tflops, 1)

                    if dtype_name == "tf32":
                        torch.backends.cuda.matmul.allow_tf32 = old_tf32

                    del a, b, c
                    torch.cuda.empty_cache()

                except Exception as e:
                    results_by_dtype[dtype_name] = f"error: {e}"
                    self.console.print(f"[yellow]  {dtype_name}: {e}[/yellow]")

                progress.advance(task)

        efficiency = {}
        for dt, achieved in results_by_dtype.items():
            if isinstance(achieved, (int, float)) and dt in dtype_map:
                efficiency[dt] = round((achieved / dtype_map[dt][1]) * 100, 1)

        return {
            "compute": {
                "per_dtype_tflops": results_by_dtype,
                "peak_tflops": {dt: dtype_map[dt][1] for dt in dtype_map},
                "efficiency_pct": efficiency,
                "per_gpu": per_gpu_results,
                "matrix_size": matrix_size,
                "warmup": warmup,
                "iterations": iterations,
            }
        }

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()

        if "memory" in results and "error" not in results["memory"]:
            mem = results["memory"]
            source = mem.get("source", "unknown")
            c.print(f"\n[bold cyan]Memory Bandwidth Results (via {source})[/bold cyan]")

            table = Table(box=None, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            table.add_column("Peak", justify="right")
            table.add_column("Efficiency", justify="right")

            for label, achieved, peak in [
                ("H2D (PCIe)", mem["h2d_bandwidth_gbps"], None),
                ("D2H (PCIe)", mem["d2h_bandwidth_gbps"], None),
                ("D2D (HBM3e)", mem["d2d_bandwidth_gbps"], mem["peak_bandwidth_gbps"]),
            ]:
                val_str = f"{achieved:.1f} GB/s" if isinstance(achieved, (int, float)) else "N/A"
                peak_str = f"{peak:.0f} GB/s" if peak else "N/A"
                if peak and isinstance(achieved, (int, float)) and achieved > 0:
                    eff = (achieved / peak) * 100
                    ec = "green" if eff >= 80 else ("yellow" if eff >= 50 else "red")
                    eff_str = f"[{ec}]{eff:.1f}%[/{ec}]"
                else:
                    eff_str = "N/A"
                table.add_row(label, val_str, peak_str, eff_str)

            c.print(table)

            by_test = mem.get("results_by_test", {})
            if by_test:
                c.print("\n  [dim]nvbandwidth breakdown:[/dim]")
                for tc, bw in sorted(by_test.items()):
                    c.print(f"    {tc}: {bw} GB/s")

            by_size = mem.get("bandwidth_by_size", {})
            if by_size:
                t2 = Table(title="Bandwidth by Transfer Size", box=None, padding=(0, 1))
                t2.add_column("Size (MB)", style="bold", justify="right")
                t2.add_column("H2D (GB/s)", justify="right")
                t2.add_column("D2H (GB/s)", justify="right")
                t2.add_column("D2D (GB/s)", justify="right")
                for sz, vals in sorted(by_size.items(), key=lambda x: int(x[0])):
                    peak = mem["peak_bandwidth_gbps"]
                    d2d_eff = (vals["d2d_gbps"] / peak) * 100
                    ec = "green" if d2d_eff >= 80 else ("yellow" if d2d_eff >= 50 else "red")
                    t2.add_row(sz, f"{vals['h2d_gbps']:.1f}", f"{vals['d2h_gbps']:.1f}",
                               f"[{ec}]{vals['d2d_gbps']:.1f}[/{ec}]")
                c.print(t2)

        if "compute" in results and "error" not in results["compute"]:
            comp = results["compute"]
            c.print(f"\n[bold cyan]Compute Throughput Results[/bold cyan]")

            table = Table(box=None, padding=(0, 1))
            table.add_column("DType", style="bold")
            table.add_column("Achieved (TFLOPS)", justify="right")
            table.add_column("Peak", justify="right")
            table.add_column("Efficiency", justify="right")

            peak = comp.get("peak_tflops", {})
            per_dtype = comp.get("per_dtype_tflops", {})
            eff = comp.get("efficiency_pct", {})

            for dt in per_dtype:
                achieved = per_dtype[dt]
                if isinstance(achieved, str):
                    table.add_row(dt, f"[red]{achieved}[/red]", str(peak.get(dt, "N/A")), "N/A")
                    continue
                pk = peak.get(dt, 0)
                ef = eff.get(dt, 0)
                ec = "green" if ef >= 80 else ("yellow" if ef >= 50 else "red")
                table.add_row(dt.upper(), f"{achieved:.1f}", f"{pk:.0f}",
                              f"[{ec}]{ef:.1f}%[/{ec}]")
            c.print(table)
