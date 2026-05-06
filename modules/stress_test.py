"""GPU stress test module — wraps gpu-burn for long-running stability tests."""

import glob
import os
import shutil
import subprocess
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text

from modules.gpu_specs import resolve_tools_dir


class StressTest:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.stress_cfg = config.get("stress", {})
        self.tools_dir = resolve_tools_dir(config)

    def _find_gpu_burn(self) -> str:
        p = shutil.which("gpu_burn")
        if p:
            return p

        local = os.path.join(self.tools_dir, "gpu-burn", "gpu_burn")
        if os.path.isfile(local) and os.access(local, shutil.os.X_OK):
            return local

        matches = glob.glob(os.path.join(self.tools_dir, "gpu-burn", "**", "gpu_burn"), recursive=True)
        for m in matches:
            if os.access(m, shutil.os.X_OK):
                return m
        return ""

    def run(self) -> dict:
        cfg = self.stress_cfg
        duration_sec = cfg.get("duration_sec", 60)
        use_doubles = cfg.get("use_doubles", False)
        use_tensor_cores = cfg.get("use_tensor_cores", True)
        memory_pct = cfg.get("memory_pct", 90)
        target_gpus = cfg.get("gpus", "all")

        gpu_burn = self._find_gpu_burn()

        if gpu_burn:
            return self._run_gpu_burn(gpu_burn, duration_sec, use_doubles, use_tensor_cores, target_gpus)

        self.console.print("[yellow]gpu_burn not found, falling back to PyTorch stress test[/yellow]")
        return self._run_pytorch_stress(duration_sec)

    def _run_gpu_burn(self, gpu_burn: str, duration: int,
                      doubles: bool, tensor_cores: bool, target_gpus: str) -> dict:
        self.console.print(f"[cyan]GPU Stress Test via gpu-burn ({duration}s)[/cyan]")

        cmd = [gpu_burn]
        if doubles:
            cmd.append("-d")
        if tensor_cores:
            cmd.append("-tc")
        if target_gpus != "all":
            cmd.extend(["-i", str(target_gpus)])
        cmd.append(str(duration))

        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 120)
            elapsed = round(time.time() - t0, 1)

            output = r.stdout + r.stderr
            passed = r.returncode == 0

            gpu_results = []
            for line in output.split("\n"):
                line = line.strip()
                if "GPU" in line and ("PASS" in line.upper() or "FAIL" in line.upper()):
                    gpu_results.append(line)

            return {
                "source": "gpu-burn",
                "passed": passed,
                "duration_sec": duration,
                "elapsed_sec": elapsed,
                "gpu_results": gpu_results,
                "raw_output_tail": output[-500:] if output else "",
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            return {
                "source": "gpu-burn",
                "passed": False,
                "duration_sec": duration,
                "error": "timeout",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "source": "gpu-burn",
                "passed": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _run_pytorch_stress(self, duration: int) -> dict:
        try:
            import torch
            if not torch.cuda.is_available():
                return {"error": "pytorch_not_available"}
        except ImportError:
            return {"error": "pytorch_not_available"}

        gpu_count = torch.cuda.device_count()
        self.console.print(f"[cyan]PyTorch Stress Test ({duration}s, {gpu_count} GPUs)[/cyan]")

        gpu_status = {}
        t0 = time.time()

        try:
            tensors = {}
            for i in range(gpu_count):
                with torch.cuda.device(i):
                    props = torch.cuda.get_device_properties(i)
                    total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
                    alloc_size = int(total_mem * 0.9) // 4
                    tensors[i] = torch.randn(alloc_size, device=f"cuda:{i}", dtype=torch.float32)

            while time.time() - t0 < duration:
                for i in range(gpu_count):
                    with torch.cuda.device(i):
                        tensors[i] = torch.matmul(tensors[i][:2048, :2048], tensors[i][:2048, :2048].T)
                        torch.cuda.synchronize()
                time.sleep(0.1)

            for i in range(gpu_count):
                gpu_status[i] = "PASS"

        except RuntimeError as e:
            for i in range(gpu_count):
                if i not in gpu_status:
                    gpu_status[i] = "FAIL"
            return {
                "source": "pytorch",
                "passed": False,
                "duration_sec": duration,
                "error": str(e),
                "gpu_status": gpu_status,
            }
        finally:
            tensors.clear()
            torch.cuda.empty_cache()

        elapsed = round(time.time() - t0, 1)
        return {
            "source": "pytorch",
            "passed": True,
            "duration_sec": duration,
            "elapsed_sec": elapsed,
            "gpu_status": gpu_status,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()
        if "error" in results:
            c.print(f"[bold red]Error: {results['error']}[/bold red]")
            return

        passed = results.get("passed", False)
        source = results.get("source", "unknown")
        duration = results.get("duration_sec", "?")
        elapsed = results.get("elapsed_sec", "?")

        verdict = "[bold green]✓ Stress Test PASSED[/bold green]" if passed else "[bold red]✗ Stress Test FAILED[/bold red]"
        c.print(f"\n{verdict} [dim](via {source})[/dim]")
        c.print(f"  Target duration: {duration}s | Actual: {elapsed}s")

        gpu_results = results.get("gpu_results", [])
        if gpu_results:
            c.print("\n  Per-GPU results:")
            for line in gpu_results:
                if "FAIL" in line.upper():
                    c.print(f"    [red]{line}[/red]")
                else:
                    c.print(f"    [green]{line}[/green]")

        gpu_status = results.get("gpu_status", {})
        if gpu_status:
            c.print("\n  Per-GPU status:")
            for gid, status in sorted(gpu_status.items()):
                color = "green" if status == "PASS" else "red"
                c.print(f"    GPU {gid}: [{color}]{status}[/{color}]")

        if results.get("error"):
            c.print(f"  [red]Error: {results['error']}[/red]")
