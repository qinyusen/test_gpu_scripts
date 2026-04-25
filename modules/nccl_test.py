"""NCCL multi-GPU communication test — wraps official nccl-tests."""

import glob
import os
import re
import shutil
import subprocess
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
except ImportError:
    pass


class NCCLTest:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.nccl_cfg = config.get("nccl", {})
        self.tools_dir = config.get("tools", {}).get("install_dir", "/opt/h200-test-tools")

    def _find_nccl_test(self, name: str) -> Optional[str]:
        p = shutil.which(name)
        if p:
            return p

        build_dir = os.path.join(self.tools_dir, "nccl-tests", "build")
        local = os.path.join(build_dir, name)
        if os.path.isfile(local) and os.access(local, shutil.os.X_OK):
            return local

        matches = glob.glob(os.path.join(self.tools_dir, "nccl-tests", "**", name), recursive=True)
        for m in matches:
            if os.access(m, shutil.os.X_OK):
                return m
        return None

    def _find_mpirun(self) -> Optional[str]:
        for cmd in ["mpirun", "mpiexec", os.path.join(self.tools_dir, "mpi", "bin", "mpirun")]:
            p = shutil.which(cmd)
            if p:
                return p
        return None

    def run(self) -> dict:
        gpu_count = 0
        if TORCH_AVAILABLE:
            gpu_count = torch.cuda.device_count()

        if gpu_count < 2:
            self.console.print(f"[yellow]NCCL test requires at least 2 GPUs (found {gpu_count})[/yellow]")
            return {"error": "need_at_least_2_gpus", "gpu_count": gpu_count}

        mpirun = self._find_mpirun()
        if not mpirun:
            self.console.print("[yellow]mpirun/mpiexec not found - falling back to torchrun[/yellow]")
            return self._run_torchrun_fallback(gpu_count)

        tests = []
        if self.nccl_cfg.get("test_allreduce", True):
            tests.append(("all_reduce_perf", "AllReduce"))
        if self.nccl_cfg.get("test_alltoall", True):
            tests.append(("alltoall_perf", "AllToAll"))
        if self.nccl_cfg.get("test_broadcast", True):
            tests.append(("broadcast_perf", "Broadcast"))
        if self.nccl_cfg.get("test_reduce_scatter", False):
            tests.append(("reduce_scatter_perf", "ReduceScatter"))
        if self.nccl_cfg.get("test_allgather", False):
            tests.append(("allgather_perf", "AllGather"))
        if self.nccl_cfg.get("test_sendrecv", False):
            tests.append(("sendrecv_perf", "SendRecv"))

        results = {}
        min_bw = self.nccl_cfg.get("min_bandwidth_gbps", 400)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(), console=self.console,
        ) as progress:
            task = progress.add_task("NCCL tests...", total=len(tests))

            for binary, label in tests:
                progress.update(task, description=f"NCCL {label}...")
                results[label.lower()] = self._run_one_nccl_test(
                    binary, label, gpu_count, mpirun, min_bw
                )
                progress.advance(task)

        all_passed = all(
            r.get("status") == "PASS"
            for r in results.values()
            if isinstance(r, dict) and "status" in r
        )

        return {
            "passed": all_passed,
            "source": "nccl-tests",
            "min_bandwidth_gbps": min_bw,
            "tests": results,
            "gpu_count": gpu_count,
            "timestamp": datetime.now().isoformat(),
        }

    def _run_one_nccl_test(self, binary_name: str, label: str,
                           gpu_count: int, mpirun: str, min_bw: float) -> dict:
        binary = self._find_nccl_test(binary_name)
        if not binary:
            return {"status": "SKIP", "error": f"{binary_name} not found"}

        sizes = "8:64:256:1024:4096:16384:65536:262144:1048576:4194304:16777216:67108864"

        ngpus_per_node = gpu_count
        cmd = [
            mpirun,
            "-np", str(ngpus_per_node),
            "--allow-run-as-root",
            "-x", "NCCL_DEBUG=WARN",
            "-x", "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(gpu_count)),
            binary,
            "-b", "8",
            "-e", "256M",
            "-f", "2",
            "-g", "1",
            "-w", "5",
            "-n", "20",
        ]

        try:
            env = os.environ.copy()
            env["NCCL_DEBUG"] = "WARN"
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)

            if r.returncode != 0:
                return {"status": "FAIL", "error": r.stderr[:300]}

            best_algbw = 0.0
            best_busbw = 0.0
            size_results = []

            for line in r.stdout.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        size = int(parts[0])
                        algbw = float(parts[-3]) if len(parts) >= 3 else 0
                        busbw = float(parts[-2]) if len(parts) >= 2 else 0
                        time_us = float(parts[2]) if len(parts) >= 3 else 0
                        size_results.append({
                            "size": size,
                            "time_us": time_us,
                            "algbw_gbps": algbw,
                            "busbw_gbps": busbw,
                        })
                        if busbw > best_busbw:
                            best_busbw = busbw
                        if algbw > best_algbw:
                            best_algbw = algbw
                    except (ValueError, IndexError):
                        continue

            status = "PASS" if best_busbw >= min_bw else "WARN"
            return {
                "status": status,
                "best_algbw_gbps": round(best_algbw, 1),
                "best_busbw_gbps": round(best_busbw, 1),
                "min_required_gbps": min_bw,
                "by_size": size_results[-5:] if size_results else [],
            }

        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "error": "timeout"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def _run_torchrun_fallback(self, gpu_count: int) -> dict:
        self.console.print("[cyan]Using torchrun fallback for NCCL test[/cyan]")
        min_bw = self.nccl_cfg.get("min_bandwidth_gbps", 400)
        size_mb = 64
        elements = size_mb * 1024 * 1024 // 4
        iters = 20

        code = f"""
import torch, torch.distributed as dist, time, os
os.environ.setdefault("MASTER_ADDR","127.0.0.1")
os.environ.setdefault("MASTER_PORT","29500")
os.environ.setdefault("NCCL_DEBUG","WARN")
rank=int(os.environ.get("LOCAL_RANK",0))
ws={gpu_count}
dist.init_process_group("nccl",rank=rank,world_size=ws)
torch.cuda.set_device(rank)
x=torch.randn({elements},device=f"cuda:{{rank}}",dtype=torch.float32)
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize()
s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
s.record()
for _ in range({iters}): dist.all_reduce(x)
e.record(); torch.cuda.synchronize()
ms=s.elapsed_time(e); gb=({elements}*4*{iters})/1e9; bw=gb/(ms/1000)
if rank==0: print(f"{{bw:.1f}}")
dist.destroy_process_group()
"""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp")
        tmp.write(code)
        tmp.close()

        try:
            r = subprocess.run(
                ["torchrun", f"--nproc_per_node={gpu_count}", tmp.name],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "NCCL_DEBUG": "WARN"},
            )
            os.unlink(tmp.name)
            lines = [l.strip() for l in r.stdout.split("\n") if l.strip()]
            bw = float(lines[-1]) if lines else 0
            status = "PASS" if bw >= min_bw else "WARN"
            return {
                "passed": status == "PASS",
                "source": "torchrun_fallback",
                "tests": {"allreduce": {
                    "status": status,
                    "best_busbw_gbps": round(bw, 1),
                    "min_required_gbps": min_bw,
                }},
                "gpu_count": gpu_count,
            }
        except Exception as e:
            return {"passed": False, "source": "torchrun_fallback", "error": str(e)}

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()
        if "error" in results:
            c.print(f"[bold red]Error: {results['error']}[/bold red]")
            return

        passed = results.get("passed", False)
        source = results.get("source", "unknown")
        verdict = "[bold green]✓ NCCL tests PASSED[/bold green]" if passed else "[bold yellow]⚠ NCCL tests WARNING[/bold yellow]"
        c.print(f"{verdict} [dim](via {source})[/dim]")

        tests = results.get("tests", {})
        for op_name, result in tests.items():
            if not isinstance(result, dict):
                continue
            c.print(f"\n[bold cyan]{op_name.upper()}[/bold cyan]")
            status = result.get("status", "FAIL")
            s_color = "green" if status == "PASS" else ("yellow" if status == "WARN" else "red")
            c.print(f"  Status: [{s_color}]{status}[/{s_color}]  "
                    f"Best bus BW: {result.get('best_busbw_gbps', 'N/A')} GB/s  "
                    f"(min: {result.get('min_required_gbps', 'N/A')} GB/s)")

            by_size = result.get("by_size", [])
            if by_size:
                t = Table(box=None, padding=(0, 1))
                t.add_column("Size", style="bold", justify="right")
                t.add_column("Time (us)", justify="right")
                t.add_column("Alg BW (GB/s)", justify="right")
                t.add_column("Bus BW (GB/s)", justify="right")
                for r in by_size:
                    sz = r.get("size", 0)
                    sz_str = f"{sz/1024:.0f}K" if sz < 1048576 else f"{sz/1048576:.0f}M"
                    t.add_row(sz_str, f"{r.get('time_us',0):.1f}",
                              f"{r.get('algbw_gbps',0):.1f}", f"{r.get('busbw_gbps',0):.1f}")
                c.print(t)
