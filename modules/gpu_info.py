"""GPU information detection module for NVIDIA H200."""

import subprocess
import shutil
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class GPUInfo:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()

    def _run_smi(self, query: str, fmt: str = "csv,noheader,nounits") -> Optional[str]:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"],
                capture_output=True, text=True, timeout=30,
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def run(self) -> dict:
        if not shutil.which("nvidia-smi"):
            self.console.print("[bold red]nvidia-smi not found![/bold red]")
            return {"error": "nvidia-smi not found", "gpu_count": 0}

        gpu_count_str = self._run_smi("count")
        if not gpu_count_str:
            return {"error": "nvidia-smi query failed", "gpu_count": 0}

        gpu_count = int(gpu_count_str.strip().split("\n")[0])

        names = self._run_smi("name").split("\n") if self._run_smi("name") else []
        uuids = self._run_smi("uuid").split("\n") if self._run_smi("uuid") else []
        pcie_bus = self._run_smi("pci.bus_id").split("\n") if self._run_smi("pci.bus_id") else []
        pcie_gen = self._run_smi("pcie_link.gen").split("\n") if self._run_smi("pcie_link.gen") else []
        pcie_width = self._run_smi("pcie_link.width").split("\n") if self._run_smi("pcie_link.width") else []
        vram_total = self._run_smi("memory.total").split("\n") if self._run_smi("memory.total") else []
        vram_used = self._run_smi("memory.used").split("\n") if self._run_smi("memory.used") else []
        vram_free = self._run_smi("memory.free").split("\n") if self._run_smi("memory.free") else []
        power_draw = self._run_smi("power.draw").split("\n") if self._run_smi("power.draw") else []
        power_limit = self._run_smi("power.limit").split("\n") if self._run_smi("power.limit") else []
        clock_sm = self._run_smi("clocks.sm").split("\n") if self._run_smi("clocks.sm") else []
        clock_mem = self._run_smi("clocks.mem").split("\n") if self._run_smi("clocks.mem") else []
        temperature = self._run_smi("temperature.gpu").split("\n") if self._run_smi("temperature.gpu") else []
        fan_speed = self._run_smi("fan.speed").split("\n") if self._run_smi("fan.speed") else []
        persistence = self._run_smi("persistence_mode").split("\n") if self._run_smi("persistence_mode") else []
        compute_mode = self._run_smi("compute_mode").split("\n") if self._run_smi("compute_mode") else []
        serial = self._run_smi("serial").split("\n") if self._run_smi("serial") else []

        ecc_single = self._run_smi("ecc.errors.single_bit.total.volatile").split("\n") if self._run_smi("ecc.errors.single_bit.total.volatile") else []
        ecc_double = self._run_smi("ecc.errors.double_bit.total.volatile").split("\n") if self._run_smi("ecc.errors.double_bit.total.volatile") else []

        driver_info = self._run_smi("driver_version", "csv,noheader")
        cuda_info = self._run_smi("cuda_version", "csv,noheader")

        def safe_get(lst, idx, default="N/A"):
            try:
                return lst[idx].strip() if idx < len(lst) else default
            except (IndexError, ValueError):
                return default

        def safe_int(val, default=0):
            try:
                return int(val) if val not in ("N/A", "", "[N/A]") else default
            except (ValueError, TypeError):
                return default

        def safe_float(val, default=0.0):
            try:
                return float(val) if val not in ("N/A", "", "[N/A]") else default
            except (ValueError, TypeError):
                return default

        gpus = []
        for i in range(gpu_count):
            gpus.append({
                "index": i,
                "name": safe_get(names, i),
                "uuid": safe_get(uuids, i),
                "pci_bus_id": safe_get(pcie_bus, i),
                "pcie_link_gen": safe_int(safe_get(pcie_gen, i)),
                "pcie_link_width": safe_int(safe_get(pcie_width, i)),
                "vram_total_mb": safe_int(safe_get(vram_total, i)),
                "vram_used_mb": safe_int(safe_get(vram_used, i)),
                "vram_free_mb": safe_int(safe_get(vram_free, i)),
                "power_draw": safe_float(safe_get(power_draw, i)),
                "power_limit": safe_float(safe_get(power_limit, i)),
                "clock_sm": safe_int(safe_get(clock_sm, i)),
                "clock_mem": safe_int(safe_get(clock_mem, i)),
                "temperature": safe_int(safe_get(temperature, i)),
                "fan_speed": safe_int(safe_get(fan_speed, i)),
                "persistence_mode": safe_get(persistence, i) == "Enabled",
                "compute_mode": safe_get(compute_mode, i),
                "serial_number": safe_get(serial, i),
                "ecc_errors_single": safe_int(safe_get(ecc_single, i)),
                "ecc_errors_double": safe_int(safe_get(ecc_double, i)),
            })

        topology = self._get_topology()

        return {
            "driver_version": safe_get(driver_info.split("\n"), 0) if driver_info else "N/A",
            "cuda_version": safe_get(cuda_info.split("\n"), 0) if cuda_info else "N/A",
            "gpu_count": gpu_count,
            "gpus": gpus,
            "topology": topology,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_topology(self) -> str:
        try:
            r = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=15,
            )
            return r.stdout if r.returncode == 0 else "Unavailable"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "Unavailable"

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()
        if "error" in results:
            c.print(f"[bold red]Error: {results['error']}[/bold red]")
            return

        c.print(f"\n[bold cyan]GPU Information[/bold cyan]")
        c.print(f"  Driver Version : {results.get('driver_version', 'N/A')}")
        c.print(f"  CUDA Version   : {results.get('cuda_version', 'N/A')}")
        c.print(f"  GPU Count      : {results.get('gpu_count', 0)}")
        c.print(f"  Timestamp      : {results.get('timestamp', 'N/A')}")

        gpus = results.get("gpus", [])
        if not gpus:
            return

        table = Table(title="GPU Details", box=None, padding=(0, 1), show_lines=False)
        table.add_column("GPU", style="bold cyan", width=5)
        table.add_column("Model", width=18)
        table.add_column("VRAM", justify="right", width=14)
        table.add_column("Temp", justify="right", width=6)
        table.add_column("Power", justify="right", width=12)
        table.add_column("SM Clk", justify="right", width=8)
        table.add_column("Mem Clk", justify="right", width=8)
        table.add_column("PCIe", width=10)
        table.add_column("Persist", width=8)

        for g in gpus:
            name = g["name"]
            if "H200" in name:
                name = f"[bold green]{name}[/bold green]"
            vram = f"{g['vram_used_mb']}/{g['vram_total_mb']} MB"
            temp = f"{g['temperature']}°C"
            temp_color = "red" if g["temperature"] > 85 else ("yellow" if g["temperature"] > 75 else "green")
            temp = f"[{temp_color}]{temp}[/{temp_color}]"
            power = f"{g['power_draw']:.0f}/{g['power_limit']:.0f}W"
            sm_clk = f"{g['clock_sm']} MHz"
            mem_clk = f"{g['clock_mem']} MHz"
            pcie = f"Gen{g['pcie_link_gen']} x{g['pcie_link_width']}"
            persist = "[green]ON[/green]" if g["persistence_mode"] else "[red]OFF[/red]"
            ecc_warn = ""
            if g["ecc_errors_double"] > 0:
                ecc_warn = " [bold red]ECC![/bold red]"

            table.add_row(
                str(g["index"]), name, vram, temp, power,
                sm_clk, mem_clk, pcie, persist + ecc_warn,
            )
        c.print(table)

        topology = results.get("topology", "")
        if topology and topology != "Unavailable":
            c.print(Panel(topology, title="[bold]NVLink/NVSwitch Topology[/bold]", border_style="cyan"))

        warnings = []
        for g in gpus:
            if not g["persistence_mode"]:
                warnings.append(f"GPU {g['index']}: Persistence mode DISABLED")
            if g["temperature"] > 85:
                warnings.append(f"GPU {g['index']}: High temperature {g['temperature']}°C")
            if g["ecc_errors_double"] > 0:
                warnings.append(f"GPU {g['index']}: {g['ecc_errors_double']} double-bit ECC errors!")

        if warnings:
            c.print("\n[bold yellow]Warnings:[/bold yellow]")
            for w in warnings:
                c.print(f"  [yellow]⚠ {w}[/yellow]")
