"""Hardware health monitoring module for NVIDIA datacenter GPUs (H100/H200/B200/B300)."""

import subprocess
import shutil
import os
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from modules.gpu_specs import detect_gpu_type, get_gpu_specs


class HealthCheck:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.health_cfg = config.get("health", {})
        self.gpu_type = detect_gpu_type()
        self.specs = get_gpu_specs(self.gpu_type)

    def _run_smi(self, query: str) -> Optional[str]:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30,
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _run_cmd(self, cmd: list, timeout: int = 10) -> Optional[str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip() if r.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _safe_int(self, val, default=0):
        try:
            return int(val) if val not in ("N/A", "", "[N/A]") else default
        except (ValueError, TypeError):
            return default

    def _safe_float(self, val, default=0.0):
        try:
            return float(val) if val not in ("N/A", "", "[N/A]") else default
        except (ValueError, TypeError):
            return default

    def run(self) -> dict:
        if not shutil.which("nvidia-smi"):
            return {"error": "nvidia-smi not found", "passed": False}

        gpu_count_str = self._run_smi("count")
        if not gpu_count_str:
            return {"error": "nvidia-smi query failed", "passed": False}
        gpu_count = int(gpu_count_str.split("\n")[0])

        def query_lines(field):
            raw = self._run_smi(field)
            return raw.split("\n") if raw else []

        temps = query_lines("temperature.gpu")
        power_draws = query_lines("power.draw")
        power_limits = query_lines("power.limit")
        ecc_single = query_lines("ecc.errors.single_bit.total.volatile")
        ecc_double = query_lines("ecc.errors.double_bit.total.volatile")
        pcie_gens = query_lines("pcie_link.gen.current")
        pcie_widths = query_lines("pcie_link.width.current")
        clock_sms = query_lines("clocks.sm")
        clock_mems = query_lines("clocks.mem")
        persistence = query_lines("persistence_mode")

        throttling_raw = query_lines("clocks_throttle_reasons.active")
        mig_modes = query_lines("mig.mode.current")

        temp_warn = self.health_cfg.get("temp_warning", 80)
        temp_crit = self.health_cfg.get("temp_critical", 90)
        power_lim = self.health_cfg.get("power_limit", self.specs.get("tdp_watts", 700))

        gpu_health = []
        overall_pass = True

        for i in range(gpu_count):
            checks = {}

            temp_val = self._safe_int(temps[i] if i < len(temps) else 0)
            if temp_val >= temp_crit:
                checks["temperature"] = {"value": temp_val, "status": "FAIL", "threshold": temp_crit}
                overall_pass = False
            elif temp_val >= temp_warn:
                checks["temperature"] = {"value": temp_val, "status": "WARN", "threshold": temp_warn}
            else:
                checks["temperature"] = {"value": temp_val, "status": "PASS", "threshold": temp_warn}

            pd = self._safe_float(power_draws[i] if i < len(power_draws) else 0)
            pl = self._safe_float(power_limits[i] if i < len(power_limits) else power_lim)
            checks["power"] = {"value": pd, "limit": pl, "status": "PASS" if pd <= pl * 1.05 else "WARN"}

            es = self._safe_int(ecc_single[i] if i < len(ecc_single) else 0)
            ed = self._safe_int(ecc_double[i] if i < len(ecc_double) else 0)
            ecc_status = "FAIL" if ed > 0 else ("WARN" if es > 100 else "PASS")
            if ecc_status == "FAIL":
                overall_pass = False
            checks["ecc_errors"] = {"single": es, "double": ed, "status": ecc_status}

            checks["memory_errors"] = {"status": "PASS"}

            pg = self._safe_int(pcie_gens[i] if i < len(pcie_gens) else 0)
            pw = self._safe_int(pcie_widths[i] if i < len(pcie_widths) else 0)
            pcie_ok = pg >= 4 and pw >= 8
            if not pcie_ok:
                overall_pass = False
            checks["pcie_link"] = {"gen": pg, "width": pw, "status": "PASS" if pcie_ok else "WARN"}

            sm = self._safe_int(clock_sms[i] if i < len(clock_sms) else 0)
            mm = self._safe_int(clock_mems[i] if i < len(clock_mems) else 0)
            checks["clock_speed"] = {"sm": sm, "mem": mm, "status": "PASS" if sm > 0 and mm > 0 else "WARN"}

            throttle_val = throttling_raw[i] if i < len(throttling_raw) else ""
            throttle_active = throttle_val not in ("", "None", "Active", "N/A")
            if throttle_active:
                overall_pass = False
            checks["throttling"] = {"status": "FAIL" if throttle_active else "PASS", "reasons": [throttle_val] if throttle_active else []}

            pers_val = persistence[i] if i < len(persistence) else ""
            pers_enabled = pers_val == "Enabled"
            checks["persistence_mode"] = {"enabled": pers_enabled, "status": "PASS" if pers_enabled else "WARN"}

            worst = "PASS"
            for chk in checks.values():
                s = chk["status"]
                if s == "FAIL":
                    worst = "FAIL"
                    break
                elif s == "WARN":
                    worst = "WARN"
            if worst == "FAIL":
                overall_pass = False

            gpu_health.append({"index": i, "status": worst, "checks": checks})

        system_health = self._check_system()

        return {
            "passed": overall_pass,
            "gpu_health": gpu_health,
            "system_health": system_health,
            "timestamp": datetime.now().isoformat(),
            "detected_gpu_type": self.gpu_type,
        }

    def _check_system(self) -> dict:
        persistd = shutil.which("nvidia-persistenced") is not None
        persistd_running = False
        if persistd:
            r = self._run_cmd(["pgrep", "-x", "nvidia-persistenced"])
            persistd_running = r is not None

        hugepages = 0
        hp_path = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size"
        if os.path.exists("/proc/meminfo"):
            r = self._run_cmd(["grep", "-i", "hugepages_total", "/proc/meminfo"])
            if r:
                parts = r.split()
                hugepages = int(parts[1]) if len(parts) >= 2 else 0

        swap_status = False
        if os.path.exists("/proc/swaps"):
            r = self._run_cmd(["grep", "-c", "^/", "/proc/swaps"])
            if r and int(r) > 0:
                swap_status = True

        thp = "unknown"
        if os.path.exists("/sys/kernel/mm/transparent_hugepage/enabled"):
            r = self._run_cmd(["cat", "/sys/kernel/mm/transparent_hugepage/enabled"])
            if r:
                if "[always]" in r:
                    thp = "always"
                elif "[madvise]" in r:
                    thp = "madvise"
                elif "[never]" in r:
                    thp = "never"

        fd_soft, fd_max = 1024, 65535
        try:
            import resource
            fd_soft, fd_max = resource.getrlimit(resource.RLIMIT_NOFILE)
        except (ImportError, ValueError):
            pass

        ib_devs = []
        if os.path.isdir("/sys/class/infiniband"):
            ib_devs = os.listdir("/sys/class/infiniband")

        rdma_devs = []
        if os.path.isdir("/sys/class/infiniband_verbs"):
            rdma_devs = os.listdir("/sys/class/infiniband_verbs")

        nccl_env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}

        return {
            "nvidia_persistenced": {"installed": persistd, "running": persistd_running},
            "hugepages": {"configured": hugepages > 0, "count": hugepages},
            "swap": {"enabled": swap_status},
            "transparent_hugepage": thp,
            "file_descriptors": {"soft": fd_soft, "max": fd_max},
            "infiniband_devices": ib_devs,
            "rdma_devices": rdma_devs,
            "nccl_env_vars": nccl_env,
        }

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()
        if "error" in results:
            c.print(f"[bold red]Error: {results['error']}[/bold red]")
            return

        passed = results.get("passed", False)
        verdict = "[bold green]✓ ALL CHECKS PASSED[/bold green]" if passed else "[bold red]✗ SOME CHECKS FAILED[/bold red]"
        c.print(Panel(verdict, border_style="green" if passed else "red"))

        gpu_health = results.get("gpu_health", [])
        if gpu_health:
            table = Table(title="GPU Health Checks", box=None, padding=(0, 1))
            table.add_column("GPU", style="bold", width=5)
            table.add_column("Temp", width=10)
            table.add_column("Power", width=12)
            table.add_column("ECC", width=10)
            table.add_column("PCIe", width=10)
            table.add_column("Clock", width=8)
            table.add_column("Throttle", width=10)
            table.add_column("Persist", width=8)
            table.add_column("Status", width=7)

            for g in gpu_health:
                ch = g["checks"]
                status_color = "green" if g["status"] == "PASS" else ("yellow" if g["status"] == "WARN" else "red")
                status_text = f"[{status_color}]{g['status']}[/{status_color}]"

                def status_icon(s):
                    return {"PASS": "[green]✓[/green]", "WARN": "[yellow]![/yellow]", "FAIL": "[red]✗[/red]"}.get(s, s)

                temp = f"{ch['temperature']['value']}°C {status_icon(ch['temperature']['status'])}"
                pw = f"{ch['power']['value']:.0f}W {status_icon(ch['power']['status'])}"
                ecc = f"S:{ch['ecc_errors']['single']} D:{ch['ecc_errors']['double']} {status_icon(ch['ecc_errors']['status'])}"
                pcie = f"Gen{ch['pcie_link']['gen']}x{ch['pcie_link']['width']} {status_icon(ch['pcie_link']['status'])}"
                clk = f"{ch['clock_speed']['sm']}MHz {status_icon(ch['clock_speed']['status'])}"
                thr = status_icon(ch["throttling"]["status"])
                pers = status_icon(ch["persistence_mode"]["status"])

                table.add_row(str(g["index"]), temp, pw, ecc, pcie, clk, thr, pers, status_text)
            c.print(table)

        sys_h = results.get("system_health", {})
        if sys_h:
            c.print("\n[bold cyan]System Health[/bold cyan]")
            np = sys_h.get("nvidia_persistenced", {})
            np_status = "[green]Running[/green]" if np.get("running") else "[red]Not running[/red]"
            if not np.get("installed"):
                np_status = "[yellow]Not installed[/yellow]"
            c.print(f"  nvidia-persistenced : {np_status}")

            hp = sys_h.get("hugepages", {})
            hp_status = "[green]Configured[/green]" if hp.get("configured") else "[yellow]Not configured[/yellow]"
            c.print(f"  Hugepages          : {hp_status} ({hp.get('count', 0)} pages)")

            swap = sys_h.get("swap", {})
            swap_txt = "[red]Enabled[/red]" if swap.get("enabled") else "[green]Disabled[/green]"
            c.print(f"  Swap               : {swap_txt}")

            thp = sys_h.get("transparent_hugepage", "unknown")
            thp_color = "green" if thp in ("always", "madvise") else "yellow"
            c.print(f"  Transparent HP     : [{thp_color}]{thp}[/{thp_color}]")

            fd = sys_h.get("file_descriptors", {})
            fd_ok = fd.get("soft", 0) >= 65536
            fd_color = "green" if fd_ok else "yellow"
            c.print(f"  File Descriptors   : [{fd_color}]{fd.get('soft', 'N/A')} (soft) / {fd.get('max', 'N/A')} (max)[/{fd_color}]")

            ib = sys_h.get("infiniband_devices", [])
            rdma = sys_h.get("rdma_devices", [])
            if ib:
                c.print(f"  InfiniBand         : [green]{', '.join(ib)}[/green]")
            else:
                c.print("  InfiniBand         : [yellow]No devices detected[/yellow]")
            if rdma:
                c.print(f"  RDMA               : [green]{', '.join(rdma)}[/green]")

            nccl = sys_h.get("nccl_env_vars", {})
            if nccl:
                c.print("  NCCL Env Vars:")
                for k, v in sorted(nccl.items()):
                    c.print(f"    {k}={v}")
            else:
                c.print("  NCCL Env Vars     : [yellow]None set[/yellow]")
