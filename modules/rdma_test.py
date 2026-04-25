"""RDMA / InfiniBand bandwidth and latency test module."""

import os
import shutil
import subprocess
from datetime import datetime
from typing import Optional, List

from rich.console import Console
from rich.table import Table


class RDMATest:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.rdma_cfg = config.get("rdma", {})

    def _find_tool(self, name: str) -> Optional[str]:
        p = shutil.which(name)
        if p:
            return p
        return None

    def _get_ib_devices(self) -> List[str]:
        devices = []
        ib_path = "/sys/class/infiniband"
        if os.path.isdir(ib_path):
            devices = sorted(os.listdir(ib_path))
        return devices

    def _get_ib_ports(self, device: str) -> List[str]:
        ports = []
        ports_dir = f"/sys/class/infiniband/{device}/ports"
        if os.path.isdir(ports_dir):
            ports = sorted(os.listdir(ports_dir))
        return ports

    def run(self) -> dict:
        devices = self._get_ib_devices()
        if not devices:
            self.console.print("[yellow]No InfiniBand devices found[/yellow]")
            return {"error": "no_ib_devices", "passed": False}

        self.console.print(f"[cyan]RDMA Test - Devices: {', '.join(devices)}[/cyan]")

        device_info = self._collect_device_info(devices)
        bw_results = self._run_bandwidth_tests(devices)
        latency_results = self._run_latency_tests(devices)

        all_passed = all(
            r.get("status") == "PASS"
            for r in bw_results + latency_results
            if isinstance(r, dict)
        )

        return {
            "passed": all_passed,
            "devices": device_info,
            "bandwidth_tests": bw_results,
            "latency_tests": latency_results,
            "timestamp": datetime.now().isoformat(),
        }

    def _collect_device_info(self, devices: List[str]) -> List[dict]:
        info = []
        for dev in devices:
            dev_info = {"name": dev, "ports": []}
            ports = self._get_ib_ports(dev)
            for port in ports:
                port_info = {"port": port}
                rate_path = f"/sys/class/infiniband/{dev}/ports/{port}/rate"
                state_path = f"/sys/class/infiniband/{dev}/ports/{port}/state"
                phys_state_path = f"/sys/class/infiniband/{dev}/ports/{port}/phys_state"
                gid_path = f"/sys/class/infiniband/{dev}/ports/{port}/gids/0"

                for label, path in [("rate", rate_path), ("state", state_path),
                                     ("phys_state", phys_state_path), ("gid", gid_path)]:
                    try:
                        with open(path) as f:
                            port_info[label] = f.read().strip()
                    except (FileNotFoundError, PermissionError):
                        port_info[label] = "N/A"

                dev_info["ports"].append(port_info)
            info.append(dev_info)
        return info

    def _run_ib_command(self, cmd: List[str], timeout: int = 60) -> dict:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return {"status": "PASS", "output": r.stdout.strip()}
            return {"status": "FAIL", "error": r.stderr.strip()[:200]}
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "error": "timeout"}
        except FileNotFoundError:
            return {"status": "SKIP", "error": "tool not found"}
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    def _run_bandwidth_tests(self, devices: List[str]) -> List[dict]:
        results = []
        ib_write_bw = self._find_tool("ib_write_bw")
        ib_read_bw = self._find_tool("ib_read_bw")
        min_bw = self.rdma_cfg.get("min_bandwidth_gbps", 50)
        msg_size = self.rdma_cfg.get("msg_size", 65536)
        iters = self.rdma_cfg.get("ib_iterations", 1000)
        dx = self.rdma_cfg.get("ib_device", None)
        port = self.rdma_cfg.get("ib_port", 1)

        for tool, label in [(ib_write_bw, "ib_write_bw"), (ib_read_bw, "ib_read_bw")]:
            if not tool:
                results.append({"test": label, "status": "SKIP", "error": "not installed"})
                continue

            server_cmd = [tool, "-d", dx or devices[0], "-i", str(port), "-s", str(msg_size)]
            client_cmd = server_cmd + ["localhost"]

            server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            import time
            time.sleep(1)

            try:
                client = subprocess.run(client_cmd, capture_output=True, text=True, timeout=60)
                server.wait(timeout=10)

                output = client.stdout + server.stdout.read() if server.stdout else ""
                bw_mbps = 0
                for line in output.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        bw_mbps = max(bw_mbps, float(parts[-1]))
                    except (ValueError, IndexError):
                        continue

                bw_gbps = bw_mbps / 1000 if bw_mbps else 0
                status = "PASS" if bw_gbps >= min_bw else "WARN"
                results.append({
                    "test": label,
                    "status": status,
                    "bandwidth_gbps": round(bw_gbps, 2),
                    "min_required_gbps": min_bw,
                })
            except Exception as e:
                server.kill()
                results.append({"test": label, "status": "FAIL", "error": str(e)})

        return results

    def _run_latency_tests(self, devices: List[str]) -> List[dict]:
        results = []
        ib_write_lat = self._find_tool("ib_write_lat")
        ib_read_lat = self._find_tool("ib_read_lat")
        max_lat_us = self.rdma_cfg.get("max_latency_us", 10)
        dx = self.rdma_cfg.get("ib_device", None)
        port = self.rdma_cfg.get("ib_port", 1)

        for tool, label in [(ib_write_lat, "ib_write_lat"), (ib_read_lat, "ib_read_lat")]:
            if not tool:
                results.append({"test": label, "status": "SKIP", "error": "not installed"})
                continue

            server_cmd = [tool, "-d", dx or devices[0], "-i", str(port)]
            client_cmd = server_cmd + ["localhost"]

            server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            import time
            time.sleep(1)

            try:
                client = subprocess.run(client_cmd, capture_output=True, text=True, timeout=60)
                server.wait(timeout=10)

                output = client.stdout + server.stdout.read() if server.stdout else ""
                lat_us = 0
                for line in output.split("\n"):
                    parts = line.strip().split()
                    try:
                        lat_us = max(lat_us, float(parts[-1]))
                    except (ValueError, IndexError):
                        continue

                status = "PASS" if 0 < lat_us <= max_lat_us else ("WARN" if lat_us > 0 else "FAIL")
                results.append({
                    "test": label,
                    "status": status,
                    "latency_us": round(lat_us, 2),
                    "max_allowed_us": max_lat_us,
                })
            except Exception as e:
                server.kill()
                results.append({"test": label, "status": "FAIL", "error": str(e)})

        return results

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()
        if "error" in results:
            c.print(f"[bold red]Error: {results['error']}[/bold red]")
            return

        devices = results.get("devices", [])
        c.print(f"\n[bold cyan]RDMA/InfiniBand Test Results[/bold cyan]")
        c.print(f"  Devices found: {len(devices)}")

        for dev in devices:
            c.print(f"\n  [bold]{dev['name']}[/bold]")
            for p in dev.get("ports", []):
                state = p.get("state", "N/A")
                color = "green" if "Active" in state else "red"
                c.print(f"    Port {p['port']}: [{color}]{state}[/{color}] | "
                        f"Rate: {p.get('rate', 'N/A')} | GID: {p.get('gid', 'N/A')[:20]}")

        bw_tests = results.get("bandwidth_tests", [])
        if bw_tests:
            c.print("\n  [bold]Bandwidth Tests[/bold]")
            for t in bw_tests:
                status = t.get("status", "SKIP")
                sc = "green" if status == "PASS" else ("yellow" if status == "WARN" else "red")
                bw = t.get("bandwidth_gbps", 0)
                c.print(f"    {t['test']}: [{sc}]{status}[/{sc}] "
                        f"({bw:.2f} GB/s, min: {t.get('min_required_gbps', 'N/A')} GB/s)" if status != "SKIP"
                        else f"    {t['test']}: [dim]SKIPPED[/dim]")

        lat_tests = results.get("latency_tests", [])
        if lat_tests:
            c.print("\n  [bold]Latency Tests[/bold]")
            for t in lat_tests:
                status = t.get("status", "SKIP")
                sc = "green" if status == "PASS" else ("yellow" if status == "WARN" else "red")
                lat = t.get("latency_us", 0)
                c.print(f"    {t['test']}: [{sc}]{status}[/{sc}] "
                        f"({lat:.2f} us, max: {t.get('max_allowed_us', 'N/A')} us)" if status != "SKIP"
                        else f"    {t['test']}: [dim]SKIPPED[/dim]")
