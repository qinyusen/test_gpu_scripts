"""Report generation module - export test results to JSON/HTML/Markdown."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Test Report - {timestamp}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
               background: #0d1117; color: #c9d1d9; padding: 2rem; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e);
                   padding: 2rem; border-radius: 8px; margin-bottom: 2rem;
                   border: 1px solid #30363d; }}
        .header h1 {{ color: #58a6ff; font-size: 1.5rem; }}
        .header .meta {{ color: #8b949e; margin-top: 0.5rem; }}
        .section {{ background: #161b22; border: 1px solid #30363d;
                    border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        .section h2 {{ color: #58a6ff; margin-bottom: 1rem; font-size: 1.2rem;
                       border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; }}
        th {{ background: #21262d; color: #8b949e; text-align: left;
             padding: 0.5rem; font-weight: 600; font-size: 0.85rem; }}
        td {{ padding: 0.5rem; border-bottom: 1px solid #21262d; font-size: 0.9rem; }}
        .pass {{ color: #3fb950; }} .warn {{ color: #d29922; }} .fail {{ color: #f85149; }}
        .metric {{ display: inline-block; background: #21262d; padding: 0.75rem 1.5rem;
                  border-radius: 6px; margin: 0.25rem; text-align: center; min-width: 120px; }}
        .metric .value {{ font-size: 1.3rem; font-weight: bold; color: #58a6ff; }}
        .metric .label {{ font-size: 0.75rem; color: #8b949e; margin-top: 0.25rem; }}
        .verdict {{ padding: 1rem; border-radius: 6px; text-align: center; font-size: 1.1rem;
                   font-weight: bold; margin: 1rem 0; }}
        .verdict.pass {{ background: #0d2818; color: #3fb950; border: 1px solid #238636; }}
        .verdict.fail {{ background: #2d0b0b; color: #f85149; border: 1px solid #da3633; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GPU Training Server Test Report</h1>
        <div class="meta">Generated: {timestamp} | Server: {hostname}</div>
    </div>
    {content}
</body>
</html>"""


class ReportGenerator:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.report_cfg = config.get("report", {})

    def generate(self, results: dict, fmt: str = None, output: str = None) -> str:
        fmt = fmt or self.report_cfg.get("format", "json")
        output_dir = self.report_cfg.get("output_dir", "./reports")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output:
            output = os.path.join(output_dir, f"gpu_report_{timestamp}.{fmt}")

        if fmt == "json":
            return self._generate_json(results, output)
        elif fmt == "html":
            return self._generate_html(results, output)
        elif fmt == "md":
            return self._generate_markdown(results, output)
        else:
            self.console.print(f"[red]Unsupported format: {fmt}[/red]")
            return ""

    def _generate_json(self, results: dict, output: str) -> str:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.console.print(f"[green]JSON report saved to: {output}[/green]")
        return output

    def _generate_html(self, results: dict, output: str) -> str:
        import socket
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sections = []

        if "gpu_info" in results:
            gpus = results["gpu_info"].get("gpus", [])
            rows = ""
            for g in gpus:
                rows += f"<tr><td>GPU {g['index']}</td><td>{g['name']}</td>"
                rows += f"<td>{g['vram_total_mb']} MB</td>"
                rows += f"<td>{g['temperature']}°C</td>"
                rows += f"<td>{g['clock_sm']} MHz</td></tr>"
            sections.append(
                f'<div class="section"><h2>GPU Information</h2>'
                f'<p>Driver: {results["gpu_info"].get("driver_version", "N/A")} | '
                f'CUDA: {results["gpu_info"].get("cuda_version", "N/A")} | '
                f'Count: {len(gpus)}</p>'
                f'<table><tr><th>GPU</th><th>Model</th><th>VRAM</th><th>Temp</th><th>SM Clock</th></tr>'
                f'{rows}</table></div>'
            )

        if "health" in results:
            h = results["health"]
            passed = h.get("passed", False)
            cls = "pass" if passed else "fail"
            txt = "ALL PASSED" if passed else "SOME CHECKS FAILED"
            sections.append(f'<div class="verdict {cls}">{txt}</div>')

        if "benchmark" in results and "memory" in results["benchmark"]:
            mem = results["benchmark"]["memory"]
            sections.append(
                f'<div class="section"><h2>Memory Bandwidth</h2>'
                f'<div class="metric"><div class="value">{mem.get("d2d_bandwidth_gbps", "N/A")} GB/s</div>'
                f'<div class="label">D2D (HBM)</div></div>'
                f'<div class="metric"><div class="value">{mem.get("efficiency_pct", "N/A")}%</div>'
                f'<div class="label">Efficiency vs Peak ({mem.get("peak_bandwidth_gbps", "N/A")} GB/s)</div></div>'
                f'</div>'
            )

        if "benchmark" in results and "compute" in results["benchmark"]:
            comp = results["benchmark"]["compute"]
            dtype_rows = ""
            per_dtype = comp.get("per_dtype_tflops", {})
            eff = comp.get("efficiency_pct", {})
            for dt, tflops in per_dtype.items():
                ef = eff.get(dt, 0)
                cls = "pass" if ef >= 80 else ("warn" if ef >= 50 else "fail")
                if isinstance(tflops, (int, float)):
                    dtype_rows += f'<tr><td>{dt.upper()}</td><td>{tflops:.1f} TFLOPS</td>'
                    dtype_rows += f'<td class="{cls}">{ef:.1f}%</td></tr>'
            if dtype_rows:
                sections.append(
                    f'<div class="section"><h2>Compute Throughput</h2>'
                    f'<table><tr><th>DType</th><th>Achieved</th><th>Efficiency</th></tr>'
                    f'{dtype_rows}</table></div>'
                )

        if "training" in results:
            t = results["training"]
            sections.append(
                f'<div class="section"><h2>Training Simulation</h2>'
                f'<div class="metric"><div class="value">{t.get("throughput_tokens_per_sec", "N/A")}</div>'
                f'<div class="label">Tokens/sec</div></div>'
                f'<div class="metric"><div class="value">{t.get("avg_step_time_ms", "N/A")} ms</div>'
                f'<div class="label">Avg Step Time</div></div>'
                f'<div class="metric"><div class="value">{t.get("peak_memory_gb", "N/A")} GB</div>'
                f'<div class="label">Peak Memory</div></div>'
                f'</div>'
            )

        content = "\n".join(sections)
        html = HTML_TEMPLATE.format(timestamp=timestamp, hostname=hostname, content=content)

        with open(output, "w") as f:
            f.write(html)
        self.console.print(f"[green]HTML report saved to: {output}[/green]")
        return output

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def _generate_markdown(self, results: dict, output: str) -> str:
        import socket
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = []

        # --- Header ---
        lines.append("# GPU Test Report\n")
        lines.append(f"- **Date:** {timestamp}")
        lines.append(f"- **Host:** {hostname}")

        # Extract GPU info for header
        gpu_info = results.get("gpu_info")
        if gpu_info and not gpu_info.get("error"):
            gpus = gpu_info.get("gpus", [])
            gpu_name = gpus[0]["name"] if gpus else "Unknown"
            lines.append(f"- **GPU:** {gpu_name} x{gpu_info.get('gpu_count', len(gpus))}")
            lines.append(f"- **Driver:** {gpu_info.get('driver_version', 'N/A')} | "
                         f"**CUDA:** {gpu_info.get('cuda_version', 'N/A')}")
        lines.append("")

        # --- Summary table ---
        summary_items = self._build_summary(results)
        if summary_items:
            lines.append("## Summary\n")
            lines.append("| Test | Result |")
            lines.append("|------|--------|")
            for name, verdict in summary_items:
                lines.append(f"| {name} | {verdict} |")
            lines.append("")

        # --- GPU Information ---
        if gpu_info and not gpu_info.get("error"):
            lines.append("## GPU Information\n")
            gpus = gpu_info.get("gpus", [])
            lines.append("| GPU | Model | VRAM | Temp | Power | SM Clock |")
            lines.append("|-----|-------|------|------|-------|----------|")
            for g in gpus:
                vram = f"{g.get('vram_total_mb', 0)} MB"
                temp = f"{g.get('temperature', 'N/A')}C"
                power = f"{g.get('power_draw', 0):.0f}/{g.get('power_limit', 0):.0f}W"
                clock = f"{g.get('clock_sm', 0)} MHz"
                lines.append(f"| {g['index']} | {g['name']} | {vram} | {temp} | {power} | {clock} |")
            lines.append("")

        # --- Health Check ---
        health = results.get("health")
        if health and not health.get("error"):
            lines.append("## Health Check\n")
            passed = health.get("passed", False)
            lines.append(f"**Overall: {'PASS' if passed else 'FAIL'}**\n")
            gpu_health = health.get("gpu_health", [])
            if gpu_health:
                lines.append("| GPU | Temp | Power | ECC | PCIe | Throttle | Status |")
                lines.append("|-----|------|-------|-----|------|----------|--------|")
                for gh in gpu_health:
                    checks = gh.get("checks", {})
                    temp_c = checks.get("temperature", {})
                    pwr = checks.get("power", {})
                    ecc = checks.get("ecc_errors", {})
                    pcie = checks.get("pcie_link", {})
                    throttle = checks.get("throttling", {})
                    temp_str = f"{temp_c.get('value', '?')}C {temp_c.get('status', '')}"
                    pwr_str = f"{pwr.get('value', 0):.0f}W {pwr.get('status', '')}"
                    ecc_str = f"S:{ecc.get('single', 0)} D:{ecc.get('double', 0)}"
                    pcie_str = f"Gen{pcie.get('gen', '?')}x{pcie.get('width', '?')}"
                    throt_str = throttle.get("status", "?")
                    status = gh.get("status", "?")
                    lines.append(f"| {gh['index']} | {temp_str} | {pwr_str} | "
                                 f"{ecc_str} | {pcie_str} | {throt_str} | **{status}** |")
            lines.append("")

        # --- Memory Bandwidth ---
        mem_data = self._extract_memory_results(results)
        if mem_data and not mem_data.get("error"):
            lines.append("## Memory Bandwidth\n")
            lines.append(f"Source: {mem_data.get('source', 'unknown')}\n")
            lines.append("| Metric | Value | Peak | Efficiency |")
            lines.append("|--------|-------|------|------------|")
            d2d = mem_data.get("d2d_bandwidth_gbps", 0)
            h2d = mem_data.get("h2d_bandwidth_gbps", 0)
            d2h = mem_data.get("d2h_bandwidth_gbps", 0)
            peak = mem_data.get("peak_bandwidth_gbps", 0)
            eff = mem_data.get("efficiency_pct", 0)
            lines.append(f"| D2D (HBM) | {d2d:.1f} GB/s | {peak:.0f} GB/s | {eff:.1f}% |")
            lines.append(f"| H2D | {h2d:.1f} GB/s | - | - |")
            lines.append(f"| D2H | {d2h:.1f} GB/s | - | - |")
            lines.append("")
            verdict = "PASS" if eff >= 80 else ("WARN" if eff >= 50 else "FAIL")
            lines.append(f"**Verdict: {verdict}** (D2D efficiency {eff:.1f}%)\n")

        # --- Compute Throughput ---
        comp_data = self._extract_compute_results(results)
        if comp_data and not comp_data.get("error"):
            lines.append("## Compute Throughput\n")
            per_dtype = comp_data.get("per_dtype_tflops", {})
            peak_tflops = comp_data.get("peak_tflops", {})
            eff_pct = comp_data.get("efficiency_pct", {})
            lines.append("| DType | Achieved (TFLOPS) | Peak | Efficiency | Status |")
            lines.append("|-------|-------------------|------|------------|--------|")
            worst_eff = 100.0
            for dt, val in per_dtype.items():
                if isinstance(val, str):
                    # skipped or error
                    lines.append(f"| {dt.upper()} | {val} | - | N/A | SKIP |")
                else:
                    pk = peak_tflops.get(dt, 0)
                    ef = eff_pct.get(dt, 0)
                    if isinstance(ef, (int, float)) and ef > 0:
                        worst_eff = min(worst_eff, ef)
                    status = "PASS" if ef >= 80 else ("WARN" if ef >= 50 else "FAIL")
                    lines.append(f"| {dt.upper()} | {val:.1f} | {pk:.0f} | {ef:.1f}% | {status} |")
            lines.append("")
            overall = "PASS" if worst_eff >= 80 else ("WARN" if worst_eff >= 50 else "FAIL")
            lines.append(f"**Verdict: {overall}** (worst efficiency {worst_eff:.1f}%)\n")

        # --- NCCL ---
        nccl = results.get("nccl")
        if nccl and not nccl.get("error"):
            lines.append("## NCCL Multi-GPU\n")
            lines.append(f"Source: {nccl.get('source', 'unknown')} | "
                         f"GPUs: {nccl.get('gpu_count', '?')}\n")
            tests = nccl.get("tests", {})
            if tests:
                lines.append("| Operation | Bus BW (GB/s) | Threshold | Status |")
                lines.append("|-----------|---------------|-----------|--------|")
                for op, data in tests.items():
                    if isinstance(data, dict) and not data.get("error"):
                        bw = data.get("best_busbw_gbps", 0)
                        req = data.get("min_required_gbps", 0)
                        status = data.get("status", "?")
                        lines.append(f"| {op} | {bw:.1f} | >= {req:.0f} | {status} |")
                    elif isinstance(data, dict) and data.get("error"):
                        lines.append(f"| {op} | - | - | ERROR: {data['error']} |")
                lines.append("")
            passed = nccl.get("passed", False)
            lines.append(f"**Overall: {'PASS' if passed else 'FAIL'}**\n")

        # --- Stress Test ---
        stress = results.get("stress")
        if stress and not stress.get("error"):
            lines.append("## Stress Test\n")
            passed = stress.get("passed", False)
            duration = stress.get("duration_sec", 0)
            elapsed = stress.get("elapsed_sec", 0)
            source = stress.get("source", "unknown")
            lines.append(f"- **Source:** {source}")
            lines.append(f"- **Duration:** {elapsed:.0f}s (requested {duration}s)")
            lines.append(f"- **Result: {'PASS' if passed else 'FAIL'}**")
            lines.append("")

        # --- RDMA ---
        rdma = results.get("rdma")
        if rdma and not rdma.get("error"):
            lines.append("## RDMA/InfiniBand\n")
            bw_tests = rdma.get("bandwidth_tests", [])
            lat_tests = rdma.get("latency_tests", [])
            if bw_tests or lat_tests:
                lines.append("| Test | Value | Threshold | Status |")
                lines.append("|------|-------|-----------|--------|")
                for bt in bw_tests:
                    if not bt.get("error"):
                        lines.append(f"| {bt['test']} | {bt.get('bandwidth_gbps', 0):.1f} GB/s | "
                                     f">= {bt.get('min_required_gbps', 0)} GB/s | {bt.get('status', '?')} |")
                for lt in lat_tests:
                    if not lt.get("error"):
                        lines.append(f"| {lt['test']} | {lt.get('latency_us', 0):.2f} us | "
                                     f"<= {lt.get('max_allowed_us', 0)} us | {lt.get('status', '?')} |")
                lines.append("")
            passed = rdma.get("passed", False)
            lines.append(f"**Overall: {'PASS' if passed else 'FAIL'}**\n")

        # --- Training ---
        training = results.get("training")
        if training and not training.get("error"):
            lines.append("## Training Simulation\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Model | {training.get('model', 'N/A')} |")
            lines.append(f"| Params | {training.get('total_params_m', 0):.1f}M |")
            lines.append(f"| Throughput | {training.get('throughput_tokens_per_sec', 0):.0f} tokens/sec |")
            lines.append(f"| Avg Step Time | {training.get('avg_step_time_ms', 0):.1f} ms |")
            lines.append(f"| Peak Memory | {training.get('peak_memory_gb', 0):.1f} GB |")
            lines.append(f"| Final Loss | {training.get('final_loss', 'N/A')} |")
            lines.append("")

        # --- Footer ---
        lines.append("---")
        lines.append(f"*Generated by GPU Test Suite v0.2.0*")

        content = "\n".join(lines)
        with open(output, "w") as f:
            f.write(content)
        self.console.print(f"[green]Markdown report saved to: {output}[/green]")
        return output

    def _extract_memory_results(self, results: dict) -> dict:
        """Extract memory benchmark data from either full-suite or single-test format."""
        if "memory_bench" in results:
            data = results["memory_bench"]
            return data.get("memory", data) if isinstance(data, dict) else {}
        if "benchmark" in results:
            bench = results["benchmark"]
            if isinstance(bench, dict) and "memory" in bench:
                return bench["memory"]
        return {}

    def _extract_compute_results(self, results: dict) -> dict:
        """Extract compute benchmark data from either full-suite or single-test format."""
        if "compute_bench" in results:
            data = results["compute_bench"]
            return data.get("compute", data) if isinstance(data, dict) else {}
        if "benchmark" in results:
            bench = results["benchmark"]
            if isinstance(bench, dict) and "compute" in bench:
                return bench["compute"]
        return {}

    def _build_summary(self, results: dict) -> list[tuple[str, str]]:
        """Build summary verdict list from results."""
        items = []

        # GPU Info
        if "gpu_info" in results:
            gi = results["gpu_info"]
            if gi.get("error"):
                items.append(("GPU Info", f"ERROR: {gi['error']}"))
            else:
                items.append(("GPU Info", f"PASS ({gi.get('gpu_count', '?')} GPUs detected)"))

        # Health
        if "health" in results:
            h = results["health"]
            if h.get("error"):
                items.append(("Health Check", f"ERROR: {h['error']}"))
            elif h.get("passed"):
                items.append(("Health Check", "PASS"))
            else:
                items.append(("Health Check", "FAIL"))

        # Memory Bandwidth
        mem = self._extract_memory_results(results)
        if mem:
            if mem.get("error"):
                items.append(("Memory Bandwidth", f"ERROR: {mem['error']}"))
            else:
                eff = mem.get("efficiency_pct", 0)
                verdict = "PASS" if eff >= 80 else ("WARN" if eff >= 50 else "FAIL")
                items.append(("Memory Bandwidth", f"{verdict} ({eff:.1f}%)"))

        # Compute
        comp = self._extract_compute_results(results)
        if comp:
            if comp.get("error"):
                items.append(("Compute Throughput", f"ERROR: {comp['error']}"))
            else:
                eff_pct = comp.get("efficiency_pct", {})
                valid_effs = [v for v in eff_pct.values() if isinstance(v, (int, float)) and v > 0]
                if valid_effs:
                    worst = min(valid_effs)
                    verdict = "PASS" if worst >= 80 else ("WARN" if worst >= 50 else "FAIL")
                    items.append(("Compute Throughput", f"{verdict} (worst {worst:.1f}%)"))
                else:
                    items.append(("Compute Throughput", "N/A"))

        # NCCL
        if "nccl" in results:
            n = results["nccl"]
            if n.get("error"):
                items.append(("NCCL", f"ERROR: {n['error']}"))
            elif n.get("passed"):
                items.append(("NCCL", "PASS"))
            else:
                items.append(("NCCL", "FAIL"))

        # Stress
        if "stress" in results:
            s = results["stress"]
            if s.get("error"):
                items.append(("Stress Test", f"ERROR: {s['error']}"))
            elif s.get("passed"):
                items.append(("Stress Test", "PASS"))
            else:
                items.append(("Stress Test", "FAIL"))

        # RDMA
        if "rdma" in results:
            r = results["rdma"]
            if r.get("error"):
                items.append(("RDMA", f"ERROR: {r['error']}"))
            elif r.get("passed"):
                items.append(("RDMA", "PASS"))
            else:
                items.append(("RDMA", "FAIL"))

        # Training
        if "training" in results:
            t = results["training"]
            if t.get("error"):
                items.append(("Training", f"ERROR: {t['error']}"))
            else:
                tps = t.get("throughput_tokens_per_sec", 0)
                items.append(("Training", f"PASS ({tps:.0f} tokens/sec)"))

        return items
