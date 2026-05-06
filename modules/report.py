"""Report generation module - export test results to JSON/HTML."""

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
