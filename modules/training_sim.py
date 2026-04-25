"""Training simulation module - LLM training workload with PyTorch."""

import time
import subprocess
import shutil
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
except ImportError:
    pass


class TrainingSim:

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.train_cfg = config.get("training", {})

    def run(self) -> dict:
        if not TORCH_AVAILABLE:
            self.console.print("[yellow]PyTorch not available - skipping training simulation[/yellow]")
            return {"error": "pytorch_not_available"}

        gpu_count = torch.cuda.device_count()
        model_name = self.train_cfg.get("model", "gpt2")
        batch_size = self.train_cfg.get("batch_size", 8)
        seq_length = self.train_cfg.get("seq_length", 2048)
        num_steps = self.train_cfg.get("num_steps", 50)
        dtype_str = self.train_cfg.get("dtype", "bf16")

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        self.console.print(f"[cyan]Training Simulation[/cyan]")
        self.console.print(f"  Model: {model_name} | Batch: {batch_size} | Seq: {seq_length} | "
                           f"DType: {dtype_str} | Steps: {num_steps} | GPUs: {gpu_count}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self.console.print("[yellow]transformers not installed - using synthetic model[/yellow]")
            return self._run_synthetic(gpu_count, batch_size, seq_length, num_steps, dtype)

        try:
            self.console.print(f"  Loading {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if gpu_count > 1 else None,
            )

            total_params = sum(p.numel() for p in model.parameters())
            self.console.print(f"  Parameters: {total_params / 1e6:.1f}M")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones_like(input_ids)

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

            if dtype in (torch.float16, torch.bfloat16):
                scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

            step_times = []
            mem_usage = []

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(), console=self.console,
            ) as progress:
                task = progress.add_task("Training steps...", total=num_steps)

                for step in range(num_steps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    input_ids = input_ids.to(model.device)
                    attention_mask = attention_mask.to(model.device)

                    if dtype in (torch.float16, torch.bfloat16) and dtype != torch.bfloat16:
                        with torch.cuda.amp.autocast(dtype=dtype):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                            loss = outputs.loss
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                        loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    step_times.append(elapsed)

                    if torch.cuda.is_available():
                        mem_used = torch.cuda.max_memory_allocated() / 1024**3
                        mem_usage.append(mem_used)
                        torch.cuda.reset_peak_memory_stats()

                    progress.advance(task)

            avg_step_time = sum(step_times) / len(step_times)
            throughput = batch_size * seq_length / avg_step_time

            return {
                "model": model_name,
                "total_params_m": round(total_params / 1e6, 1),
                "gpu_count": gpu_count,
                "dtype": dtype_str,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "num_steps": num_steps,
                "avg_step_time_ms": round(avg_step_time * 1000, 1),
                "throughput_tokens_per_sec": round(throughput, 0),
                "throughput_samples_per_sec": round(batch_size / avg_step_time, 2),
                "peak_memory_gb": round(max(mem_usage) if mem_usage else 0, 2),
                "final_loss": round(loss.item(), 4) if hasattr(loss, 'item') else None,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.console.print(f"[yellow]Model loading failed: {e}[/yellow]")
            return self._run_synthetic(gpu_count, batch_size, seq_length, num_steps, dtype)

    def _run_synthetic(self, gpu_count, batch_size, seq_length, num_steps, dtype) -> dict:
        self.console.print("  Running synthetic training benchmark...")

        hidden_size = 4096
        num_layers = 6
        num_heads = 32
        vocab_size = 32000

        class SyntheticTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_size, nhead=num_heads,
                        dim_feedforward=hidden_size * 4,
                        batch_first=True,
                        dtype=dtype,
                    ) for _ in range(num_layers)
                ])
                self.head = torch.nn.Linear(hidden_size, vocab_size, dtype=dtype)

            def forward(self, x):
                h = self.embed(x).to(dtype)
                for layer in self.layers:
                    h = layer(h)
                return self.head(h)

        model = SyntheticTransformer().cuda()
        total_params = sum(p.numel() for p in model.parameters())

        self.console.print(f"  Synthetic params: {total_params / 1e6:.1f}M")

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()

        step_times = []
        mem_usage = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=self.console,
        ) as progress:
            task = progress.add_task("Synthetic training...", total=num_steps)

            for step in range(num_steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), input_ids.view(-1)
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                step_times.append(elapsed)

                mem_used = torch.cuda.max_memory_allocated() / 1024**3
                mem_usage.append(mem_used)
                torch.cuda.reset_peak_memory_stats()

                progress.advance(task)

        avg_step_time = sum(step_times) / len(step_times)
        throughput = batch_size * seq_length / avg_step_time

        return {
            "model": "synthetic_transformer",
            "total_params_m": round(total_params / 1e6, 1),
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "gpu_count": gpu_count,
            "dtype": str(dtype).replace("torch.", ""),
            "batch_size": batch_size,
            "seq_length": seq_length,
            "num_steps": num_steps,
            "avg_step_time_ms": round(avg_step_time * 1000, 1),
            "throughput_tokens_per_sec": round(throughput, 0),
            "throughput_samples_per_sec": round(batch_size / avg_step_time, 2),
            "peak_memory_gb": round(max(mem_usage) if mem_usage else 0, 2),
            "final_loss": round(loss.item(), 4),
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def print_results(results: dict, console: Console = None):
        c = console or Console()
        if "error" in results:
            c.print(f"[bold red]Error: {results['error']}[/bold red]")
            return

        c.print(f"\n[bold cyan]Training Simulation Results[/bold cyan]")

        table = Table(box=None, padding=(0, 1))
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        metrics = [
            ("Model", results.get("model", "N/A")),
            ("Parameters", f"{results.get('total_params_m', 'N/A')}M"),
            ("GPU Count", str(results.get("gpu_count", "N/A"))),
            ("DType", results.get("dtype", "N/A")),
            ("Batch Size", str(results.get("batch_size", "N/A"))),
            ("Seq Length", str(results.get("seq_length", "N/A"))),
            ("Steps", str(results.get("num_steps", "N/A"))),
            ("Avg Step Time", f"{results.get('avg_step_time_ms', 'N/A')} ms"),
            ("Throughput", f"{results.get('throughput_tokens_per_sec', 'N/A')} tokens/s"),
            ("Samples/sec", f"{results.get('throughput_samples_per_sec', 'N/A')}"),
            ("Peak Memory", f"{results.get('peak_memory_gb', 'N/A')} GB"),
            ("Final Loss", str(results.get("final_loss", "N/A"))),
        ]
        for label, val in metrics:
            table.add_row(label, str(val))

        c.print(table)
