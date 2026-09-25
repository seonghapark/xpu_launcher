#!/usr/bin/env python3
"""
Training monitor and summary generator for xpu_launcher TorchTitan jobs.
Monitors training logs, collects metrics, and generates a summary document.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import psutil


class TrainingMetricsCollector:
    """Collects metrics from training logs and system monitoring."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.start_time = datetime.now()
        self.metrics = {
            "losses": [],
            "accuracies": [],
            "timestamps": [],
            "loss_timestamps": [],
            "accuracy_timestamps": [],
        }
        self.system_metrics = {
            "max_memory_percent": 0,
            "max_cpu_percent": 0,
            "peak_memory_gb": 0,
        }
        self.errors = []
        self.warnings = []
        self.output_paths = {}

    def parse_training_logs(self) -> None:
        """Parse training logs to extract loss, accuracy, and other metrics."""
        log_files = sorted(self.log_dir.glob("*.log"))
        if not log_files:
            log_files = sorted(self.log_dir.glob("**/*.log"))

        for log_file in log_files:
            self._parse_single_log(log_file)

        # Also look for checkpoint files
        ckpt_dirs = list(self.log_dir.glob("**/checkpoint*"))
        for ckpt_dir in ckpt_dirs:
            if ckpt_dir.is_dir():
                checkpoints = list(ckpt_dir.glob("step-*"))
                for ckpt in checkpoints:
                    step_num = ckpt.name.replace("step-", "")
                    if not self.output_paths.get("checkpoints"):
                        self.output_paths["checkpoints"] = []
                    self.output_paths["checkpoints"].append(str(ckpt))

    def _parse_single_log(self, log_file: Path) -> None:
        """Parse a single log file for metrics."""
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract loss values (common patterns from TorchTitan)
            loss_pattern = r"(?:loss|Loss)[\s:=]+([0-9.]+)"
            for match in re.finditer(loss_pattern, content, re.IGNORECASE):
                try:
                    loss_val = float(match.group(1))
                    if 0 <= loss_val < 1e6:  # sanity check
                        self.metrics["losses"].append(loss_val)
                        self.metrics["loss_timestamps"].append(datetime.now())
                except ValueError:
                    pass

            # Extract accuracy values
            accuracy_pattern = r"(?:accuracy|Accuracy)[\s:=]+([0-9.]+)"
            for match in re.finditer(accuracy_pattern, content, re.IGNORECASE):
                try:
                    acc_val = float(match.group(1))
                    if 0 <= acc_val <= 1:
                        self.metrics["accuracies"].append(acc_val)
                        self.metrics["accuracy_timestamps"].append(datetime.now())
                except ValueError:
                    pass

            # Extract errors
            error_pattern = r"(?:ERROR|Error|error)[\s:]+(.+?)(?:\n|$)"
            for match in re.finditer(error_pattern, content):
                error_msg = match.group(1).strip()
                if error_msg and error_msg not in self.errors:
                    self.errors.append(error_msg[:200])  # truncate long errors

            # Extract warnings
            warning_pattern = r"(?:WARNING|Warning|warning)[\s:]+(.+?)(?:\n|$)"
            for match in re.finditer(warning_pattern, content):
                warning_msg = match.group(1).strip()
                if warning_msg and warning_msg not in self.warnings:
                    self.warnings.append(warning_msg[:200])

            # Record the log file path
            if "logs" not in self.output_paths:
                self.output_paths["logs"] = []
            self.output_paths["logs"].append(str(log_file))

        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")

    def monitor_system(self) -> None:
        """Monitor system resource usage."""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            memory_percent = process.memory_percent()

            self.system_metrics["max_memory_percent"] = max(
                self.system_metrics["max_memory_percent"], memory_percent
            )
            self.system_metrics["peak_memory_gb"] = max(
                self.system_metrics["peak_memory_gb"], mem_info.rss / 1024 / 1024 / 1024
            )

            try:
                cpu_percent = process.cpu_percent(interval=0.1)
                self.system_metrics["max_cpu_percent"] = max(
                    self.system_metrics["max_cpu_percent"], cpu_percent
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        except Exception:
            pass

    def get_metrics_summary(self) -> dict[str, Any]:
        """Return a summary of collected metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        summary = {
            "elapsed_seconds": elapsed,
            "elapsed_time": self._format_duration(elapsed),
            "losses": {
                "count": len(self.metrics["losses"]),
                "min": min(self.metrics["losses"]) if self.metrics["losses"] else None,
                "max": max(self.metrics["losses"]) if self.metrics["losses"] else None,
                "mean": (
                    sum(self.metrics["losses"]) / len(self.metrics["losses"])
                    if self.metrics["losses"]
                    else None
                ),
                "trend": self._compute_trend(self.metrics["losses"]),
            },
            "accuracies": {
                "count": len(self.metrics["accuracies"]),
                "min": min(self.metrics["accuracies"])
                if self.metrics["accuracies"]
                else None,
                "max": max(self.metrics["accuracies"])
                if self.metrics["accuracies"]
                else None,
                "mean": (
                    sum(self.metrics["accuracies"]) / len(self.metrics["accuracies"])
                    if self.metrics["accuracies"]
                    else None
                ),
                "trend": self._compute_trend(self.metrics["accuracies"]),
            },
            "system": self.system_metrics,
            "errors": self.errors[:10],  # top 10
            "warnings": self.warnings[:10],  # top 10
            "output_paths": self.output_paths,
        }
        return summary

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds as HH:MM:SS."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(int(td.total_seconds()), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _compute_trend(values: list[float]) -> dict[str, Any]:
        """Compute trend (improving/degrading) from a list of values."""
        if len(values) < 2:
            return {"status": "insufficient_data", "change": 0}

        first_half = sum(values[: len(values) // 2]) / max(1, len(values) // 2)
        second_half = sum(values[len(values) // 2 :]) / max(1, len(values) - len(values) // 2)
        change = second_half - first_half

        # For loss, decreasing is good; for accuracy, increasing is good
        return {
            "change": change,
            "direction": "improving" if change < 0 else "degrading" if change > 0 else "stable",
        }


def generate_summary_document(
    log_dir: Path, output_file: Optional[Path] = None
) -> Path:
    """Generate a comprehensive summary document."""
    collector = TrainingMetricsCollector(log_dir)
    collector.parse_training_logs()
    collector.monitor_system()

    if output_file is None:
        output_file = log_dir / "TRAINING_SUMMARY.md"

    metrics = collector.get_metrics_summary()

    # Generate markdown document
    doc_lines = [
        "# Training Summary Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Training Duration",
        f"- **Elapsed Time**: {metrics['elapsed_time']}",
        f"- **Start Time**: {collector.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Loss Metrics",
        f"- **Observations**: {metrics['losses']['count']}",
    ]

    if metrics["losses"]["mean"] is not None:
        doc_lines.extend([
            f"- **Mean Loss**: {metrics['losses']['mean']:.6f}",
            f"- **Min Loss**: {metrics['losses']['min']:.6f}",
            f"- **Max Loss**: {metrics['losses']['max']:.6f}",
            f"- **Trend**: {metrics['losses']['trend']['direction']} "
            f"(Δ {metrics['losses']['trend']['change']:.6f})",
        ])
    else:
        doc_lines.append("- **No loss data found in logs**")

    doc_lines.extend(["", "## Accuracy Metrics", f"- **Observations**: {metrics['accuracies']['count']}"])

    if metrics["accuracies"]["mean"] is not None:
        doc_lines.extend([
            f"- **Mean Accuracy**: {metrics['accuracies']['mean']:.6f}",
            f"- **Min Accuracy**: {metrics['accuracies']['min']:.6f}",
            f"- **Max Accuracy**: {metrics['accuracies']['max']:.6f}",
            f"- **Trend**: {metrics['accuracies']['trend']['direction']} "
            f"(Δ {metrics['accuracies']['trend']['change']:.6f})",
        ])
    else:
        doc_lines.append("- **No accuracy data found in logs**")

    doc_lines.extend([
        "",
        "## System Resource Consumption",
        f"- **Peak Memory**: {metrics['system']['peak_memory_gb']:.2f} GB",
        f"- **Max Memory %**: {metrics['system']['max_memory_percent']:.2f}%",
        f"- **Max CPU %**: {metrics['system']['max_cpu_percent']:.2f}%",
        "",
        "## Output Paths",
    ])

    if metrics["output_paths"]:
        for key, paths in metrics["output_paths"].items():
            doc_lines.append(f"### {key.replace('_', ' ').title()}")
            if isinstance(paths, list):
                for path in paths:
                    doc_lines.append(f"- `{path}`")
            else:
                doc_lines.append(f"- `{paths}`")
            doc_lines.append("")
    else:
        doc_lines.append("- No output paths found")

    if collector.errors:
        doc_lines.extend(["", "## Errors", ""])
        for i, error in enumerate(collector.errors, 1):
            doc_lines.append(f"{i}. {error}")
        doc_lines.append("")

    if collector.warnings:
        doc_lines.extend(["", "## Warnings", ""])
        for i, warning in enumerate(collector.warnings, 1):
            doc_lines.append(f"{i}. {warning}")
        doc_lines.append("")

    doc_lines.extend([
        "## Additional Information",
        f"- **Report Generated**: {datetime.now().isoformat()}",
        f"- **Log Directory**: `{log_dir}`",
        "- For detailed metrics, check individual log files in the output directory.",
    ])

    # Write the document
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))

    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_monitor.py <log_dir> [output_summary.md]")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not log_dir.exists():
        print(f"Error: log directory does not exist: {log_dir}")
        sys.exit(1)

    summary_path = generate_summary_document(log_dir, output_file)
    print(f"Summary document generated: {summary_path}")
