#!/usr/bin/env python3
"""Aggregate scaling study results from multiple runs.

Reads results.json files from one or more scaling study output directories
and generates a consolidated scaling report with averages and std devs.

Usage:
    # Single run:
    python3 aggregate_scaling.py outputs/scaling_study/20260412_091635

    # Multiple runs (averages across runs):
    python3 aggregate_scaling.py outputs/scaling_study/20260412_091635 outputs/scaling_study/20260412_130217
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


MODELS = [
    "agpt_2b",
    "agpt_20b",
    "agpt_80b",
    "agpt_80b_wide",
    "moe_2b",
    "moe_7b",
    "moe_10b_2b",
]


def _find_results_files(base_dir: Path) -> list[Path]:
    """Find all results.json files under base_dir.

    Handles both flat layout (n4/results.json) and group layout
    (n4/light/results.json, n4/heavy/results.json).
    """
    files: list[Path] = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("n"):
            continue

        # Direct results.json in n<N>/
        direct = subdir / "results.json"
        if direct.exists():
            files.append(direct)

        # Group subdirs (light/, heavy/) in n<N>/
        for group_dir in sorted(subdir.iterdir()):
            if not group_dir.is_dir():
                continue
            group_file = group_dir / "results.json"
            if group_file.exists():
                files.append(group_file)

    return files


def load_results(base_dir: Path) -> dict[int, dict[str, dict]]:
    """Load results.json from each n<N>/ subdirectory.

    Returns: {nodes: {model: result_dict}}
    """
    data: dict[int, dict[str, dict]] = {}

    for results_file in _find_results_files(base_dir):
        with open(results_file) as f:
            job_data = json.load(f)

        nodes = job_data["nodes"]
        if nodes not in data:
            data[nodes] = {}

        for result in job_data["results"]:
            model = result["model"]
            data[nodes][model] = result

    return data


def load_multi_results(
    dirs: list[Path],
) -> dict[int, dict[str, list[dict]]]:
    """Load results from multiple run directories.

    Handles both flat (n4/results.json) and grouped
    (n4/light/results.json + n4/heavy/results.json) layouts.

    Returns: {nodes: {model: [result_dict, ...]}}
    """
    multi: dict[int, dict[str, list[dict]]] = {}

    for d in dirs:
        run_data = load_results(d)
        for nodes, model_results in run_data.items():
            if nodes not in multi:
                multi[nodes] = {}
            for model, result in model_results.items():
                if model not in multi[nodes]:
                    multi[nodes][model] = []
                multi[nodes][model].append(result)

    return multi


def avg_and_std(values: list[float | int | None]) -> tuple[float | None, float | None]:
    """Compute mean and std dev, ignoring None values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, None
    variance = sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)
    return mean, math.sqrt(variance)


def fmt_num(val, width: int = 8) -> str:
    if val is None:
        return "-".center(width)
    if isinstance(val, float):
        if val >= 100:
            return f"{val:>{width},.0f}"
        return f"{val:>{width}.1f}"
    return f"{val:>{width},}"


def fmt_avg_std(mean, std, width: int = 12) -> str:
    """Format as 'mean +/- std' or just 'mean' if no std."""
    if mean is None:
        return "-".center(width)
    if std is None or std == 0:
        if isinstance(mean, float) and mean < 100:
            return f"{mean:>{width}.1f}"
        return f"{int(mean):>{width},}"
    if mean >= 100:
        return f"{int(mean):>,}+/-{int(std)}".rjust(width)
    return f"{mean:.1f}+/-{std:.1f}".rjust(width)


def fmt_pct(val, width: int = 7) -> str:
    if val is None:
        return "-".center(width)
    return f"{val:>{width}.1f}%"


def generate_report(
    multi: dict[int, dict[str, list[dict]]],
    num_runs: int,
    output_dir: Path,
) -> str:
    node_counts = sorted(multi.keys())
    if not node_counts:
        return "No results found.\n"

    lines: list[str] = []
    lines.append("# Scaling Study Report")
    lines.append("")

    # Metadata from first available result
    first_results = multi[node_counts[0]]
    sample_list = next(iter(first_results.values()), [])
    sample = sample_list[0] if sample_list else {}
    lines.append(f"- **Machine:** {sample.get('machine', 'unknown')}")
    lines.append(f"- **Node counts:** {', '.join(str(n) for n in node_counts)}")
    lines.append(f"- **Steps per run:** {sample.get('steps', '?')}")
    lines.append(f"- **Runs averaged:** {num_runs}")
    lines.append("")

    node_headers = [f"{n}N" for n in node_counts]
    device_headers = [f"({n * 12})" for n in node_counts]

    # Use wider columns for avg+/-std format
    has_multi = num_runs > 1
    col_w = 14 if has_multi else 9

    def table_header(title: str, subtitle: str = "") -> list[str]:
        hdr = [f"## {title}"]
        if subtitle:
            hdr.append(f"_{subtitle}_")
        hdr.append("")
        row1 = f"| {'Model':<10} |"
        row2 = f"|{'-' * 12}|"
        for i, nh in enumerate(node_headers):
            cell = f"{nh} {device_headers[i]}"
            row1 += f" {cell:>{col_w}} |"
            row2 += f"-{'-' * col_w}-|"
        hdr.append(row1)
        hdr.append(row2)
        return hdr

    def get_ok_values(
        results: list[dict], key: str
    ) -> list[float | int | None]:
        return [
            r.get(key)
            for r in results
            if r.get("status") == "OK" and r.get(key) is not None
        ]

    # NOTE: TPS reported by torchtitan is per-GPU throughput.
    # Total throughput = TPS * num_devices.

    # ---- Table 1: Per-GPU TPS ----
    lines.extend(table_header("Per-GPU Throughput (tokens/s/GPU)"))
    for model in MODELS:
        row = f"| {model:<10} |"
        for n in node_counts:
            results = multi[n].get(model, [])
            tps_vals = get_ok_values(results, "tps")
            if tps_vals:
                mean, std = avg_and_std(tps_vals)
                if has_multi:
                    row += f" {fmt_avg_std(mean, std, col_w)} |"
                else:
                    row += f" {fmt_num(mean, col_w)} |"
            else:
                statuses = [r.get("status", "N/A") for r in results]
                status = statuses[0] if statuses else "N/A"
                row += f" {status:>{col_w}} |"
        lines.append(row)
    lines.append("")

    # ---- Table 2: Total TPS ----
    lines.extend(
        table_header("Total Throughput (tokens/s)", "per-GPU TPS x num_devices")
    )
    for model in MODELS:
        row = f"| {model:<10} |"
        for n in node_counts:
            results = multi[n].get(model, [])
            tps_vals = get_ok_values(results, "tps")
            if tps_vals:
                devices = n * 12
                total_vals = [v * devices for v in tps_vals if v is not None]
                mean, std = avg_and_std(total_vals)
                if has_multi:
                    row += f" {fmt_avg_std(mean, std, col_w)} |"
                else:
                    row += f" {fmt_num(mean, col_w)} |"
            else:
                row += f" {'-':>{col_w}} |"
        lines.append(row)
    lines.append("")

    # ---- Table 3: Weak Scaling Efficiency ----
    # Per-GPU TPS should stay constant for perfect weak scaling.
    # Efficiency = per_gpu_tps(N) / per_gpu_tps(baseline) x 100
    baseline_n = node_counts[0]
    lines.extend(
        table_header(
            "Weak Scaling Efficiency",
            f"per-GPU TPS(N) / per-GPU TPS({baseline_n}N) x 100",
        )
    )
    for model in MODELS:
        row = f"| {model:<10} |"
        baseline_results = multi[baseline_n].get(model, [])
        baseline_vals = get_ok_values(baseline_results, "tps")
        baseline_tps, _ = avg_and_std(baseline_vals) if baseline_vals else (None, None)

        for n in node_counts:
            results = multi[n].get(model, [])
            tps_vals = get_ok_values(results, "tps")
            mean_tps, _ = avg_and_std(tps_vals) if tps_vals else (None, None)
            if baseline_tps and mean_tps:
                efficiency = (mean_tps / baseline_tps) * 100
                row += f" {fmt_pct(efficiency, col_w)} |"
            else:
                row += f" {'-':>{col_w}} |"
        lines.append(row)
    lines.append("")

    # ---- Table 4: MFU ----
    lines.extend(table_header("Model FLOPs Utilization (MFU %)"))
    for model in MODELS:
        row = f"| {model:<10} |"
        for n in node_counts:
            results = multi[n].get(model, [])
            mfu_vals = get_ok_values(results, "mfu")
            if mfu_vals:
                mean, _ = avg_and_std(mfu_vals)
                row += f" {fmt_pct(mean, col_w)} |"
            else:
                row += f" {'-':>{col_w}} |"
        lines.append(row)
    lines.append("")

    # ---- Table 5: Memory ----
    lines.extend(table_header("Memory Usage"))
    for model in MODELS:
        row = f"| {model:<10} |"
        for n in node_counts:
            results = multi[n].get(model, [])
            ok_results = [r for r in results if r.get("status") == "OK"]
            if ok_results and ok_results[0].get("memory"):
                mem = ok_results[0]["memory"]
                row += f" {mem:>{col_w}} |"
            else:
                statuses = [r.get("status", "N/A") for r in results]
                status = statuses[0] if statuses else "N/A"
                row += f" {status:>{col_w}} |"
        lines.append(row)
    lines.append("")

    # ---- Table 6: Status Overview ----
    lines.extend(table_header("Run Status"))
    for model in MODELS:
        row = f"| {model:<10} |"
        for n in node_counts:
            results = multi[n].get(model, [])
            if not results:
                row += f" {'MISSING':>{col_w}} |"
            else:
                statuses = [r.get("status", "?") for r in results]
                ok_count = sum(1 for s in statuses if s == "OK")
                if ok_count == len(statuses):
                    cell = f"OK({ok_count})" if has_multi else "OK"
                elif ok_count > 0:
                    cell = f"{ok_count}/{len(statuses)} OK"
                else:
                    cell = statuses[0]
                row += f" {cell:>{col_w}} |"
        lines.append(row)
    lines.append("")

    lines.append(f"Results directory: `{output_dir}/`")

    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <dir1> [dir2] [dir3] ...",
            file=sys.stderr,
        )
        print(
            "  Single dir:  aggregate one run",
            file=sys.stderr,
        )
        print(
            "  Multiple dirs: average across runs",
            file=sys.stderr,
        )
        sys.exit(1)

    dirs = [Path(d) for d in sys.argv[1:]]
    for d in dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory", file=sys.stderr)
            sys.exit(1)

    multi = load_multi_results(dirs)
    if not multi:
        print(
            f"Error: no results.json files found in {dirs}",
            file=sys.stderr,
        )
        sys.exit(1)

    report = generate_report(multi, num_runs=len(dirs), output_dir=dirs[0])

    # Write report to the first directory
    report_path = dirs[0] / "scaling_report.md"
    report_path.write_text(report)

    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
