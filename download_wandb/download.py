#!/usr/bin/env python3
"""Download metrics and metadata from one Weights & Biases run or project.

Examples:
    python download.py entity/project/run_id --output ./wandb_exports
    python download.py entity/project --output ./wandb_exports

Authenticate first with ``wandb login`` or set the ``WANDB_API_KEY`` environment
variable. By default the export contains hyperparameters, summaries, and
training/evaluation history. Add ``--include-system`` for CPU/GPU/RAM metrics,
or ``--include-files`` for checkpoints, videos, logs, and other run files.
"""


# /opt/conda/bin/python /tdmpc2/download_wandb/download.py   longyouyuan2022-italian-institute-of-telemedicine/tdmpc2   --output /tdmpc2/download_wandb/wandb_exports

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import wandb


def json_default(value: Any) -> str:
    """Convert API values that are not directly JSON serializable."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def run_snapshot(run: Any, include_system: bool) -> dict[str, Any]:
    """Return UI-visible run information that is useful offline."""
    fields = (
        "id", "name", "display_name", "entity", "project", "path", "url", "state",
        "created_at", "updated_at", "notes", "tags", "group", "job_type", "sweep_name",
        "username", "user", "host",
    )
    details: dict[str, Any] = {}
    for field in fields:
        try:
            details[field] = getattr(run, field)
        except (AttributeError, KeyError):
            # Older W&B clients do not expose every field.
            continue

    snapshot = {
        "run": details,
        "config": dict(run.config),
        "raw_config": dict(run.rawconfig),
        "summary": dict(run.summary),
        "summary_metrics": dict(run.summary_metrics),
    }
    if include_system:
        snapshot["system_metrics"] = dict(run.system_metrics)
        snapshot["metadata"] = run.metadata
    return snapshot


def metric_keys(row: dict[str, Any], group: str) -> list[str]:
    """Find metric names such as ``train/loss`` or ``eval.accuracy``."""
    prefixes = (f"{group}/", f"{group}_", f"{group}.")
    return [
        key
        for key in row
        if (
            isinstance(key, str)
            and key.lower().startswith(prefixes)
            and row[key] is not None
        )
    ]


def download_train_eval_history(run: Any, history_dir: Path) -> None:
    """Save only training and evaluation metrics into separate JSONL files."""
    paths = {
        "train": history_dir / "train_history.jsonl",
        "eval": history_dir / "eval_history.jsonl",
    }
    counts = {"train": 0, "eval": 0}
    x_axis_keys = {"_step", "_timestamp", "_runtime"}

    with paths["train"].open("w", encoding="utf-8") as train_file, paths[
        "eval"
    ].open("w", encoding="utf-8") as eval_file:
        files = {"train": train_file, "eval": eval_file}
        for row in run.scan_history(page_size=1000):
            for group, output_file in files.items():
                keys = metric_keys(row, group)
                if keys:
                    selected = {
                        key: value for key, value in row.items() if key in x_axis_keys or key in keys
                    }
                    output_file.write(json.dumps(selected, default=json_default) + "\n")
                    counts[group] += 1

    write_json(history_dir / "export_result.json", {"format": "jsonl", **counts})
    print(
        f"Downloaded {counts['train']} train row(s) and {counts['eval']} eval row(s)."
    )


def download_system_history(run: Any, history_dir: Path) -> None:
    """Save the system-metrics stream (CPU/GPU/RAM/network measurements)."""
    # W&B exposes this stream separately from training history.  Request a large
    # limit; the raw W&B files are downloaded too, so they remain available if a
    # particularly long run exceeds the Public API export limit.
    rows = run.history(samples=100_000, stream="system", pandas=False)
    path = history_dir / "system_history.jsonl"
    with path.open("w", encoding="utf-8") as system_file:
        for row in rows:
            system_file.write(json.dumps(row, default=json_default) + "\n")
    print(f"Downloaded {len(rows)} system-metric row(s) to {path.name}.")


def download_run(
    run_path: str,
    output_root: Path,
    overwrite: bool,
    include_files: bool,
    include_system: bool,
    api: Any | None = None,
) -> Path:
    api = api or wandb.Api()
    run = api.run(run_path)

    # Keep local paths readable: the project and W&B run ID are sufficient to
    # identify a run; the entity/workspace can be very long.
    destination = output_root / f"{run.project}-{run.id}"
    destination.mkdir(parents=True, exist_ok=True)
    files_dir = destination / "files"
    history_dir = destination / "history"
    files_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)

    write_json(destination / "metadata.json", run_snapshot(run, include_system))
    download_train_eval_history(run, history_dir)
    if include_system:
        download_system_history(run, history_dir)

    if include_files:
        downloaded = 0
        for run_file in run.files(per_page=1000):
            run_file.download(root=str(files_dir), replace=overwrite, exist_ok=not overwrite)
            downloaded += 1
        print(f"Downloaded {downloaded} run file(s) to {files_dir}.")
    print(f"Export complete: {destination}")
    return destination


def download_project(
    project_path: str,
    output_root: Path,
    overwrite: bool,
    include_files: bool,
    include_system: bool,
) -> None:
    """Download every run in ``entity/project``, continuing after a failed run."""
    api = wandb.Api()
    runs = list(api.runs(project_path, per_page=100))
    if not runs:
        print(f"No runs found in project {project_path!r}.")
        return

    print(f"Found {len(runs)} run(s) in {project_path}; starting download.")
    failed: list[tuple[str, str]] = []
    for index, run in enumerate(runs, start=1):
        run_path = "/".join(run.path)
        print(f"\n[{index}/{len(runs)}] {run_path}")
        try:
            download_run(
                run_path, output_root, overwrite, include_files, include_system, api
            )
        except Exception as exc:
            # A bad/deleted run should not prevent the other runs being saved.
            failed.append((run_path, str(exc)))
            print(f"FAILED: {exc}")

    write_json(
        output_root / "project_export_result.json",
        {
            "project": project_path,
            "runs_found": len(runs),
            "runs_downloaded": len(runs) - len(failed),
            "failed": [{"run": path, "error": error} for path, error in failed],
        },
    )
    print(f"\nProject export complete. Failed runs: {len(failed)}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help=(
            "W&B path: entity/project/run_id for one run, or entity/project "
            "to download every run in that project"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wandb_exports"),
        help="Directory in which to create the export (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already-downloaded run files instead of resuming them",
    )
    parser.add_argument(
        "--include-files",
        action="store_true",
        help="Also download checkpoints, media, logs, and all other attached run files",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Also download CPU/GPU/RAM/network system-metric history",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parts = args.path.strip("/").split("/")
    if len(parts) == 2:
        download_project(
            args.path,
            args.output,
            args.overwrite,
            args.include_files,
            args.include_system,
        )
    elif len(parts) == 3:
        download_run(
            args.path,
            args.output,
            args.overwrite,
            args.include_files,
            args.include_system,
        )
    else:
        raise SystemExit("Path must be entity/project or entity/project/run_id.")
