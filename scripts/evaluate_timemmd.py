#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chronos_echo import Chronos2EchoPipeline
from chronos_echo.timemmd import TimeMMDBatchDataset, TimeMMDWindowDataset


AURORA_CONFIG = {
    "Agriculture": (192, 48, 256),
    "Climate": (192, 48, 256),
    "Economy": (192, 48, 256),
    "Energy": (1056, 48, 256),
    "Environment": (528, 48, 256),
    "Health": (96, 48, 256),
    "Security": (220, 24, 256),
    "SocialGood": (192, 48, 256),
    "Traffic": (96, 48, 256),
    "Weather": (1440, 48, 256),
    "EWJ": (1056, 48, 256),
    "KR": (1056, 48, 256),
    "MDT": (528, 48, 256),
}

CSV_FIELDS = ["model", "domain", "pred_len", "seq_len", "mse", "mae", "rmse", "mape", "mspe", "rse", "corr", "n_windows"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Chronos-2-ECHO and Aurora on Time-MMD.")
    parser.add_argument("--checkpoint", required=True, help="Chronos-2-ECHO checkpoint path.")
    parser.add_argument("--aurora-model", default="../Aurora/aurora", help="Local Aurora model checkpoint path.")
    parser.add_argument("--aurora-root", default="../Aurora", help="Local Aurora repository path.")
    parser.add_argument("--data-root", required=True, help="Directory containing Time-MMD CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics.csv, comparison.csv, summary.json, repro.txt.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--domains", default=None, help="Comma-separated Time-MMD domains. Defaults to Aurora domains.")
    parser.add_argument("--manifest", default=None, help="Optional CSV manifest overriding Aurora defaults.")
    return parser


def set_reproducible(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic = True
    except Exception:
        deterministic = False
    return {"seed": seed, "torch_deterministic_algorithms": deterministic}


def horizons_for(domain: str) -> list[int]:
    if domain in {"Energy", "Health"}:
        return [12, 24, 36, 48]
    if domain in {"Environment", "Weather"}:
        return [48, 96, 192, 336]
    return [6, 8, 10, 12]


def default_tasks(domains: list[str] | None, batch_size: int | None) -> list[dict[str, Any]]:
    selected = domains or list(AURORA_CONFIG)
    tasks = []
    for domain in selected:
        seq_len, inference_token_len, default_batch_size = AURORA_CONFIG[domain]
        for pred_len in horizons_for(domain):
            tasks.append(
                {
                    "domain": domain,
                    "data_path": f"{domain}.csv",
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "features": "S",
                    "target": "OT",
                    "text_column": "fact",
                    "split": "test",
                    "inference_token_len": inference_token_len,
                    "batch_size": batch_size or default_batch_size,
                }
            )
    return tasks


def load_manifest(path: Path, batch_size: int | None) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    tasks = []
    for row in rows:
        domain = row["domain"]
        seq_len, inference_token_len, default_batch_size = AURORA_CONFIG.get(domain, (int(row["seq_len"]), 48, 256))
        tasks.append(
            {
                "domain": domain,
                "data_path": row.get("data_path") or f"{domain}.csv",
                "seq_len": int(row.get("seq_len") or seq_len),
                "pred_len": int(row["pred_len"]),
                "features": row.get("features") or "S",
                "target": row.get("target") or "OT",
                "text_column": row.get("text_column") or "fact",
                "split": row.get("split") or "test",
                "inference_token_len": int(row.get("inference_token_len") or inference_token_len),
                "batch_size": int(row.get("batch_size") or batch_size or default_batch_size),
            }
        )
    return tasks


def metric_row(model: str, task: dict[str, Any], pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(math.sqrt(mse))
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(diff / true)
        spe = np.square(diff / true)
    mape = float(np.mean(ape[np.isfinite(ape)])) if np.isfinite(ape).any() else float("nan")
    mspe = float(np.mean(spe[np.isfinite(spe)])) if np.isfinite(spe).any() else float("nan")
    denom = float(np.sqrt(np.sum((true - true.mean()) ** 2)))
    rse = float(np.sqrt(np.sum(diff**2)) / denom) if denom else float("nan")
    corr_denom = np.sqrt(np.sum((true - true.mean()) ** 2 * (pred - pred.mean()) ** 2)) + 1e-12
    corr = float(0.01 * np.sum((true - true.mean()) * (pred - pred.mean())) / corr_denom)
    return {
        "model": model,
        "domain": task["domain"],
        "pred_len": task["pred_len"],
        "seq_len": task["seq_len"],
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "rse": rse,
        "corr": corr,
        "n_windows": int(true.size),
    }


def evaluate_chronos(*, model_name: str, tasks: list[dict[str, Any]], checkpoint: Path, data_root: Path, device: str) -> list[dict[str, Any]]:
    pipeline = Chronos2EchoPipeline.from_pretrained(checkpoint, device_map=device)
    rows = []
    for task in tasks:
        output = pipeline.predict_timemmd(
            root_path=data_root,
            data_path=task["data_path"],
            target=task["target"],
            seq_len=task["seq_len"],
            pred_len=task["pred_len"],
            features=task["features"],
            batch_size=task["batch_size"],
            flag=task["split"],
            text_column=task["text_column"],
            missing_text="No information available",
        )
        rows.append(metric_row(model_name, task, output["predictions"].numpy(), output["targets"].numpy()))
    return rows


def load_aurora_model(aurora_root: Path, aurora_model: Path, device: str):
    aurora_root = aurora_root.resolve()
    if str(aurora_root) not in sys.path:
        sys.path.insert(0, str(aurora_root))
    try:
        from aurora.modeling_aurora import AuroraForPrediction
    except Exception as exc:
        raise RuntimeError(f"Failed to import Aurora from {aurora_root}: {exc}") from exc
    model = AuroraForPrediction.from_pretrained(str(aurora_model)).to(device)
    model.eval()
    return model


def load_aurora_tokenizer(aurora_root: Path):
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(str(aurora_root / "aurora" / "bert_config"), local_files_only=True)


def evaluate_aurora(
    *,
    model_name: str,
    tasks: list[dict[str, Any]],
    aurora_model: Path,
    aurora_root: Path,
    data_root: Path,
    device: str,
) -> list[dict[str, Any]]:
    model = load_aurora_model(aurora_root, aurora_model, device)
    tokenizer = load_aurora_tokenizer(aurora_root)
    rows = []
    for task in tasks:
        window_dataset = TimeMMDWindowDataset(
            root_path=data_root,
            data_path=task["data_path"],
            flag=task["split"],
            seq_len=task["seq_len"],
            pred_len=task["pred_len"],
            target=task["target"],
            features=task["features"],
            tokenizer=tokenizer,
            max_text_length=500,
            text_column=task["text_column"],
            missing_text="No information available",
        )
        dataset = TimeMMDBatchDataset(
            window_dataset,
            batch_size=task["batch_size"],
            output_patch_size=task["pred_len"],
            shuffle=False,
            repeat=False,
        )
        preds = []
        trues = []
        loader = DataLoader(dataset, batch_size=None, pin_memory=str(device).startswith("cuda"))
        with torch.no_grad():
            for batch in loader:
                context = batch["context"].float().to(device)
                text_input_ids = batch["text_input_ids"].to(device)
                text_attention_mask = batch["text_attention_mask"].to(device)
                text_token_type_ids = batch["text_token_type_ids"].to(device)
                generated = model.generate(
                    inputs=context,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    text_token_type_ids=text_token_type_ids,
                    inference_token_len=task["inference_token_len"],
                    max_output_length=task["pred_len"],
                    max_text_token_length=500,
                    num_samples=100,
                )
                prediction = generated.mean(1).detach().cpu().numpy()
                target = batch["future_target"].detach().cpu().numpy()
                preds.append(prediction)
                trues.append(target)
        pred = np.concatenate(preds, axis=0)[..., : task["pred_len"]]
        true = np.concatenate(trues, axis=0)[..., : task["pred_len"]]
        # Aurora generates in the scaled Time-MMD space; convert S-task rows back to CSV units to match ECHO.
        if window_dataset.n_targets == 1:
            pred = window_dataset.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
            true = window_dataset.inverse_transform(true.reshape(-1, 1)).reshape(true.shape)
        rows.append(metric_row(model_name, task, pred, true))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in metrics:
        by_key.setdefault((row["domain"], int(row["pred_len"])), {})[row["model"]] = row
    rows = []
    for (domain, pred_len), values in sorted(by_key.items()):
        chronos = values.get("chronos2_echo")
        aurora = values.get("aurora")
        if not chronos or not aurora:
            continue
        aurora_mse = float(aurora["mse"])
        chronos_mse = float(chronos["mse"])
        rows.append(
            {
                "domain": domain,
                "pred_len": pred_len,
                "chronos2_echo_mse": chronos_mse,
                "aurora_mse": aurora_mse,
                "delta_mse": chronos_mse - aurora_mse,
                "relative_mse_change": (chronos_mse - aurora_mse) / aurora_mse if aurora_mse else float("nan"),
                "chronos2_echo_mae": chronos["mae"],
                "aurora_mae": aurora["mae"],
            }
        )
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_repro(path: Path, args: argparse.Namespace, repro: dict[str, Any]) -> None:
    lines = [
        "command: " + " ".join(sys.argv),
        f"cwd: {Path.cwd().resolve()}",
        f"git_commit: {git_commit()}",
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"checkpoint: {Path(args.checkpoint).resolve()}",
        f"aurora_model: {Path(args.aurora_model).resolve()}",
        f"aurora_root: {Path(args.aurora_root).resolve()}",
        f"data_root: {Path(args.data_root).resolve()}",
        f"seed: {repro['seed']}",
        f"torch_deterministic_algorithms: {repro['torch_deterministic_algorithms']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repro = set_reproducible(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains.split(",") if args.domains else None
    tasks = load_manifest(Path(args.manifest), args.batch_size) if args.manifest else default_tasks(domains, args.batch_size)
    metrics = []
    failures = []
    try:
        metrics.extend(
            evaluate_chronos(
                model_name="chronos2_echo",
                tasks=tasks,
                checkpoint=Path(args.checkpoint),
                data_root=Path(args.data_root),
                device=args.device,
            )
        )
        metrics.extend(
            evaluate_aurora(
                model_name="aurora",
                tasks=tasks,
                aurora_model=Path(args.aurora_model),
                aurora_root=Path(args.aurora_root),
                data_root=Path(args.data_root),
                device=args.device,
            )
        )
    except Exception as exc:
        failures.append(str(exc))

    comparisons = comparison_rows(metrics)
    write_csv(output_dir / "metrics.csv", metrics, CSV_FIELDS)
    write_csv(
        output_dir / "comparison.csv",
        comparisons,
        [
            "domain",
            "pred_len",
            "chronos2_echo_mse",
            "aurora_mse",
            "delta_mse",
            "relative_mse_change",
            "chronos2_echo_mae",
            "aurora_mae",
        ],
    )
    summary = {
        "config": vars(args),
        "reproducibility": repro,
        "tasks": tasks,
        "n_metrics": len(metrics),
        "n_comparisons": len(comparisons),
        "failures": failures,
        "paths_exist": {
            "checkpoint": Path(args.checkpoint).exists(),
            "aurora_model": Path(args.aurora_model).exists(),
            "aurora_root": Path(args.aurora_root).exists(),
            "data_root": Path(args.data_root).exists(),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_repro(output_dir / "repro.txt", args, repro)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
