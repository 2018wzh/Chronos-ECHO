from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chronos_echo import Chronos2EchoConfig, Chronos2EchoPipeline
from .aurora_reference import SOURCE_URL, get_reference
from .dataset import TimeMMDBatchDataset, TimeMMDWindowDataset, create_timemmd_tokenizer
from .validate_dataset import validate_data_root


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.csv"
DEFAULT_CHECKPOINT_ROOT = ROOT / "checkpoints" / "chronos2_echo_fewshot"

PROJECT2_ECHO_CONFIG: dict[str, Any] = {
    "vision_model_name_or_path": "google/vit-base-patch16-224",
    "freeze_vision_backbone": True,
    "reconstruction_loss_weight": 0.5,
    "residual_scale_init": 1.0,
    "use_pseudo_image": False,
    "guard_against_baseline": False,
}


class NoopTokenizer:
    def __call__(self, text, *, padding, truncation, max_length, return_tensors):
        del text, padding, truncation
        if return_tensors != "pt":
            raise ValueError("NoopTokenizer only supports return_tensors='pt'")
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.zeros_like(input_ids),
            "token_type_ids": torch.zeros_like(input_ids),
        }


def load_manifest(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path is not None else DEFAULT_MANIFEST
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    tasks: list[dict[str, Any]] = []
    for row in rows:
        task = dict(row)
        for key in ["seq_len", "pred_len", "inference_token_len", "batch_size"]:
            task[key] = int(task[key])
        tasks.append(task)
    return tasks


def set_reproducible(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"seed": seed, "torch_deterministic_algorithms": True}


def aurora_metric_row(model: str, task: dict[str, Any], pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    diff = pred - true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(diff / true)))
        mspe = float(np.mean(np.square(diff / true)))
    rse_denom = np.sqrt(np.sum((true - true.mean()) ** 2))
    rse = float(np.sqrt(np.sum(diff**2)) / rse_denom) if rse_denom else float("nan")
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)) + 1e-12
    corr = float(np.asarray(0.01 * (u / d).mean(-1)).mean())
    return {
        "model": model,
        "domain": task["domain"],
        "pred_len": int(task["pred_len"]),
        "seq_len": int(task["seq_len"]),
        "mse": mse,
        "mae": mae,
        "rmse": float(math.sqrt(mse)),
        "mape": mape,
        "mspe": mspe,
        "rse": rse,
        "corr": corr,
        "n_windows": int(true.shape[0]) if true.ndim else int(true.size),
    }


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, list):
        if value and torch.is_tensor(value[0]):
            value = torch.stack([item.detach().cpu() for item in value])
        else:
            value = np.asarray(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _median_from_quantiles(quantiles: Any, means: Any) -> np.ndarray:
    means_array = _as_numpy(means)
    if means_array.ndim == 3 and means_array.shape[1] == 1:
        means_array = means_array[:, 0, :]
    if means_array.ndim == 2:
        return means_array

    quantile_array = _as_numpy(quantiles)
    if quantile_array.ndim == 4 and quantile_array.shape[1] == 1:
        quantile_array = quantile_array[:, 0, :, :]
    if quantile_array.ndim == 3:
        return quantile_array[..., quantile_array.shape[-1] // 2]
    return quantile_array


def _timemmd_window_dataset(task: dict[str, Any], data_root: Path, tokenizer: Any) -> TimeMMDWindowDataset:
    return TimeMMDWindowDataset(
        root_path=data_root,
        data_path=task["data_path"],
        flag=task["split"],
        seq_len=task["seq_len"],
        pred_len=task["pred_len"],
        target=task["target"],
        features=task["features"],
        tokenizer=tokenizer,
        text_column=task["text_column"],
    )


def _echo_tokenizer(pipeline: Chronos2EchoPipeline, tokenizer: Any | None = None) -> Any:
    if tokenizer is not None:
        return tokenizer
    config = pipeline._unwrap_echo_model(pipeline.model).echo_config
    return create_timemmd_tokenizer(config.text_tokenizer_name_or_path or config.text_model_name_or_path)


def _echo_timemmd_dataset(
    pipeline: Chronos2EchoPipeline,
    task: dict[str, Any],
    *,
    data_root: Path,
    flag: str,
    batch_size: int,
    shuffle: bool,
    repeat: bool,
    tokenizer: Any | None = None,
    few_shot_ratio: float = 0.1,
) -> tuple[TimeMMDWindowDataset, TimeMMDBatchDataset]:
    echo_config = pipeline._unwrap_echo_model(pipeline.model).echo_config
    window_dataset = TimeMMDWindowDataset(
        root_path=data_root,
        data_path=task["data_path"],
        flag=flag,
        seq_len=task["seq_len"],
        pred_len=task["pred_len"],
        target=task["target"],
        features=task["features"],
        tokenizer=_echo_tokenizer(pipeline, tokenizer),
        max_text_length=echo_config.max_text_length,
        text_column=task.get("text_column", "fact"),
        image_size=echo_config.vision_image_size,
        few_shot_ratio=few_shot_ratio,
    )
    return window_dataset, TimeMMDBatchDataset(
        window_dataset,
        batch_size=batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        shuffle=shuffle,
        repeat=repeat,
    )


def evaluate_chronos2(
    model_name: str,
    tasks: list[dict[str, Any]],
    *,
    data_root: Path,
    model_path: str,
    device: str,
    batch_size: int | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=device)
    rows = []
    for task in tasks:
        dataset = _timemmd_window_dataset(task, data_root, NoopTokenizer())
        preds = []
        trues = []
        bs = batch_size or task["batch_size"]
        for start in range(0, len(dataset), bs):
            items = [dataset[idx] for idx in range(start, min(start + bs, len(dataset)))]
            context = torch.stack([item["context"].squeeze(0) for item in items])
            quantiles, means = pipeline.predict_quantiles(
                context,
                prediction_length=task["pred_len"],
                quantile_levels=[0.5],
                limit_prediction_length=False,
            )
            pred = _median_from_quantiles(quantiles, means)[..., : task["pred_len"]]
            true = torch.stack([item["future_target"].squeeze(0) for item in items]).numpy()
            preds.append(np.asarray(pred).reshape(len(items), task["pred_len"], 1))
            trues.append(true.reshape(len(items), task["pred_len"], 1))
        rows.append(aurora_metric_row(model_name, task, np.concatenate(preds), np.concatenate(trues)))
    return rows


def _default_tokenizer_path(aurora_root: Path) -> str:
    local = aurora_root / "TimeMMD" / "aurora" / "bert_config"
    if not local.exists():
        raise FileNotFoundError(
            f"Aurora TimeMMD tokenizer config not found at {local}. "
            "Pass --aurora-root pointing to ../Aurora or set --text-tokenizer explicitly."
        )
    return str(local)


def resolve_echo_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(PROJECT2_ECHO_CONFIG)
    if args.echo_config:
        config.update(json.loads(args.echo_config))
    config.setdefault("text_tokenizer_name_or_path", args.text_tokenizer or _default_tokenizer_path(Path(args.aurora_root)))
    return config


def load_echo_pipeline(model_path: str | Path, echo_config: dict[str, Any], device: str) -> Chronos2EchoPipeline:
    model_path = Path(model_path) if not str(model_path).startswith("amazon/") else model_path
    is_peft = (
        isinstance(model_path, Path)
        and model_path.is_dir()
        and (model_path / "adapter_config.json").is_file()
        and not (model_path / "model.safetensors").is_file()
    )
    if not is_peft:
        return Chronos2EchoPipeline.from_pretrained(model_path, echo_config=Chronos2EchoConfig(**echo_config), device_map=device)

    from peft import PeftModel

    adapter_config = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = adapter_config.get("base_model_name_or_path", "amazon/chronos-2")
    base_config = Chronos2EchoConfig(**echo_config)
    base_config.residual_scale_init = 0.0
    base_config.guard_against_baseline = False
    base_pipeline = Chronos2EchoPipeline.from_pretrained(base_model, echo_config=base_config, device_map=device)
    model = PeftModel.from_pretrained(base_pipeline.model, str(model_path))
    return Chronos2EchoPipeline(model=model)


def _evaluate_echo_pipeline(
    model_name: str,
    tasks: list[dict[str, Any]],
    *,
    data_root: Path,
    pipeline_for_task,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        pipeline = pipeline_for_task(task)
        _, dataset = _echo_timemmd_dataset(
            pipeline,
            task,
            data_root=data_root,
            flag=task["split"],
            batch_size=batch_size or task["batch_size"],
            shuffle=False,
            repeat=False,
        )
        device = pipeline._unwrap_echo_model(pipeline.model).device
        loader = DataLoader(dataset, batch_size=None, pin_memory=device.type == "cuda")
        quantile_windows = []
        target_windows = []
        median_idx = min(range(len(pipeline.quantiles)), key=lambda idx: abs(float(pipeline.quantiles[idx]) - 0.5))
        pipeline.model.eval()
        with torch.no_grad():
            for batch in loader:
                target_idx_ranges = batch.pop("target_idx_ranges")
                future_target = batch["future_target"]
                model_inputs = {}
                for key, value in batch.items():
                    if key == "future_target":
                        continue
                    model_inputs[key] = value.to(device) if torch.is_tensor(value) else value
                output = pipeline.model(**model_inputs)
                quantile_preds = output.quantile_preds[..., : task["pred_len"]].cpu()
                for start, end in target_idx_ranges:
                    quantile_windows.append(quantile_preds[start:end].permute(2, 0, 1))
                    target_windows.append(future_target[start:end].permute(1, 0))
        quantiles = torch.stack(quantile_windows, dim=0).numpy()
        targets = torch.stack(target_windows, dim=0).numpy()
        rows.append(aurora_metric_row(model_name, task, quantiles[..., median_idx], targets))
    return rows


def evaluate_echo_zero_shot(
    model_name: str,
    tasks: list[dict[str, Any]],
    *,
    data_root: Path,
    model_path: str,
    device: str,
    echo_config: dict[str, Any],
    batch_size: int | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    pipeline = load_echo_pipeline(model_path, echo_config, device)
    return _evaluate_echo_pipeline(
        model_name,
        tasks,
        data_root=data_root,
        pipeline_for_task=lambda task: pipeline,
        batch_size=batch_size,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_checkpoint_dir(checkpoint_root: Path, task: dict[str, Any]) -> Path:
    return checkpoint_root / task["domain"] / f"H{task['seq_len']}_F{task['pred_len']}" / "finetuned-ckpt"


def _task_fingerprint(task: dict[str, Any], data_root: Path, echo_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": {key: task[key] for key in ["domain", "data_path", "seq_len", "pred_len", "features", "target"]},
        "data_sha256": _file_sha256(data_root / task["data_path"]),
        "echo_config": echo_config,
        "training": {
            "learning_rate": args.fewshot_lr,
            "num_steps": args.fewshot_steps,
            "warmup_steps": args.fewshot_warmup_steps,
            "batch_size": args.fewshot_batch_size,
            "lr_scheduler_type": args.fewshot_lr_scheduler,
            "finetune_mode": "lora",
            "flag": "fewshot",
            "few_shot_ratio": 0.1,
        },
    }


def fit_echo_timemmd(
    pipeline: Chronos2EchoPipeline,
    *,
    data_root: Path,
    task: dict[str, Any],
    batch_size: int,
    learning_rate: float,
    num_steps: int,
    warmup_steps: int,
    lr_scheduler_type: str,
    output_dir: Path,
    finetuned_ckpt_name: str = "finetuned-ckpt",
) -> None:
    from transformers.training_args import TrainingArguments

    from chronos_echo.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

    train_pipeline = pipeline.fit_echo(finetune_mode="lora")
    train_base_model = train_pipeline._unwrap_echo_model(train_pipeline.model)
    _, train_dataset = _echo_timemmd_dataset(
        train_pipeline,
        task,
        data_root=data_root,
        flag="fewshot",
        batch_size=batch_size,
        shuffle=True,
        repeat=True,
        few_shot_ratio=0.1,
    )
    _, eval_dataset = _echo_timemmd_dataset(
        train_pipeline,
        task,
        data_root=data_root,
        flag="val",
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        few_shot_ratio=0.1,
    )

    use_cpu = str(train_base_model.device) == "cpu"
    has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=100,
        disable_tqdm=False,
        report_to="none",
        max_steps=num_steps,
        dataloader_num_workers=0,
        tf32=has_sm80 and not use_cpu,
        bf16=has_sm80 and not use_cpu,
        save_only_model=True,
        prediction_loss_only=True,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        label_names=["future_target"],
        use_cpu=use_cpu,
        remove_unused_columns=False,
    )
    if not use_cpu:
        training_args._n_gpu = 1

    trainer = Chronos2Trainer(
        model=train_pipeline.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvaluateAndSaveFinalStepCallback()],
    )
    trainer.train()

    trained_model = trainer.model
    trained_base_model = train_pipeline._unwrap_echo_model(trained_model)
    trained_base_model.chronos_config.context_length = max(trained_base_model.chronos_config.context_length, task["seq_len"])
    trained_base_model.chronos_config.max_output_patches = max(
        trained_base_model.chronos_config.max_output_patches,
        (task["pred_len"] + train_pipeline.model_output_patch_size - 1) // train_pipeline.model_output_patch_size,
    )
    if hasattr(trained_base_model, "config"):
        trained_base_model.config.chronos_config = trained_base_model.chronos_config.__dict__
        trained_base_model.config.echo_config = trained_base_model.echo_config.__dict__
        trained_base_model.config.chronos_pipeline_class = "Chronos2EchoPipeline"
        trained_base_model.config.architectures = ["Chronos2EchoModel"]

    Chronos2EchoPipeline(model=trained_model).save_pretrained(output_dir / finetuned_ckpt_name)  # type: ignore[arg-type]


def train_echo_fewshot(task: dict[str, Any], *, data_root: Path, checkpoint_root: Path, args: argparse.Namespace, echo_config: dict[str, Any]) -> None:
    checkpoint_dir = _task_checkpoint_dir(checkpoint_root, task)
    output_dir = checkpoint_dir.parent
    fingerprint = _task_fingerprint(task, data_root, echo_config, args)
    fingerprint_path = output_dir / "fingerprint.json"
    if (
        not args.force_retrain
        and checkpoint_dir.exists()
        and fingerprint_path.exists()
        and json.loads(fingerprint_path.read_text(encoding="utf-8")) == fingerprint
    ):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Chronos2EchoPipeline.from_pretrained(
        args.echo_base_model,
        echo_config=Chronos2EchoConfig(**echo_config),
        device_map=args.device,
    )
    fit_echo_timemmd(
        pipeline,
        data_root=data_root,
        task=task,
        batch_size=args.fewshot_batch_size,
        learning_rate=args.fewshot_lr,
        num_steps=args.fewshot_steps,
        warmup_steps=args.fewshot_warmup_steps,
        lr_scheduler_type=args.fewshot_lr_scheduler,
        output_dir=output_dir,
        finetuned_ckpt_name="finetuned-ckpt",
    )
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")


def evaluate_echo_fewshot(
    model_name: str,
    tasks: list[dict[str, Any]],
    *,
    data_root: Path,
    checkpoint_root: Path,
    device: str,
    echo_config: dict[str, Any],
    batch_size: int | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    cache: dict[Path, Chronos2EchoPipeline] = {}

    def pipeline_for_task(task: dict[str, Any]) -> Chronos2EchoPipeline:
        checkpoint_dir = _task_checkpoint_dir(checkpoint_root, task)
        if checkpoint_dir not in cache:
            cache[checkpoint_dir] = load_echo_pipeline(checkpoint_dir, echo_config, device)
        return cache[checkpoint_dir]

    return _evaluate_echo_pipeline(
        model_name,
        tasks,
        data_root=data_root,
        pipeline_for_task=pipeline_for_task,
        batch_size=batch_size,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.write_text("", encoding="utf-8")


def comparison_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in metrics:
        reference_model = "aurora_few_shot" if row["model"] == "chronos2_echo_few_shot" else "aurora_zero_shot"
        reference = get_reference(row["domain"], int(row["pred_len"]), reference_model)
        if reference is None:
            continue
        rows.append(
            {
                "model": row["model"],
                "domain": row["domain"],
                "pred_len": row["pred_len"],
                "reference_model": reference_model,
                "model_mse": row["mse"],
                "reference_mse": reference["mse"],
                "delta_mse": row["mse"] - reference["mse"],
                "relative_mse_change": (row["mse"] - reference["mse"]) / reference["mse"] if reference["mse"] else float("nan"),
                "model_mae": row["mae"],
                "reference_mae": reference["mae"],
                "delta_mae": row["mae"] - reference["mae"],
                "relative_mae_change": (row["mae"] - reference["mae"]) / reference["mae"] if reference["mae"] else float("nan"),
            }
        )
    return rows


def domain_summary_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = pd.DataFrame(comparison_rows(metrics))
    if comparisons.empty:
        return []
    grouped = comparisons.groupby(["model", "domain", "reference_model"], as_index=False).agg(
        mean_model_mse=("model_mse", "mean"),
        mean_reference_mse=("reference_mse", "mean"),
        mean_delta_mse=("delta_mse", "mean"),
        mean_model_mae=("model_mae", "mean"),
        mean_reference_mae=("reference_mae", "mean"),
        mean_delta_mae=("delta_mae", "mean"),
    )
    return grouped.to_dict(orient="records")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()


def write_repro(path: Path, args: argparse.Namespace, repro: dict[str, Any]) -> None:
    lines = [
        "command: " + " ".join(sys.argv),
        f"cwd: {Path.cwd().resolve()}",
        f"git_commit: {git_commit()}",
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"device: {args.device}",
        f"data_root: {Path(args.data_root).resolve()}",
        f"manifest: {Path(args.manifest).resolve()}",
        f"aurora_reference: {SOURCE_URL}",
        f"seed: {repro['seed']}",
        f"torch_deterministic_algorithms: {repro['torch_deterministic_algorithms']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aurora-standard TimeMMD benchmark for Chronos-2 and Chronos-2-ECHO.")
    parser.add_argument("--data-root", required=True, help="Directory containing Aurora-compatible TimeMMD CSV files.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Task manifest CSV.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to TimeMMD/runs/<timestamp>.")
    parser.add_argument("--models", default="chronos2,echo_zero_shot,echo_few_shot")
    parser.add_argument("--chronos-model", default="amazon/chronos-2")
    parser.add_argument("--echo-base-model", default="amazon/chronos-2")
    parser.add_argument("--echo-config", default=None, help="JSON overrides merged into the Project_2 ECHO config.")
    parser.add_argument("--aurora-root", default=str((ROOT.parent / ".." / "Aurora").resolve()))
    parser.add_argument("--text-tokenizer", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--fewshot-lr", type=float, default=5e-6)
    parser.add_argument("--fewshot-steps", type=int, default=1000)
    parser.add_argument("--fewshot-warmup-steps", type=int, default=100)
    parser.add_argument("--fewshot-batch-size", type=int, default=8)
    parser.add_argument("--fewshot-lr-scheduler", default="cosine")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root)
    tasks = load_manifest(args.manifest)
    validation = validate_data_root(data_root, tasks)
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    echo_config = resolve_echo_config(args)

    if args.dry_run:
        print(f"Validated {len(tasks)} tasks from {args.manifest}")
        print(f"Models: {', '.join(sorted(selected))}")
        print(f"Data root: {data_root.resolve()}")
        return 0

    repro = set_reproducible(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    metrics: list[dict[str, Any]] = []
    if "chronos2" in selected:
        metrics.extend(
            evaluate_chronos2(
                "chronos2",
                tasks,
                data_root=data_root,
                model_path=args.chronos_model,
                device=args.device,
                batch_size=args.batch_size,
            )
        )
    if "echo_zero_shot" in selected:
        metrics.extend(
            evaluate_echo_zero_shot(
                "chronos2_echo_zero_shot",
                tasks,
                data_root=data_root,
                model_path=args.echo_base_model,
                device=args.device,
                echo_config=echo_config,
                batch_size=args.batch_size,
            )
        )
    if "echo_few_shot" in selected:
        checkpoint_root = Path(args.checkpoint_root)
        for task in tasks:
            train_echo_fewshot(task, data_root=data_root, checkpoint_root=checkpoint_root, args=args, echo_config=echo_config)
        metrics.extend(
            evaluate_echo_fewshot(
                "chronos2_echo_few_shot",
                tasks,
                data_root=data_root,
                checkpoint_root=checkpoint_root,
                device=args.device,
                echo_config=echo_config,
                batch_size=args.batch_size,
            )
        )

    comparisons = comparison_rows(metrics)
    summaries = domain_summary_rows(metrics)
    write_csv(output_dir / "metrics.csv", metrics)
    write_csv(output_dir / "comparison.csv", comparisons)
    write_csv(output_dir / "domain_summary.csv", summaries)
    summary = {
        "config": vars(args),
        "aurora_reference": SOURCE_URL,
        "reproducibility": repro,
        "n_tasks": len(tasks),
        "n_metrics": len(metrics),
        "n_comparisons": len(comparisons),
        "models": sorted(selected),
        "echo_config": echo_config,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_repro(output_dir / "repro.txt", args, repro)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
