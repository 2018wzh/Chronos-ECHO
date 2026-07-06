import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


class TestTokenizer:
    def __call__(self, text, *, padding, truncation, max_length, return_tensors):
        del text, padding, truncation
        assert return_tensors == "pt"
        input_ids = torch.ones((1, max_length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "token_type_ids": torch.zeros_like(input_ids),
        }


def _write_domain_csv(path: Path, n_rows: int = 80, *, blank_text: bool = False, bad_target: bool = False) -> None:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.astype(str),
            "OT": np.arange(n_rows, dtype=np.float32),
            "prior_history_avg": np.arange(n_rows, dtype=np.float32),
            "start_date": dates.astype(str),
            "end_date": dates.astype(str),
            "fact": [f"fact {idx}" for idx in range(n_rows)],
        }
    )
    if blank_text:
        frame.loc[0, "fact"] = ""
    if bad_target:
        frame.loc[0, "OT"] = np.inf
    frame.to_csv(path, index=False)


def test_timemmd_manifest_matches_aurora_protocol():
    from TimeMMD.run_benchmark import load_manifest

    tasks = load_manifest()
    assert len(tasks) == 36

    by_domain = {}
    for task in tasks:
        by_domain.setdefault(task["domain"], []).append(task)
        assert task["features"] == "S"
        assert task["target"] == "OT"
        assert task["split"] == "test"

    assert list(by_domain) == [
        "Agriculture",
        "Climate",
        "Economy",
        "Energy",
        "Environment",
        "Health",
        "Security",
        "Traffic",
        "SocialGood",
    ]
    assert [task["pred_len"] for task in by_domain["Energy"]] == [12, 24, 36, 48]
    assert [task["pred_len"] for task in by_domain["Environment"]] == [48, 96, 192, 336]
    assert [task["pred_len"] for task in by_domain["Agriculture"]] == [6, 8, 10, 12]
    assert {task["seq_len"] for task in by_domain["Security"]} == {220}


def test_aurora_metric_formula_matches_reference_metrics():
    from TimeMMD.run_benchmark import aurora_metric_row

    pred = np.array([[[1.0], [2.0]], [[2.0], [4.0]]])
    true = np.array([[[1.5], [1.0]], [[3.0], [3.0]]])

    row = aurora_metric_row("model", {"domain": "Tiny", "pred_len": 2, "seq_len": 4}, pred, true)

    assert row["n_windows"] == 2
    assert row["mae"] == pytest.approx(np.mean(np.abs(pred - true)))
    assert row["mse"] == pytest.approx(np.mean((pred - true) ** 2))
    assert row["rmse"] == pytest.approx(np.sqrt(row["mse"]))
    assert row["rse"] == pytest.approx(
        np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
    )
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)) + 1e-12
    assert row["corr"] == pytest.approx(float(np.asarray(0.01 * (u / d).mean(-1)).mean()))


def test_default_tokenizer_requires_aurora_timemmd_bert_config(tmp_path):
    from TimeMMD.run_benchmark import _default_tokenizer_path

    preferred = tmp_path / "TimeMMD" / "aurora" / "bert_config"
    with pytest.raises(FileNotFoundError, match="Aurora TimeMMD tokenizer config not found"):
        _default_tokenizer_path(tmp_path)

    preferred.mkdir(parents=True)

    assert _default_tokenizer_path(tmp_path) == str(preferred)


def test_validate_data_root_catches_bad_timemmd_inputs(tmp_path):
    from TimeMMD.run_benchmark import load_manifest
    from TimeMMD.validate_dataset import validate_data_root

    manifest = load_manifest()
    tiny_task = dict(manifest[0], seq_len=12, pred_len=4)

    _write_domain_csv(tmp_path / "Agriculture.csv")
    report = validate_data_root(tmp_path, [tiny_task])
    assert report["ok"] is True
    assert report["tasks"][0]["n_windows"] > 0

    (tmp_path / "Agriculture.csv").write_text("date,OT\n2024-01-01,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_data_root(tmp_path, [tiny_task])

    _write_domain_csv(tmp_path / "Agriculture.csv", bad_target=True)
    with pytest.raises(ValueError, match="non-finite values"):
        validate_data_root(tmp_path, [tiny_task])

    _write_domain_csv(tmp_path / "Agriculture.csv", blank_text=True)
    with pytest.raises(ValueError, match="empty fact"):
        validate_data_root(tmp_path, [tiny_task])

    _write_domain_csv(tmp_path / "Agriculture.csv", n_rows=16)
    with pytest.raises(ValueError, match="no test windows"):
        validate_data_root(tmp_path, [tiny_task])


def test_timemmd_dataset_and_batch_live_in_timemmd_package(tmp_path):
    from TimeMMD.dataset import TimeMMDBatchDataset, TimeMMDWindowDataset

    _write_domain_csv(tmp_path / "Agriculture.csv")
    default_ratio = TimeMMDWindowDataset(
        root_path=tmp_path,
        data_path="Agriculture.csv",
        flag="fewshot",
        seq_len=12,
        pred_len=4,
        target="OT",
        features="S",
        tokenizer=TestTokenizer(),
        max_text_length=16,
    )
    larger_ratio = TimeMMDWindowDataset(
        root_path=tmp_path,
        data_path="Agriculture.csv",
        flag="fewshot",
        seq_len=12,
        pred_len=4,
        target="OT",
        features="S",
        tokenizer=TestTokenizer(),
        max_text_length=16,
        few_shot_ratio=0.2,
    )
    batch = next(
        iter(
            TimeMMDBatchDataset(
                default_ratio,
                batch_size=2,
                output_patch_size=16,
                shuffle=False,
                repeat=False,
            )
        )
    )

    assert len(default_ratio) == 3
    assert len(larger_ratio) == 9
    assert batch["context"].shape == (2, 12)
    assert batch["future_target"].shape == (2, 4)
    assert batch["text_input_ids"].shape == (2, 16)


def test_run_benchmark_writes_comparable_outputs(tmp_path, monkeypatch):
    from TimeMMD import run_benchmark

    _write_domain_csv(tmp_path / "Agriculture.csv")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "domain": "Agriculture",
                "data_path": "Agriculture.csv",
                "seq_len": 12,
                "pred_len": 6,
                "features": "S",
                "target": "OT",
                "text_column": "fact",
                "split": "test",
                "inference_token_len": 48,
                "batch_size": 2,
            }
        ]
    ).to_csv(manifest_path, index=False)

    def fake_eval(model_name, tasks, **kwargs):
        del kwargs
        return [
            {
                "model": model_name,
                "domain": task["domain"],
                "pred_len": task["pred_len"],
                "seq_len": task["seq_len"],
                "mse": 0.5 if model_name == "chronos2" else 0.4,
                "mae": 0.6 if model_name == "chronos2" else 0.5,
                "rmse": 0.7,
                "mape": 0.0,
                "mspe": 0.0,
                "rse": 0.0,
                "corr": 0.0,
                "n_windows": 1,
            }
            for task in tasks
        ]

    monkeypatch.setattr(run_benchmark, "evaluate_chronos2", fake_eval)
    monkeypatch.setattr(run_benchmark, "evaluate_echo_zero_shot", fake_eval)
    monkeypatch.setattr(run_benchmark, "train_echo_fewshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_benchmark, "evaluate_echo_fewshot", fake_eval)

    run_dir = tmp_path / "run"
    exit_code = run_benchmark.main(
        [
            "--data-root",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(run_dir),
            "--models",
            "chronos2,echo_zero_shot,echo_few_shot",
            "--text-tokenizer",
            "dummy-tokenizer",
        ]
    )

    assert exit_code == 0
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "domain_summary.csv").exists()
    assert json.loads((run_dir / "input_validation.json").read_text(encoding="utf-8"))["ok"] is True
    comparison = pd.read_csv(run_dir / "comparison.csv")
    assert set(comparison["reference_model"]) == {"aurora_zero_shot", "aurora_few_shot"}


def test_train_echo_fewshot_uses_project2_training_defaults(tmp_path, monkeypatch):
    from TimeMMD import run_benchmark

    _write_domain_csv(tmp_path / "Agriculture.csv")
    task = dict(run_benchmark.load_manifest()[0], seq_len=12, pred_len=6)
    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["from_pretrained"] = {"args": args, "kwargs": kwargs}
            return cls()

    def fake_fit_echo_timemmd(pipeline, **kwargs):
        captured["fit_echo_timemmd"] = {"pipeline": pipeline, "kwargs": kwargs}

    monkeypatch.setattr(run_benchmark, "Chronos2EchoPipeline", FakePipeline)
    monkeypatch.setattr(run_benchmark, "fit_echo_timemmd", fake_fit_echo_timemmd)
    args = SimpleNamespace(
        force_retrain=False,
        echo_base_model="amazon/chronos-2",
        device="cpu",
        fewshot_lr=5e-6,
        fewshot_steps=1000,
        fewshot_warmup_steps=100,
        fewshot_batch_size=8,
        fewshot_lr_scheduler="cosine",
    )

    run_benchmark.train_echo_fewshot(
        task,
        data_root=tmp_path,
        checkpoint_root=tmp_path / "checkpoints",
        args=args,
        echo_config=run_benchmark.PROJECT2_ECHO_CONFIG,
    )

    fit_kwargs = captured["fit_echo_timemmd"]["kwargs"]
    assert fit_kwargs["learning_rate"] == 5e-6
    assert fit_kwargs["num_steps"] == 1000
    assert fit_kwargs["warmup_steps"] == 100
    assert fit_kwargs["batch_size"] == 8
    assert fit_kwargs["lr_scheduler_type"] == "cosine"
