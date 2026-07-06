import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "scripts" / "timemmd"
sys.path.insert(0, str(BENCHMARK_DIR))


def test_timemmd_benchmark_is_not_packaged_as_import_package():
    import tomllib

    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert not (BENCHMARK_DIR / "__init__.py").exists()
    assert "TimeMMD" not in wheel.get("packages", [])
    assert "scripts/timemmd" not in wheel.get("packages", [])
    assert "artifacts" not in wheel


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


def _write_domain_csv(
    path: Path,
    n_rows: int = 80,
    *,
    blank_text: bool = False,
    missing_text: bool = False,
    bad_target: bool = False,
) -> None:
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
        frame.loc[0, "fact"] = "   "
    if missing_text:
        frame.loc[0, "fact"] = np.nan
    if bad_target:
        frame.loc[0, "OT"] = np.inf
    frame.to_csv(path, index=False)


def test_timemmd_manifest_matches_aurora_protocol():
    from run_benchmark import load_manifest

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
    from run_benchmark import aurora_metric_row

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


def test_default_tokenizer_uses_bundled_aurora_timemmd_bert_config():
    from run_benchmark import DEFAULT_TOKENIZER_PATH, _default_tokenizer_path

    assert _default_tokenizer_path() == str(DEFAULT_TOKENIZER_PATH)
    assert (DEFAULT_TOKENIZER_PATH / "vocab.txt").is_file()


def test_validate_data_root_catches_bad_timemmd_inputs(tmp_path):
    from run_benchmark import load_manifest
    from validate_dataset import validate_data_root

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

    _write_domain_csv(tmp_path / "Agriculture.csv", missing_text=True)
    report = validate_data_root(tmp_path, [tiny_task])
    assert report["ok"] is True

    _write_domain_csv(tmp_path / "Agriculture.csv", n_rows=16)
    with pytest.raises(ValueError, match="no test windows"):
        validate_data_root(tmp_path, [tiny_task])


def test_timemmd_dataset_and_batch_live_in_script_tree(tmp_path):
    from dataset import TimeMMDBatchDataset, TimeMMDWindowDataset

    _write_domain_csv(tmp_path / "Agriculture.csv")
    _write_domain_csv(tmp_path / "MissingFact.csv", missing_text=True)
    missing_fact = TimeMMDWindowDataset(
        root_path=tmp_path,
        data_path="MissingFact.csv",
        flag="train",
        seq_len=12,
        pred_len=4,
        target="OT",
        features="S",
        tokenizer=TestTokenizer(),
        max_text_length=16,
    )
    assert missing_fact.text[0] == "No information available"

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
    import run_benchmark

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


def test_chronos2_zero_shot_passes_three_dimensional_context(tmp_path, monkeypatch):
    import chronos
    from run_benchmark import evaluate_chronos2

    _write_domain_csv(tmp_path / "Agriculture.csv")
    task = {
        "domain": "Agriculture",
        "data_path": "Agriculture.csv",
        "seq_len": 12,
        "pred_len": 4,
        "features": "S",
        "target": "OT",
        "text_column": "fact",
        "split": "test",
        "batch_size": 3,
    }
    shapes = []

    class FakeChronos2Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def predict_quantiles(self, context, *, prediction_length, quantile_levels, limit_prediction_length):
            del quantile_levels, limit_prediction_length
            shapes.append(tuple(context.shape))
            batch_size = context.shape[0]
            means = [torch.zeros((1, prediction_length)) for _ in range(batch_size)]
            quantiles = [mean.unsqueeze(-1) for mean in means]
            return quantiles, means

    monkeypatch.setattr(chronos, "Chronos2Pipeline", FakeChronos2Pipeline)

    rows = evaluate_chronos2("chronos2", [task], data_root=tmp_path, model_path="dummy", device="cpu")

    assert rows[0]["n_windows"] > 0
    assert shapes
    assert all(shape[1:] == (1, 12) for shape in shapes)


def test_echo_zero_shot_does_not_pass_timemmd_prior_as_risk_feature(monkeypatch):
    import run_benchmark

    captured_inputs = {}

    class FakeEchoConfig:
        max_text_length = 16

    class FakeEchoModel:
        device = torch.device("cpu")
        echo_config = FakeEchoConfig()

        def eval(self):
            return None

        def __call__(self, **kwargs):
            captured_inputs.update(kwargs)
            return SimpleNamespace(quantile_preds=torch.zeros((1, 1, 4)))

    class FakePipeline:
        model = FakeEchoModel()
        model_output_patch_size = 16
        quantiles = [0.5]

        def _unwrap_echo_model(self, model):
            return model

    batch = {
        "context": torch.zeros((1, 12)),
        "future_target": torch.zeros((1, 4)),
        "future_covariates": torch.full((1, 4), float("nan")),
        "group_ids": torch.zeros((1,), dtype=torch.long),
        "num_output_patches": 1,
        "target_idx_ranges": [(0, 1)],
        "text_input_ids": torch.ones((1, 16), dtype=torch.long),
        "text_attention_mask": torch.ones((1, 16), dtype=torch.long),
        "text_token_type_ids": torch.zeros((1, 16), dtype=torch.long),
        "risk_features": torch.ones((1, 4, 1)),
    }
    task = {
        "domain": "Agriculture",
        "data_path": "Agriculture.csv",
        "seq_len": 12,
        "pred_len": 4,
        "features": "S",
        "target": "OT",
        "text_column": "fact",
        "split": "test",
        "batch_size": 1,
    }

    monkeypatch.setattr(run_benchmark, "_echo_timemmd_dataset", lambda *args, **kwargs: (None, [batch]))
    monkeypatch.setattr(run_benchmark, "DataLoader", lambda dataset, **kwargs: iter(dataset))

    rows = run_benchmark._evaluate_echo_pipeline(
        "chronos2_echo_zero_shot",
        [task],
        data_root=Path("."),
        pipeline_for_task=lambda _: FakePipeline(),
    )

    assert rows[0]["n_windows"] == 1
    assert "risk_features" not in captured_inputs


def test_fewshot_split_validation_catches_untrainable_aurora_tasks(tmp_path):
    from run_benchmark import validate_fewshot_splits

    _write_domain_csv(tmp_path / "Energy.csv", n_rows=100)
    task = {
        "domain": "Energy",
        "data_path": "Energy.csv",
        "seq_len": 68,
        "pred_len": 4,
        "features": "S",
        "target": "OT",
        "text_column": "fact",
    }

    with pytest.raises(ValueError, match="fewshot=0"):
        validate_fewshot_splits(tmp_path, [task])


def test_train_echo_fewshot_uses_project2_training_defaults(tmp_path, monkeypatch):
    import run_benchmark

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
