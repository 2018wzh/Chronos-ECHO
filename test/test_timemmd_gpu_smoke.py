import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "scripts" / "timemmd"
sys.path.insert(0, str(BENCHMARK_DIR))

import run_benchmark  # noqa: E402


def _write_smoke_csv(path: Path, n_rows: int = 80) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["date", "OT", "prior_history_avg", "start_date", "end_date", "fact"],
        )
        writer.writeheader()
        for idx in range(n_rows):
            date = f"2024-01-{idx % 28 + 1:02d}"
            writer.writerow(
                {
                    "date": date,
                    "OT": float(idx),
                    "prior_history_avg": float(idx - 1),
                    "start_date": date,
                    "end_date": date,
                    "fact": f"smoke event {idx}",
                }
            )


def test_timemmd_echo_zero_shot_smoke_runs_real_gpu_evaluation(tmp_path, capsys):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU TimeMMD smoke test")

    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_smoke_csv(data_root / "Smoke.csv")

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "domain": "Smoke",
                "data_path": "Smoke.csv",
                "seq_len": 16,
                "pred_len": 4,
                "features": "S",
                "target": "OT",
                "text_column": "fact",
                "split": "test",
                "inference_token_len": 16,
                "batch_size": 2,
            }
        ]
    ).to_csv(manifest, index=False)

    output_dir = tmp_path / "run"
    echo_config = {
        "num_attention_heads": 2,
        "num_echo_layers": 1,
        "num_text_tokens": 2,
        "num_vision_tokens": 2,
        "vision_model_name_or_path": None,
        "vision_patch_size": 8,
        "vision_image_size": 16,
        "max_text_length": 16,
        "use_pseudo_image": True,
    }

    torch.cuda.reset_peak_memory_stats()
    exit_code = run_benchmark.main(
        [
            "--data-root",
            str(data_root),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--models",
            "echo_zero_shot",
            "--echo-base-model",
            str(REPO_ROOT / "test" / "dummy-chronos2-model"),
            "--text-tokenizer",
            str(BENCHMARK_DIR / "aurora" / "bert_config"),
            "--echo-config",
            json.dumps(echo_config),
            "--device",
            "cuda",
            "--batch-size",
            "2",
        ]
    )

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err
    assert exit_code == 0
    assert "MISSING" not in combined_output or "echo." not in combined_output
    assert torch.cuda.max_memory_allocated() > 0
    assert "device: cuda" in (output_dir / "repro.txt").read_text(encoding="utf-8")

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert metrics["model"].tolist() == ["chronos2_echo_zero_shot"]
    assert int(metrics.loc[0, "n_windows"]) > 0
