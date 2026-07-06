from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "OT", "prior_history_avg", "start_date", "end_date", "fact"}


def _test_window_count(n_rows: int, seq_len: int, pred_len: int) -> int:
    num_test = int(n_rows * 0.2)
    test_start = max(0, n_rows - num_test - seq_len)
    return max(0, n_rows - test_start - seq_len - pred_len + 1)


def validate_data_root(data_root: str | Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    data_root = Path(data_root)
    results = []
    for task in tasks:
        path = data_root / task["data_path"]
        if not path.exists():
            raise FileNotFoundError(f"TimeMMD file not found: {path}")

        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

        for col in ["date", "start_date", "end_date"]:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.isna().any():
                raise ValueError(f"{path.name} has unparsable {col} values")

        target = pd.to_numeric(df["OT"], errors="coerce")
        if target.isna().any():
            raise ValueError(f"{path.name} has non-numeric OT values")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} has non-finite values in OT")

        fact = df["fact"]
        if fact.isna().any() or fact.astype(str).str.strip().eq("").any():
            raise ValueError(f"{path.name} contains missing or empty fact text")

        n_windows = _test_window_count(len(df), int(task["seq_len"]), int(task["pred_len"]))
        if n_windows == 0:
            raise ValueError(
                f"{path.name} has no test windows for seq_len={task['seq_len']} pred_len={task['pred_len']}"
            )

        results.append(
            {
                "domain": task["domain"],
                "pred_len": int(task["pred_len"]),
                "data_path": task["data_path"],
                "n_rows": int(len(df)),
                "n_windows": int(n_windows),
            }
        )

    return {"ok": True, "data_root": str(data_root.resolve()), "tasks": results}
