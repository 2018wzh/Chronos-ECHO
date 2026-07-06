from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

AURORA_MISSING_TEXT = "No information available"


def create_timemmd_tokenizer(tokenizer_name_or_path: str | None) -> Any:
    if tokenizer_name_or_path is None:
        raise ValueError("Set text_tokenizer_name_or_path or pass an explicit tokenizer for TimeMMD text input.")

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name_or_path)


class TimeMMDWindowDataset(Dataset):
    required_columns = {"date", "prior_history_avg", "start_date", "end_date"}

    def __init__(
        self,
        *,
        root_path: str | Path,
        data_path: str,
        flag: str = "train",
        seq_len: int,
        pred_len: int,
        target: str = "OT",
        features: str = "S",
        tokenizer: Any | None = None,
        max_text_length: int = 500,
        text_column: str = "fact",
        image_column: str | None = None,
        image_root_path: str | Path | None = None,
        image_size: int = 64,
        scale: bool = True,
        few_shot_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        if flag not in {"train", "val", "test", "fewshot"}:
            raise ValueError("flag must be one of 'train', 'val', 'test', or 'fewshot'")
        if features not in {"S", "M", "MS"}:
            raise ValueError("features must be one of 'S', 'M', or 'MS'")
        if tokenizer is None:
            raise ValueError("TimeMMDWindowDataset requires an explicit tokenizer")

        self.root_path = Path(root_path)
        self.data_path = data_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target = target
        self.features = features
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.text_column = text_column
        self.image_column = image_column
        if image_root_path is None:
            self.image_root_path = self.root_path
        else:
            image_root_path = Path(image_root_path)
            self.image_root_path = image_root_path if image_root_path.is_absolute() else self.root_path / image_root_path
        self.image_size = image_size
        self.scale = scale
        self.few_shot_ratio = few_shot_ratio
        self._read_data()

    def _read_data(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.root_path / self.data_path)
        missing = self.required_columns - set(df.columns)
        if missing:
            raise ValueError(f"TimeMMD CSV is missing required columns: {sorted(missing)}")
        if self.target not in df.columns:
            raise ValueError(f"Target column {self.target!r} not found in TimeMMD CSV")
        if self.text_column not in df.columns:
            raise ValueError(f"Text column {self.text_column!r} not found in TimeMMD CSV")
        if self.image_column is not None and self.image_column not in df.columns:
            raise ValueError(f"Image column {self.image_column!r} not found in TimeMMD CSV")

        metadata_columns = self.required_columns | {"date", self.text_column}
        if self.image_column is not None:
            metadata_columns.add(self.image_column)
        numeric_columns = []
        for col in [col for col in df.columns if col not in metadata_columns]:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                df[col] = series
                numeric_columns.append(col)
        if self.target not in numeric_columns:
            raise ValueError(f"Target column {self.target!r} must be numeric")

        if self.features == "S":
            self.target_columns = [self.target]
            self.covariate_columns: list[str] = []
        elif self.features == "M":
            self.target_columns = numeric_columns
            self.covariate_columns = []
        else:
            self.target_columns = [self.target]
            self.covariate_columns = [col for col in numeric_columns if col != self.target and col != "prior_history_avg"]

        value_columns = self.target_columns + self.covariate_columns
        values = df[value_columns].to_numpy(dtype=np.float32)
        prior = pd.to_numeric(df["prior_history_avg"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        text_series = df[self.text_column].fillna(AURORA_MISSING_TEXT).astype(str)
        if text_series.str.strip().eq("").any():
            raise ValueError(
                f"TimeMMD CSV contains empty {self.text_column} values; text must be provided explicitly"
            )
        text = text_series.to_numpy()
        image_paths = df[self.image_column].fillna("").astype(str).to_numpy() if self.image_column else None

        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        num_val = len(df) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df)]

        if self.flag == "train":
            border1, border2 = border1s[0], border2s[0]
        elif self.flag == "val":
            border1, border2 = border1s[1], border2s[1]
        elif self.flag == "test":
            border1, border2 = border1s[2], border2s[2]
        else:
            border1 = int((1 - self.few_shot_ratio) * num_train) - self.seq_len
            border2 = num_train

        train_values = values[border1s[0] : border2s[0]]
        if self.scale:
            self.value_loc = np.nanmean(train_values, axis=0, keepdims=True)
            self.value_scale = np.nanstd(train_values, axis=0, keepdims=True)
            self.value_scale = np.where(self.value_scale == 0, 1.0, self.value_scale)
            values = (values - self.value_loc) / self.value_scale
            target_idx = value_columns.index(self.target)
            self.prior_loc = float(self.value_loc[0, target_idx])
            self.prior_scale = float(self.value_scale[0, target_idx])
            prior = (prior - self.prior_loc) / self.prior_scale
        else:
            self.value_loc = np.zeros((1, len(value_columns)), dtype=np.float32)
            self.value_scale = np.ones((1, len(value_columns)), dtype=np.float32)
            self.prior_loc = 0.0
            self.prior_scale = 1.0

        self.values = values[border1:border2]
        self.prior = prior[border1:border2]
        self.text = text[border1:border2]
        self.image_paths = image_paths[border1:border2] if image_paths is not None else None
        self.border1 = border1
        self.border2 = border2
        self.n_targets = len(self.target_columns)
        self.n_variates = len(value_columns)
        self.length = max(0, len(self.values) - self.seq_len - self.pred_len + 1)

    def __len__(self) -> int:
        return self.length

    def _load_image(self, raw_path: str) -> torch.Tensor:
        raw_path = raw_path.strip()
        if not raw_path:
            raise ValueError("TimeMMD image path is empty; image input must be provided explicitly")

        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = self.image_root_path / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"TimeMMD image file not found: {image_path}")

        from PIL import Image

        with Image.open(image_path) as image:
            image = image.convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        context = self.values[index : index + self.seq_len].T
        future = self.values[index + self.seq_len : index + self.seq_len + self.pred_len].T
        future_target = future.copy()
        future_covariates = np.full_like(future, fill_value=np.nan)
        if self.n_targets < self.n_variates:
            future_target[self.n_targets :] = np.nan

        tokenized = self.tokenizer(
            self.text[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        risk = self.prior[index + self.seq_len : index + self.seq_len + self.pred_len].reshape(self.pred_len, 1)
        item = {
            "context": torch.tensor(context, dtype=torch.float32),
            "future_target": torch.tensor(future_target, dtype=torch.float32),
            "future_covariates": torch.tensor(future_covariates, dtype=torch.float32),
            "risk_features": torch.tensor(risk, dtype=torch.float32),
            "text_input_ids": tokenized["input_ids"].squeeze(0).to(torch.long),
            "text_attention_mask": tokenized["attention_mask"].squeeze(0).to(torch.long),
            "text_token_type_ids": tokenized.get("token_type_ids", torch.zeros_like(tokenized["input_ids"]))
            .squeeze(0)
            .to(torch.long),
            "n_targets": self.n_targets,
        }
        if self.image_paths is not None:
            item["vision_values"] = self._load_image(self.image_paths[index])
        return item

    def inverse_transform(self, values: np.ndarray, column_indices: list[int] | None = None) -> np.ndarray:
        loc = self.value_loc
        scale = self.value_scale
        if column_indices is not None:
            loc = loc[:, column_indices]
            scale = scale[:, column_indices]
        return values * scale + loc


def build_timemmd_batch(items: list[dict[str, torch.Tensor | int]], output_patch_size: int) -> dict[str, Any]:
    batch_context = []
    batch_future_target = []
    batch_future_covariates = []
    batch_group_ids = []
    batch_text_input_ids = []
    batch_text_attention_mask = []
    batch_text_token_type_ids = []
    batch_risk_features = []
    batch_vision_values = []
    target_idx_ranges: list[tuple[int, int]] = []

    target_start_idx = 0
    for group_id, item in enumerate(items):
        context = item["context"]
        future_target = item["future_target"]
        future_covariates = item["future_covariates"]
        risk_features = item["risk_features"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(future_target, torch.Tensor)
        assert isinstance(future_covariates, torch.Tensor)
        assert isinstance(risk_features, torch.Tensor)
        n_variates = context.shape[0]
        n_targets = int(item["n_targets"])

        batch_context.append(context)
        batch_future_target.append(future_target)
        batch_future_covariates.append(future_covariates)
        batch_group_ids.append(torch.full((n_variates,), fill_value=group_id, dtype=torch.long))
        target_idx_ranges.append((target_start_idx, target_start_idx + n_targets))
        target_start_idx += n_variates

        for key, target_list in [
            ("text_input_ids", batch_text_input_ids),
            ("text_attention_mask", batch_text_attention_mask),
            ("text_token_type_ids", batch_text_token_type_ids),
        ]:
            tensor = item[key]
            assert isinstance(tensor, torch.Tensor)
            target_list.append(tensor.unsqueeze(0).repeat(n_variates, 1))
        batch_risk_features.append(risk_features.unsqueeze(0).repeat(n_variates, 1, 1))
        vision_values = item.get("vision_values")
        if vision_values is not None:
            assert isinstance(vision_values, torch.Tensor)
            batch_vision_values.append(vision_values.unsqueeze(0).repeat(n_variates, 1, 1, 1))

    prediction_length = batch_future_target[0].shape[-1]
    batch = {
        "context": torch.cat(batch_context, dim=0),
        "future_target": torch.cat(batch_future_target, dim=0),
        "future_covariates": torch.cat(batch_future_covariates, dim=0),
        "group_ids": torch.cat(batch_group_ids, dim=0),
        "num_output_patches": math.ceil(prediction_length / output_patch_size),
        "target_idx_ranges": target_idx_ranges,
        "text_input_ids": torch.cat(batch_text_input_ids, dim=0),
        "text_attention_mask": torch.cat(batch_text_attention_mask, dim=0),
        "text_token_type_ids": torch.cat(batch_text_token_type_ids, dim=0),
        "risk_features": torch.cat(batch_risk_features, dim=0),
    }
    if batch_vision_values:
        batch["vision_values"] = torch.cat(batch_vision_values, dim=0)
    return batch


class TimeMMDBatchDataset(IterableDataset):
    def __init__(
        self,
        window_dataset: TimeMMDWindowDataset,
        *,
        batch_size: int,
        output_patch_size: int,
        shuffle: bool,
        repeat: bool,
    ) -> None:
        super().__init__()
        self.window_dataset = window_dataset
        self.batch_size = batch_size
        self.output_patch_size = output_patch_size
        self.shuffle = shuffle
        self.repeat = repeat
        if len(window_dataset) == 0:
            raise ValueError(
                "TimeMMD split contains no sliding windows; reduce seq_len/pred_len or use a longer CSV file"
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        indices = np.arange(len(self.window_dataset))
        while True:
            if self.shuffle:
                np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                if len(batch_indices) == 0:
                    continue
                yield build_timemmd_batch(
                    [self.window_dataset[int(idx)] for idx in batch_indices],
                    self.output_patch_size,
                )
            if not self.repeat:
                break
