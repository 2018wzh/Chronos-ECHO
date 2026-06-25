# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-dataset support for Chronos-2-ECHO, inspired by Aurora's ConcatDataset approach."""

from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import ConcatDataset, IterableDataset

from .timemmd import TimeMMDBatchDataset, TimeMMDWindowDataset


class MultiTimeMMDDataset(IterableDataset):
    """Iterable dataset that concatenates multiple TimeMMD CSV files into a single training stream.

    This mirrors Aurora's ``ConcatDataset`` strategy for pretraining on diverse,
    cross-domain multimodal time series corpora. Each source is wrapped in its own
    ``TimeMMDWindowDataset``, unified via ``ConcatDataset``, and then fed into a
    single ``TimeMMDBatchDataset`` for batching.

    Parameters
    ----------
    sources
        List of ``(root_path, data_path)`` tuples, each identifying a TimeMMD CSV file.
        ``root_path`` is the directory containing the CSV and optional image directory.
        ``data_path`` is the CSV filename relative to ``root_path``.
    flag
        ``"train"``, ``"val"``, ``"test"``, or ``"fewshot"``.
    seq_len
        Length of the historical context window.
    pred_len
        Length of the prediction horizon.
    target
        Name of the target column in each CSV.
    features
        ``"S"``, ``"M"``, or ``"MS"`` — feature mode for each dataset.
    batch_size
        Number of sliding windows per batch.
    output_patch_size
        Patch size used by the Chronos-2 model.
    shuffle
        Shuffle window indices before batching.
    repeat
        Repeat the dataset indefinitely (set ``True`` for training).
    tokenizer
        HF tokenizer for text encoding.
    max_text_length
        Maximum tokenised text length.
    image_column
        Column name for image paths (auto-detected if ``None``).
    image_root_path
        Base directory for relative image paths.
    image_size
        Target size (square) for loaded images.
    """

    def __init__(
        self,
        sources: list[tuple[str | Path, str]],
        *,
        flag: str = "train",
        seq_len: int,
        pred_len: int,
        target: str = "OT",
        features: str = "S",
        batch_size: int = 256,
        output_patch_size: int,
        shuffle: bool,
        repeat: bool,
        tokenizer: Any,
        max_text_length: int = 500,
        image_column: str | None = None,
        image_root_path: str | Path | None = None,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if len(sources) == 0:
            raise ValueError("sources must contain at least one (root_path, data_path) tuple")

        window_datasets = []
        for root_path, data_path in sources:
            ds = TimeMMDWindowDataset(
                root_path=root_path,
                data_path=data_path,
                flag=flag,
                seq_len=seq_len,
                pred_len=pred_len,
                target=target,
                features=features,
                tokenizer=tokenizer,
                max_text_length=max_text_length,
                image_column=image_column,
                image_root_path=image_root_path,
                image_size=image_size,
            )
            window_datasets.append(ds)

        self.concat_dataset = ConcatDataset(window_datasets)
        self.batch_dataset = TimeMMDBatchDataset(
            self.concat_dataset,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            shuffle=shuffle,
            repeat=repeat,
        )

    @property
    def batch_size(self) -> int:
        return self.batch_dataset.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.batch_dataset)
