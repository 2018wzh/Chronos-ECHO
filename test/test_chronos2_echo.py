import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from chronos_echo import Chronos2EchoConfig, Chronos2EchoPipeline
from chronos_echo.timemmd import TimeMMDWindowDataset, build_timemmd_batch


DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"

with open(DUMMY_MODEL_PATH / "config.json") as fp:
    DUMMY_CONFIG = json.load(fp)
DEFAULT_MODEL_NUM_QUANTILES = len(DUMMY_CONFIG["chronos_config"]["quantiles"])


def _echo_config() -> Chronos2EchoConfig:
    return Chronos2EchoConfig(
        num_attention_heads=2,
        num_echo_layers=1,
        num_text_tokens=2,
        num_vision_tokens=2,
        vision_patch_size=8,
        vision_image_size=16,
        max_text_length=16,
    )


class TestTokenizer:
    def __call__(self, text, *, padding, truncation, max_length, return_tensors):
        del padding, truncation
        assert return_tensors == "pt"
        pieces = str(text).split()
        ids = [101] + [104 + (sum(piece.encode("utf-8")) % 100) for piece in pieces] + [102]
        ids = ids[:max_length]
        ids = ids + [0] * (max_length - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = input_ids.ne(0).to(torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
        }


@pytest.fixture
def echo_pipeline() -> Chronos2EchoPipeline:
    return Chronos2EchoPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu", echo_config=_echo_config())


def _write_timemmd_csv(path: Path, n_rows: int = 80) -> None:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    values = np.sin(np.arange(n_rows) / 5.0).astype(np.float32)
    frame = pd.DataFrame(
        {
            "date": dates.astype(str),
            "feat": values * 0.5,
            "OT": values,
            "prior_history_avg": pd.Series(values).rolling(3, min_periods=1).mean(),
            "start_date": dates.astype(str),
            "end_date": dates.astype(str),
            "fact": [f"event text {i}" for i in range(n_rows)],
        }
    )
    frame.to_csv(path, index=False)


def _write_timemmd_csv_with_images(path: Path, n_rows: int = 80) -> None:
    from PIL import Image

    _write_timemmd_csv(path, n_rows=n_rows)
    frame = pd.read_csv(path)
    image_dir = path.parent / "images"
    image_dir.mkdir()
    image_paths = []
    for idx in range(n_rows):
        image = np.full((16, 16), fill_value=idx % 255, dtype=np.uint8)
        image_path = image_dir / f"{idx:04d}.png"
        Image.fromarray(image).save(image_path)
        image_paths.append(Path("images") / image_path.name)
    frame["image_path"] = [str(image_path) for image_path in image_paths]
    frame.to_csv(path, index=False)


def test_timemmd_dataset_parses_required_format(tmp_path):
    csv_path = tmp_path / "Tiny.csv"
    _write_timemmd_csv(csv_path)

    dataset = TimeMMDWindowDataset(
        root_path=tmp_path,
        data_path=csv_path.name,
        flag="train",
        seq_len=12,
        pred_len=4,
        target="OT",
        features="S",
        tokenizer=TestTokenizer(),
        max_text_length=16,
    )
    item = dataset[0]

    assert item["context"].shape == (1, 12)
    assert item["future_target"].shape == (1, 4)
    assert item["risk_features"].shape == (4, 1)
    assert item["text_input_ids"].shape == (16,)

    ms_dataset = TimeMMDWindowDataset(
        root_path=tmp_path,
        data_path=csv_path.name,
        flag="train",
        seq_len=12,
        pred_len=4,
        target="OT",
        features="MS",
        tokenizer=TestTokenizer(),
        max_text_length=16,
    )
    ms_item = ms_dataset[0]
    assert ms_item["context"].shape == (2, 12)
    assert not torch.isnan(ms_item["future_target"][0]).any()
    assert torch.isnan(ms_item["future_target"][1]).all()
    assert torch.isnan(ms_item["future_covariates"]).all()


def test_timemmd_dataset_requires_real_text(tmp_path):
    csv_path = tmp_path / "TinyMissingText.csv"
    _write_timemmd_csv(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[0, "fact"] = np.nan
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing or empty fact"):
        TimeMMDWindowDataset(
            root_path=tmp_path,
            data_path=csv_path.name,
            flag="train",
            seq_len=12,
            pred_len=4,
            target="OT",
            features="S",
            tokenizer=TestTokenizer(),
            max_text_length=16,
        )


def test_timemmd_dataset_loads_real_image_paths(tmp_path):
    csv_path = tmp_path / "TinyImages.csv"
    _write_timemmd_csv_with_images(csv_path)

    dataset = TimeMMDWindowDataset(
        root_path=tmp_path,
        data_path=csv_path.name,
        flag="train",
        seq_len=12,
        pred_len=4,
        target="OT",
        features="S",
        tokenizer=TestTokenizer(),
        image_column="image_path",
        image_size=16,
        max_text_length=16,
    )
    item = dataset[3]
    assert item["vision_values"].shape == (1, 16, 16)
    assert item["vision_values"].max() > 0

    batch = build_timemmd_batch([item, dataset[4]], output_patch_size=16)
    assert batch["vision_values"].shape == (2, 1, 16, 16)


def test_echo_forward_shapes_and_quantile_monotonicity(echo_pipeline):
    context = torch.rand(3, 16)
    text_input_ids = torch.randint(104, 200, (3, 16))
    text_attention_mask = torch.ones_like(text_input_ids)

    output = echo_pipeline.model(
        context=context,
        num_output_patches=1,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        vision_values=torch.rand(3, 1, 16, 16),
    )

    assert output.quantile_preds.shape == (3, DEFAULT_MODEL_NUM_QUANTILES, 16)
    assert not torch.isnan(output.quantile_preds).any()
    assert torch.all(output.quantile_preds[:, 1:] >= output.quantile_preds[:, :-1])


def test_echo_initially_matches_base_forecast(echo_pipeline):
    context = torch.rand(3, 16)
    text_input_ids = torch.randint(104, 200, (3, 16))
    text_attention_mask = torch.ones_like(text_input_ids)

    output = echo_pipeline.model(
        context=context,
        num_output_patches=1,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        risk_features=torch.rand(3, 4, 1),
    )

    assert torch.allclose(output.quantile_preds, output.base_quantile_preds, atol=1e-6)


def test_echo_force_base_prediction_bypasses_residual(echo_pipeline):
    context = torch.rand(3, 16)
    text_input_ids = torch.randint(104, 200, (3, 16))
    text_attention_mask = torch.ones_like(text_input_ids)
    echo_pipeline.model.echo.residual_scale.data.fill_(1.0)

    output = echo_pipeline.model(
        context=context,
        num_output_patches=1,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        force_base_prediction=True,
    )

    assert torch.allclose(output.quantile_preds, output.base_quantile_preds, atol=1e-6)


def test_echo_can_use_pseudo_image_without_external_modalities(echo_pipeline):
    context = torch.rand(2, 32)

    output = echo_pipeline.model(context=context, num_output_patches=2)

    assert output.quantile_preds.shape == (2, DEFAULT_MODEL_NUM_QUANTILES, 32)
    assert output.echo_gate.shape == (2, 1, 32)


def test_echo_gate_is_forecast_patch_specific(echo_pipeline):
    context = torch.rand(2, 32)

    output = echo_pipeline.model(context=context, num_output_patches=2, risk_features=torch.rand(2, 4, 1))

    assert output.echo_gate.shape == (2, 1, 32)
    assert not torch.allclose(output.echo_gate[..., :16], output.echo_gate[..., 16:])


def test_echo_requires_real_modality_when_pseudo_image_disabled(echo_pipeline):
    echo_pipeline.model.echo_config.use_pseudo_image = False
    echo_pipeline.model.echo.echo_config.use_pseudo_image = False
    context = torch.rand(2, 16)

    with pytest.raises(ValueError, match="requires at least one modality"):
        echo_pipeline.model(context=context, num_output_patches=1)


def test_predict_timemmd_returns_time_major_predictions(tmp_path, echo_pipeline):
    csv_path = tmp_path / "Tiny.csv"
    _write_timemmd_csv(csv_path)

    output = echo_pipeline.predict_timemmd(
        root_path=tmp_path,
        data_path=csv_path.name,
        target="OT",
        seq_len=12,
        pred_len=4,
        features="S",
        batch_size=8,
        tokenizer=TestTokenizer(),
    )

    assert output["quantiles"].shape[-1] == DEFAULT_MODEL_NUM_QUANTILES
    assert output["predictions"].shape[1:] == (4, 1)
    assert output["targets"].shape[1:] == (4, 1)
    assert output["targets"].abs().max() <= 1.0


def test_predict_timemmd_uses_real_image_paths(tmp_path, echo_pipeline):
    csv_path = tmp_path / "TinyImages.csv"
    _write_timemmd_csv_with_images(csv_path)

    output = echo_pipeline.predict_timemmd(
        root_path=tmp_path,
        data_path=csv_path.name,
        target="OT",
        seq_len=12,
        pred_len=4,
        features="S",
        batch_size=8,
        tokenizer=TestTokenizer(),
        image_column="image_path",
        image_size=16,
    )

    assert output["quantiles"].shape[-1] == DEFAULT_MODEL_NUM_QUANTILES
    assert output["predictions"].shape[1:] == (4, 1)


def test_predict_timemmd_can_return_base_predictions(tmp_path, echo_pipeline):
    csv_path = tmp_path / "Tiny.csv"
    _write_timemmd_csv(csv_path)

    output = echo_pipeline.predict_timemmd(
        root_path=tmp_path,
        data_path=csv_path.name,
        target="OT",
        seq_len=12,
        pred_len=4,
        features="S",
        batch_size=8,
        tokenizer=TestTokenizer(),
        return_base=True,
    )

    assert output["base_quantiles"].shape == output["quantiles"].shape
    assert output["base_predictions"].shape == output["predictions"].shape


def test_fit_echo_defaults_to_echo_only(echo_pipeline):
    ft_pipeline = echo_pipeline.fit_echo()

    trainable = [name for name, param in ft_pipeline.model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(".echo." in f".{name}." for name in trainable)


def test_validation_guard_resets_residual_scale_when_echo_is_worse(echo_pipeline):
    echo_pipeline.model.echo.residual_scale.data.fill_(1.0)

    def fake_rmse(model, dataset, *, force_base_prediction):
        del model, dataset
        return 2.0 if not force_base_prediction else 1.0

    echo_pipeline._dataset_rmse = fake_rmse
    with pytest.warns(UserWarning, match="exceeded baseline RMSE"):
        echo_pipeline._guard_against_baseline(echo_pipeline.model, object())

    assert echo_pipeline.model.echo.residual_scale.item() == 0.0


def test_fit_timemmd_trains_only_lora_and_echo_params(tmp_path, echo_pipeline):
    pytest.importorskip("peft")
    csv_path = tmp_path / "Tiny.csv"
    _write_timemmd_csv(csv_path)

    ft_pipeline = echo_pipeline.fit_timemmd(
        root_path=tmp_path,
        data_path=csv_path.name,
        target="OT",
        seq_len=12,
        pred_len=4,
        features="S",
        batch_size=8,
        num_steps=2,
        output_dir=tmp_path / "out",
        eval_strategy="no",
        save_strategy="no",
        validation_flag=None,
        tokenizer=TestTokenizer(),
    )

    trainable = [name for name, param in ft_pipeline.model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(("lora_" in name or ".echo." in f".{name}." or "modules_to_save" in name) for name in trainable)
