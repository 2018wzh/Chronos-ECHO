import json
from pathlib import Path

import pytest
import torch

from chronos_echo import Chronos2EchoConfig, Chronos2EchoPipeline


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


@pytest.fixture
def echo_pipeline() -> Chronos2EchoPipeline:
    return Chronos2EchoPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu", echo_config=_echo_config())


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
    echo_pipeline.model.echo.residual_scale.data.fill_(1.0)

    output = echo_pipeline.model(context=context, num_output_patches=1, force_base_prediction=True)

    assert torch.allclose(output.quantile_preds, output.base_quantile_preds, atol=1e-6)


def test_echo_can_use_pseudo_image_without_external_modalities(echo_pipeline):
    context = torch.rand(2, 32)

    output = echo_pipeline.model(context=context, num_output_patches=2)

    assert output.quantile_preds.shape == (2, DEFAULT_MODEL_NUM_QUANTILES, 32)
    assert output.echo_gate.shape == (2, 1, 32)


def test_echo_gate_is_forecast_patch_specific(echo_pipeline):
    context = torch.rand(2, 32)
    gate = echo_pipeline.model.echo.event_gate
    gate.position_scale.data.fill_(0.1)
    gate.net[2].weight.data.zero_()
    gate.net[2].bias.data.zero_()

    output = echo_pipeline.model(context=context, num_output_patches=2, risk_features=torch.rand(2, 4, 1))

    assert output.echo_gate.shape == (2, 1, 32)
    assert not torch.allclose(output.echo_gate[..., :16], output.echo_gate[..., 16:])


def test_echo_safety_reset_restores_gate_position_scale(echo_pipeline):
    gate = echo_pipeline.model.echo.event_gate
    gate.position_scale.data.fill_(float("nan"))
    echo_pipeline.model.echo.text_encoder.distiller.query_tokens.data.fill_(float("nan"))

    echo_pipeline.model.reset_echo_safety_parameters(zero_residual_head=False)

    assert torch.isfinite(gate.position_scale)
    assert gate.position_scale.item() == pytest.approx(1e-3)
    assert torch.isfinite(echo_pipeline.model.echo.text_encoder.distiller.query_tokens).all()


def test_echo_requires_real_modality_when_pseudo_image_disabled(echo_pipeline):
    echo_pipeline.model.echo_config.use_pseudo_image = False
    echo_pipeline.model.echo.echo_config.use_pseudo_image = False
    context = torch.rand(2, 16)

    with pytest.raises(ValueError, match="requires at least one modality"):
        echo_pipeline.model(context=context, num_output_patches=1)


def test_fit_echo_defaults_to_echo_only(echo_pipeline):
    ft_pipeline = echo_pipeline.fit_echo()

    trainable = [name for name, param in ft_pipeline.model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(".echo." in f".{name}." for name in trainable)
