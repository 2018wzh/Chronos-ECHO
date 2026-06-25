# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.utils import ModelOutput

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from .config import Chronos2EchoConfig


def freq_mask(x: torch.Tensor, mask_ratio: float = 0.5, thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)) -> torch.Tensor:
    """Apply frequency-domain masking to a 1-D time series, inspired by Aurora.

    For each threshold ratio, the input is transformed via FFT and either the
    low-frequency or high-frequency components are zeroed out at random.
    The resulting masked variants are stacked and interleaved with the original.

    Parameters
    ----------
    x
        Input tensor of shape ``(batch, seq_len)``.
    mask_ratio
        Probability of masking the high-frequency (vs. low-frequency) components.
    thresholds
        Cutoff ratios for frequency truncation.

    Returns
    -------
    torch.Tensor
        Masked variants of shape ``(batch * (len(thresholds) + 1), seq_len)``
        where the first ``batch`` entries are the original (unmasked) input.
    """
    x_fft = torch.fft.rfft(x, dim=-1)
    masked_list = [x]
    for ratio in thresholds:
        temp = x_fft.clone()
        truncation = int(temp.shape[-1] * ratio)
        if random.random() > mask_ratio:
            temp[:, :truncation] = 0.0   # mask low-frequency → keep high-freq details
        else:
            temp[:, truncation:] = 0.0   # mask high-frequency → keep low-freq trend
        masked_list.append(torch.fft.irfft(temp, dim=-1))
    return torch.cat(masked_list, dim=0)


class ContextReconstructor(nn.Module):
    """Lightweight head that reconstructs original context patches from encoder hidden states."""

    def __init__(self, hidden_size: int, input_patch_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, input_patch_size)

    def forward(self, hidden_states: torch.Tensor, num_context_patches: int) -> torch.Tensor:
        # Take only context token positions, projecting each to original patch values
        context_tokens = hidden_states[:, :num_context_patches]
        return self.proj(context_tokens)


def _as_echo_config(config: Chronos2CoreConfig) -> Chronos2EchoConfig:
    raw_config = getattr(config, "echo_config", None)
    if raw_config is None:
        return Chronos2EchoConfig()
    if isinstance(raw_config, Chronos2EchoConfig):
        return raw_config
    known_fields = {field.name for field in fields(Chronos2EchoConfig)}
    return Chronos2EchoConfig(**{key: value for key, value in raw_config.items() if key in known_fields})


def _choose_num_heads(hidden_size: int, requested: int | None, default_heads: int) -> int:
    candidates = [requested, default_heads, 8, 4, 2, 1]
    for candidate in candidates:
        if candidate is not None and candidate > 0 and hidden_size % candidate == 0:
            return candidate
    return 1


class EchoTokenDistiller(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, token_states: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        queries = repeat(self.query_tokens, "n d -> b n d", b=token_states.shape[0])
        distilled, _ = self.attn(queries, token_states, token_states, key_padding_mask=key_padding_mask)
        return self.norm(queries + distilled)


class EchoTextEncoder(nn.Module):
    def __init__(self, echo_config: Chronos2EchoConfig, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.backbone = None
        self.embedding: nn.Embedding | None = None

        if echo_config.text_model_name_or_path is not None:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(echo_config.text_model_name_or_path)
            text_hidden_size = self.backbone.config.hidden_size
            if echo_config.freeze_text_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            self.embedding = nn.Embedding(echo_config.text_vocab_size, hidden_size, padding_idx=0)
            text_hidden_size = hidden_size

        self.projection = nn.Linear(text_hidden_size, hidden_size)
        self.distiller = EchoTokenDistiller(
            hidden_size=hidden_size,
            num_tokens=echo_config.num_text_tokens,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        text_input_ids: torch.Tensor | None,
        text_attention_mask: torch.Tensor | None = None,
        text_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if text_input_ids is None:
            return None

        if text_attention_mask is None:
            text_attention_mask = text_input_ids.ne(0).to(dtype=torch.long)

        if self.backbone is not None:
            kwargs = {"input_ids": text_input_ids, "attention_mask": text_attention_mask}
            if text_token_type_ids is not None:
                kwargs["token_type_ids"] = text_token_type_ids
            try:
                token_states = self.backbone(**kwargs).last_hidden_state
            except TypeError:
                kwargs.pop("token_type_ids", None)
                token_states = self.backbone(**kwargs).last_hidden_state
        else:
            assert self.embedding is not None
            token_states = self.embedding(text_input_ids.clamp_min(0))

        token_states = self.projection(token_states)
        key_padding_mask = text_attention_mask.eq(0)
        return self.distiller(token_states, key_padding_mask=key_padding_mask)


class EchoVisionEncoder(nn.Module):
    def __init__(self, echo_config: Chronos2EchoConfig, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.echo_config = echo_config
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.vision_model: nn.Module | None = None
        self.patch_embed: nn.Conv2d | None = None
        self.patch_size = echo_config.vision_patch_size

        if echo_config.vision_model_name_or_path is not None:
            # Frozen ViT backbone (inspired by Aurora) — extract general-purpose
            # visual features from rendered time series images.
            from transformers import ViTModel

            self.vision_model = ViTModel.from_pretrained(echo_config.vision_model_name_or_path)
            vit_hidden = self.vision_model.config.hidden_size
            if echo_config.freeze_vision_backbone:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
            self.vit_projection = nn.Linear(vit_hidden, hidden_size)

            # Cross-attention distiller: learnable query tokens attend to ViT patch
            # outputs, mirroring Aurora's token distillation strategy.
            self.distiller = _ViTDistiller(
                hidden_size=hidden_size,
                num_tokens=echo_config.num_vision_tokens,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
        else:
            self.patch_embed = nn.Conv2d(1, hidden_size, kernel_size=echo_config.vision_patch_size, stride=echo_config.vision_patch_size)
            self.distiller = EchoTokenDistiller(
                hidden_size=hidden_size,
                num_tokens=echo_config.num_vision_tokens,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )

    def _pseudo_image_from_context(self, context: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(context.float(), nan=0.0)
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        length = x.shape[-1]
        spectrum = torch.fft.rfft(x, dim=-1).abs().mean(dim=0)
        if spectrum.numel() > 1:
            spectrum[0] = 0.0
            peak = int(spectrum.argmax().item())
        else:
            peak = 0
        period = length // peak if peak > 0 else min(self.patch_size, length)
        period = max(1, min(period, length))
        padding = (period - (length % period)) % period
        x = F.pad(x, (padding, 0))
        image = rearrange(x, "b (f p) -> b 1 f p", p=period)
        return F.interpolate(
            image,
            size=(self.echo_config.vision_image_size, self.echo_config.vision_image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _normalize_image_input(self, vision_values: torch.Tensor) -> torch.Tensor:
        if vision_values.ndim == 3:
            vision_values = vision_values.unsqueeze(1)
        if vision_values.ndim != 4:
            raise ValueError(
                "vision_values must have shape (batch, height, width) or (batch, channels, height, width)"
            )
        if vision_values.shape[1] != 1:
            vision_values = vision_values.float().mean(dim=1, keepdim=True)
        return vision_values.float()

    def forward(self, vision_values: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        if vision_values is None:
            if context is None or not self.echo_config.use_pseudo_image:
                raise ValueError("vision_values or pseudo-image context must be provided for EchoVisionEncoder")
            image = self._pseudo_image_from_context(context)
        else:
            image = self._normalize_image_input(vision_values)

        if self.vision_model is not None:
            # ViT expects 3-channel input — replicate the single-channel image
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            image = image.to(
                device=next(self.vision_model.parameters()).device,
                dtype=next(self.vision_model.parameters()).dtype,
            )
            vit_outputs = self.vision_model(pixel_values=image)
            token_states = self.vit_projection(vit_outputs.last_hidden_state)
            return self.distiller(token_states)
        else:
            image = image.to(device=self.patch_embed.weight.device, dtype=self.patch_embed.weight.dtype)  # type: ignore[union-attr]
            patches = self.patch_embed(image)  # type: ignore[union-attr]
            token_states = rearrange(patches, "b d h w -> b (h w) d")
            return self.distiller(token_states)


class _ViTDistiller(nn.Module):
    """Cross-attention distillation for ViT outputs, inspired by Aurora's token distillation.

    Unlike ``EchoTokenDistiller`` (which uses self-attention based query tokens),
    this module uses a ``TransformerDecoder`` layer so that learnable query tokens
    can cross-attend to the variable-length ViT patch sequence.
    """

    def __init__(self, hidden_size: int, num_tokens: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.02)
        self.cross_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=1,
            norm=nn.LayerNorm(hidden_size),
        )

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        queries = repeat(self.query_tokens, "n d -> b n d", b=token_states.shape[0])
        return self.cross_attn(queries, token_states)


class EchoFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(in_dim, hidden_size)

    def forward(self, features: torch.Tensor | None) -> torch.Tensor | None:
        if features is None:
            return None
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError("Echo feature tensors must have shape (batch, dim) or (batch, tokens, dim)")
        return self.projection(features.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype))


class EchoCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attn(query_states, memory_states, memory_states)
        hidden_states = self.norm1(query_states + attended)
        return self.norm2(hidden_states + self.ffn(hidden_states))


class EchoGuidedFusionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    @staticmethod
    def _modality_similarity(query_states: torch.Tensor, tokens: torch.Tensor | None) -> torch.Tensor | None:
        if tokens is None:
            return None
        attn = torch.softmax(query_states @ tokens.transpose(-1, -2) / (query_states.shape[-1] ** 0.5), dim=-1)
        return attn @ attn.transpose(-1, -2)

    def _guided_bias(
        self,
        query_states: torch.Tensor,
        text_tokens: torch.Tensor | None,
        vision_tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        biases = [
            bias for bias in (
                self._modality_similarity(query_states, text_tokens),
                self._modality_similarity(query_states, vision_tokens),
            )
            if bias is not None
        ]
        if not biases:
            return None
        bias = sum(biases) / len(biases)
        return torch.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0).repeat_interleave(self.num_heads, dim=0)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        text_tokens: torch.Tensor | None = None,
        vision_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        guided_bias = self._guided_bias(query_states, text_tokens, vision_tokens)
        attended, _ = self.self_attn(query_states, query_states, query_states, attn_mask=guided_bias)
        hidden_states = self.norm1(query_states + attended)
        connected, _ = self.cross_attn(hidden_states, memory_states, memory_states)
        hidden_states = self.norm2(hidden_states + connected)
        return self.norm3(hidden_states + self.ffn(hidden_states))


class EventGate(nn.Module):
    def __init__(self, hidden_size: int, output_patch_size: int, num_heads: int, gate_bias_init: float):
        super().__init__()
        self.output_patch_size = output_patch_size
        self.position_scale = nn.Parameter(torch.tensor(1e-3))
        self.memory_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[2].bias, gate_bias_init)

    def forward(self, future_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        modal_context, _ = self.memory_attn(future_states, memory_states, memory_states)
        modal_context = torch.nan_to_num(modal_context, nan=0.0, posinf=0.0, neginf=0.0)
        gate = self.net(torch.cat([future_states, modal_context], dim=-1))
        positions = torch.linspace(0.0, 1.0, gate.shape[1], device=gate.device, dtype=gate.dtype).view(1, -1, 1)
        gate = (gate + self.position_scale.to(gate.dtype) * positions).clamp(0.0, 1.0)
        gate = repeat(gate, "b n 1 -> b 1 (n p)", p=self.output_patch_size)
        return gate


class QuantileResidualHead(nn.Module):
    def __init__(self, hidden_size: int, num_quantiles: int, output_patch_size: int):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.output_patch_size = output_patch_size
        self.proj = nn.Linear(hidden_size, num_quantiles * output_patch_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.proj(hidden_states)
        return rearrange(
            residual,
            "b n (q p) -> b q (n p)",
            q=self.num_quantiles,
            p=self.output_patch_size,
        )


class Chronos2EchoAdapter(nn.Module):
    def __init__(self, core_config: Chronos2CoreConfig, echo_config: Chronos2EchoConfig, num_quantiles: int):
        super().__init__()
        self.echo_config = echo_config
        self.model_dim = core_config.d_model
        self.hidden_size = echo_config.echo_hidden_size or core_config.d_model
        self.output_patch_size = core_config.chronos_config["output_patch_size"]
        dropout_rate = echo_config.dropout_rate if echo_config.dropout_rate is not None else core_config.dropout_rate
        num_heads = _choose_num_heads(
            self.hidden_size,
            requested=echo_config.num_attention_heads,
            default_heads=core_config.num_heads,
        )

        self.query_projection = nn.Linear(core_config.d_model, self.hidden_size)
        self.text_encoder = EchoTextEncoder(echo_config, self.hidden_size, num_heads, dropout_rate)
        self.vision_encoder = EchoVisionEncoder(echo_config, self.hidden_size, num_heads, dropout_rate)
        self.risk_encoder = EchoFeatureEncoder(echo_config.risk_feature_dim, self.hidden_size)
        self.residual_encoder = EchoFeatureEncoder(echo_config.residual_feature_dim, self.hidden_size)
        block_cls = EchoGuidedFusionBlock if echo_config.use_guided_fusion else EchoCrossAttentionBlock
        self.fusion_blocks = nn.ModuleList(
            [block_cls(self.hidden_size, num_heads, dropout_rate) for _ in range(echo_config.num_echo_layers)]
        )
        self.residual_head = QuantileResidualHead(self.hidden_size, num_quantiles, self.output_patch_size)
        self.event_gate = EventGate(
            self.hidden_size,
            self.output_patch_size,
            num_heads,
            echo_config.gate_bias_init,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(echo_config.residual_scale_init)))
        self.context_reconstructor = ContextReconstructor(self.hidden_size, core_config.chronos_config["input_patch_size"])

    def _maybe_keep_modality(self, tokens: torch.Tensor | None, required: bool = False) -> torch.Tensor | None:
        if tokens is None:
            return None
        if required or not self.training or self.echo_config.modality_dropout_rate <= 0:
            return tokens
        if torch.rand((), device=tokens.device) < self.echo_config.modality_dropout_rate:
            return None
        return tokens

    def forward(
        self,
        future_states: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        text_token_type_ids: torch.Tensor | None = None,
        vision_values: torch.Tensor | None = None,
        risk_features: torch.Tensor | None = None,
        residual_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        raw_tokens = []

        text_tokens = self.text_encoder(text_input_ids, text_attention_mask, text_token_type_ids)
        if text_tokens is not None:
            raw_tokens.append(("text", text_tokens))

        vision_tokens = None
        if vision_values is not None or (self.echo_config.use_pseudo_image and context is not None):
            vision_tokens = self.vision_encoder(vision_values=vision_values, context=context)
            raw_tokens.append(("vision", vision_tokens))

        risk_tokens = self.risk_encoder(risk_features)
        if risk_tokens is not None:
            raw_tokens.append(("risk", risk_tokens))

        residual_tokens = self.residual_encoder(residual_features)
        if residual_tokens is not None:
            raw_tokens.append(("residual", residual_tokens))

        if not raw_tokens:
            raise ValueError("Chronos2EchoAdapter requires at least one modality or feature input")

        kept = []
        kept_names = []
        for name, tokens in raw_tokens:
            tokens = self._maybe_keep_modality(tokens, required=len(raw_tokens) == 1)
            if tokens is not None:
                kept.append(tokens)
                kept_names.append(name)
        if not kept:
            name, tokens = raw_tokens[0]
            kept = [tokens]
            kept_names = [name]

        memory_states = torch.cat(kept, dim=1).to(future_states.dtype)
        text_tokens = text_tokens if "text" in kept_names else None
        vision_tokens = vision_tokens if "vision" in kept_names else None
        echo_states = self.query_projection(future_states)
        for block in self.fusion_blocks:
            if isinstance(block, EchoGuidedFusionBlock):
                echo_states = block(
                    echo_states,
                    memory_states,
                    text_tokens=text_tokens,
                    vision_tokens=vision_tokens,
                )
            else:
                echo_states = block(echo_states, memory_states)
            echo_states = torch.nan_to_num(echo_states, nan=0.0, posinf=0.0, neginf=0.0)

        delta = self.residual_head(echo_states)
        if self.echo_config.use_event_gate:
            gate = self.event_gate(echo_states, memory_states)
        else:
            gate = torch.ones(delta.shape[0], 1, delta.shape[-1], device=delta.device, dtype=delta.dtype)

        return delta, gate


@dataclass
class Chronos2EchoOutput(ModelOutput):
    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    base_quantile_preds: torch.Tensor | None = None
    delta_quantile_preds: torch.Tensor | None = None
    echo_gate: torch.Tensor | None = None
    enc_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    enc_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None


class Chronos2EchoModel(Chronos2Model):
    def __init__(self, config: Chronos2CoreConfig):
        if not hasattr(config, "echo_config"):
            config.echo_config = Chronos2EchoConfig().__dict__
        super().__init__(config)
        self.echo_config = _as_echo_config(config)
        self.echo = Chronos2EchoAdapter(config, self.echo_config, self.num_quantiles)
        self.config.echo_config = self.echo_config.__dict__
        self.config.architectures = ["Chronos2EchoModel"]
        self.config.chronos_pipeline_class = "Chronos2EchoPipeline"
        self.reset_echo_safety_parameters()

    def reset_echo_safety_parameters(self) -> None:
        nn.init.zeros_(self.echo.residual_head.proj.weight)
        nn.init.zeros_(self.echo.residual_head.proj.bias)
        self.echo.residual_scale.data.fill_(float(self.echo_config.residual_scale_init))
        nn.init.constant_(self.echo.event_gate.net[2].bias, float(self.echo_config.gate_bias_init))

    @staticmethod
    def _duplicate_modality(tensor: torch.Tensor | None, num_copies: int) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.repeat(num_copies, *([1] * (tensor.ndim - 1)))

    @staticmethod
    def _monotonic_quantiles(quantile_preds: torch.Tensor) -> torch.Tensor:
        return torch.cummax(quantile_preds, dim=1).values

    def _unscale_quantiles(
        self,
        quantile_preds: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        num_output_patches: int,
    ) -> torch.Tensor:
        horizon = num_output_patches * self.chronos_config.output_patch_size
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=horizon,
        )
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        return rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=horizon,
        )

    def _compute_median_huber_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        median_idx = min(range(self.num_quantiles), key=lambda idx: abs(float(self.quantiles[idx]) - 0.5))
        median_preds = quantile_preds[:, median_idx]
        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.to(self.device)
        future_target_mask = (
            future_target_mask.to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        if median_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (*future_target.shape[:-1], median_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat([future_target, torch.zeros(padding_shape).to(future_target)], dim=-1)
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(padding_shape).to(future_target_mask)], dim=-1
            )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)
        inv_future_covariate_mask = 1 - rearrange(
            patched_future_covariates_mask,
            "b n p -> b (n p)",
            b=future_target.shape[0],
            n=num_output_patches,
            p=self.chronos_config.output_patch_size,
        )
        loss_mask = future_target_mask.float() * inv_future_covariate_mask
        loss = nn.functional.smooth_l1_loss(median_preds, future_target, reduction="none") * loss_mask
        return loss.sum() / loss_mask.sum().clamp_min(1.0)

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        text_token_type_ids: torch.Tensor | None = None,
        vision_values: torch.Tensor | None = None,
        risk_features: torch.Tensor | None = None,
        residual_features: torch.Tensor | None = None,
        target_idx_ranges: list[tuple[int, int]] | None = None,
        force_base_prediction: bool = False,
    ) -> Chronos2EchoOutput:
        if not self.echo_config.use_pseudo_image and all(
            value is None
            for value in (
                text_input_ids,
                vision_values,
                risk_features,
                residual_features,
            )
        ):
            raise ValueError("Chronos2EchoModel requires at least one modality or feature input")

        batch_size = context.shape[0]
        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
        )
        hidden_states: torch.Tensor = encoder_outputs[0]
        forecast_embeds = hidden_states[:, -num_output_patches:]
        base_quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)
        base_quantile_preds = rearrange(
            base_quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )
        base_quantile_preds = self._monotonic_quantiles(base_quantile_preds)

        if force_base_prediction:
            delta = torch.zeros_like(base_quantile_preds)
            gate = torch.zeros(
                base_quantile_preds.shape[0],
                1,
                base_quantile_preds.shape[-1],
                device=base_quantile_preds.device,
                dtype=base_quantile_preds.dtype,
            )
            echo_quantile_preds = base_quantile_preds
        else:
            delta, gate = self.echo(
                forecast_embeds,
                context=context,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                vision_values=vision_values,
                risk_features=risk_features,
                residual_features=residual_features,
            )
            assert delta is not None and gate is not None
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            adjusted_quantile_preds = base_quantile_preds + self.echo.residual_scale * gate * delta
            adjusted_quantile_preds = torch.where(
                torch.isfinite(adjusted_quantile_preds),
                adjusted_quantile_preds,
                base_quantile_preds,
            )
            echo_quantile_preds = self._monotonic_quantiles(adjusted_quantile_preds)

        loss = None
        if future_target is not None:
            loss = self._compute_loss(
                quantile_preds=echo_quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )
            loss = loss * self.echo_config.quantile_loss_weight
            if self.echo_config.point_loss_weight > 0:
                loss = loss + self.echo_config.point_loss_weight * self._compute_median_huber_loss(
                    quantile_preds=echo_quantile_preds,
                    future_target=future_target,
                    future_target_mask=future_target_mask,
                    patched_future_covariates_mask=patched_future_covariates_mask,
                    loc_scale=loc_scale,
                    num_output_patches=num_output_patches,
                )
            if self.echo_config.delta_loss_weight > 0:
                loss = loss + self.echo_config.delta_loss_weight * (gate * delta).pow(2).mean()
            if self.echo_config.base_loss_weight > 0:
                base_loss = self._compute_loss(
                    quantile_preds=base_quantile_preds,
                    future_target=future_target,
                    future_target_mask=future_target_mask,
                    patched_future_covariates_mask=patched_future_covariates_mask,
                    loc_scale=loc_scale,
                    num_output_patches=num_output_patches,
                )
                loss = loss + self.echo_config.base_loss_weight * base_loss

            # Frequency-masked context reconstruction loss (inspired by Aurora)
            if self.training and self.echo_config.reconstruction_loss_weight > 0:
                recon_weight = self.echo_config.reconstruction_loss_weight
                recon_ratio = self.echo_config.freq_mask_ratio
                recon_thresholds = self.echo_config.freq_mask_thresholds
                num_mask_variants = len(recon_thresholds)

                # Create masked variants of the context
                masked_context = freq_mask(context, mask_ratio=recon_ratio, thresholds=recon_thresholds)

                # Duplicate modality inputs for the masked variants
                dup_text_ids = self._duplicate_modality(text_input_ids, num_mask_variants + 1)
                dup_text_mask = self._duplicate_modality(text_attention_mask, num_mask_variants + 1)
                dup_text_type = self._duplicate_modality(text_token_type_ids, num_mask_variants + 1)
                dup_vision = self._duplicate_modality(vision_values, num_mask_variants + 1)
                dup_risk = self._duplicate_modality(risk_features, num_mask_variants + 1)
                dup_residual = self._duplicate_modality(residual_features, num_mask_variants + 1)

                # Forward the augmented batch through the encoder to obtain representations
                # for the masked variants.  The encoder parameters receive no gradient from
                # this pass — only the Echo adapter and reconstructor are trained by the
                # reconstruction loss.
                recon_encode_kwargs: dict = {
                    "context": masked_context,
                    "num_output_patches": num_output_patches,
                }
                if context_mask is not None:
                    recon_encode_kwargs["context_mask"] = context_mask.repeat(num_mask_variants + 1, 1)
                if group_ids is not None:
                    recon_encode_kwargs["group_ids"] = group_ids.repeat(num_mask_variants + 1)
                if future_covariates is not None:
                    recon_encode_kwargs["future_covariates"] = future_covariates.repeat(num_mask_variants + 1, 1)
                if future_covariates_mask is not None:
                    recon_encode_kwargs["future_covariates_mask"] = \
                        future_covariates_mask.repeat(num_mask_variants + 1, 1)
                if future_target is not None:
                    recon_encode_kwargs["future_target"] = future_target.repeat(num_mask_variants + 1, 1)
                if future_target_mask is not None:
                    recon_encode_kwargs["future_target_mask"] = \
                        future_target_mask.repeat(num_mask_variants + 1, 1)
                with torch.no_grad():
                    recon_enc, _, _, _ = self.encode(**recon_encode_kwargs)
                recon_hidden: torch.Tensor = recon_enc[0]
                # Only the masked variant tokens are used for reconstruction; the original
                # (first batch_size) entries are skipped since they have nothing to reconstruct.
                recon_context_tokens = recon_hidden[batch_size:, :num_context_patches]
                recon_preds = self.echo.context_reconstructor(recon_context_tokens, num_context_patches)
                recon_preds = rearrange(recon_preds, "b n p -> b (n p)")
                # Trim to the original context length — Patch prepends NaN padding when
                # context.shape[-1] is not divisible by input_patch_size, so the
                # reconstructor produces more values than the original context has.
                recon_preds = recon_preds[..., -context.shape[-1] :]

                # The ground‑truth for each masked variant is the original context
                recon_targets = context.repeat(num_mask_variants, 1)
                recon_loss = nn.functional.mse_loss(recon_preds, recon_targets)
                loss = loss + recon_weight * recon_loss

                # Also run the Echo adapter on the masked forecast_embeds so that the
                # reconstruction head gradients can flow through the adapter's modality
                # encoders (detached forecast_embeds to avoid affecting the main task).
                recon_forecast = recon_hidden[batch_size:, -num_output_patches:].detach().requires_grad_(True)
                _, _ = self.echo(
                    recon_forecast,
                    context=masked_context[batch_size:],
                    text_input_ids=dup_text_ids[batch_size:] if dup_text_ids is not None else None,
                    text_attention_mask=dup_text_mask[batch_size:] if dup_text_mask is not None else None,
                    text_token_type_ids=dup_text_type[batch_size:] if dup_text_type is not None else None,
                    vision_values=dup_vision[batch_size:] if dup_vision is not None else None,
                    risk_features=dup_risk[batch_size:] if dup_risk is not None else None,
                    residual_features=dup_residual[batch_size:] if dup_residual is not None else None,
                )

        quantile_preds = self._unscale_quantiles(echo_quantile_preds, loc_scale, batch_size, num_output_patches)
        base_quantile_preds = self._unscale_quantiles(base_quantile_preds, loc_scale, batch_size, num_output_patches)

        return Chronos2EchoOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            base_quantile_preds=base_quantile_preds,
            delta_quantile_preds=delta,
            echo_gate=gate,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )
