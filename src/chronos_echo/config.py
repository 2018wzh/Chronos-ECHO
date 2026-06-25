# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class Chronos2EchoConfig:
    """Configuration for the Chronos-2-ECHO multimodal adapter."""

    echo_hidden_size: int | None = None
    num_echo_layers: int = 1
    num_attention_heads: int | None = None
    dropout_rate: float | None = None
    use_event_gate: bool = True
    max_text_length: int = 500
    text_model_name_or_path: str | None = None
    text_tokenizer_name_or_path: str | None = None
    text_vocab_size: int = 30522
    freeze_text_backbone: bool = True
    num_text_tokens: int = 8
    num_vision_tokens: int = 8
    vision_patch_size: int = 16
    vision_image_size: int = 64
    vision_model_name_or_path: str | None = None
    freeze_vision_backbone: bool = True
    risk_feature_dim: int = 1
    residual_feature_dim: int = 1
    use_guided_fusion: bool = True
    modality_dropout_rate: float = 0.1
    point_loss_weight: float = 0.2
    delta_loss_weight: float = 0.01
    residual_scale_init: float = 0.0
    gate_bias_init: float = -3.0
    use_pseudo_image: bool = True
    guard_against_baseline: bool = True
    quantile_loss_weight: float = 1.0
    base_loss_weight: float = 0.0
    reconstruction_loss_weight: float = 0.0
    freq_mask_ratio: float = 0.5
    freq_mask_thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)
