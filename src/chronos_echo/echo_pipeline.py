# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import torch
from transformers import AutoConfig

from chronos.base import BaseChronosPipeline
from chronos.chronos2.pipeline import Chronos2Pipeline

from .config import Chronos2EchoConfig
from .echo import Chronos2EchoModel

if TYPE_CHECKING:
    from peft import LoraConfig


class Chronos2EchoPipeline(Chronos2Pipeline):
    def __init__(self, model: Chronos2EchoModel):
        super().__init__(model=model)
        self.model = model

    @staticmethod
    def _echo_config_to_dict(echo_config: Chronos2EchoConfig | dict | None) -> dict:
        if echo_config is None:
            return Chronos2EchoConfig().__dict__
        if isinstance(echo_config, Chronos2EchoConfig):
            return echo_config.__dict__
        return dict(echo_config)

    def _clone_as_echo_model(self, echo_config: Chronos2EchoConfig | dict | None = None) -> Chronos2EchoModel:
        config = deepcopy(self.model.config)
        if echo_config is not None or not hasattr(config, "echo_config"):
            config.echo_config = self._echo_config_to_dict(echo_config)
        model = Chronos2EchoModel(config).to(self.model.device)  # type: ignore[arg-type]
        target_state = model.state_dict()
        source_state = {
            name: value
            for name, value in self.model.state_dict().items()
            if name in target_state and target_state[name].shape == value.shape
        }
        model.load_state_dict(source_state, strict=False)
        model.config.echo_config = model.echo_config.__dict__
        model.config.chronos_pipeline_class = "Chronos2EchoPipeline"
        model.config.architectures = ["Chronos2EchoModel"]
        return model

    @staticmethod
    def _mark_echo_trainable(model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if ".echo." in f".{name}.":
                param.requires_grad = True

    def _prepare_trainable_model(
        self,
        model: Chronos2EchoModel,
        finetune_mode: Literal["echo_only", "lora", "full"],
        lora_config: "LoraConfig | dict | None",
    ) -> torch.nn.Module:
        if finetune_mode == "full":
            for param in model.parameters():
                param.requires_grad = True
            return model

        for param in model.parameters():
            param.requires_grad = False

        if finetune_mode == "lora":
            from transformers.utils.import_utils import is_peft_available

            if not is_peft_available():
                raise ImportError("`peft` is required for `finetune_mode='lora'`. Install it with `pip install peft`.")
            from peft import LoraConfig, get_peft_model

            if lora_config is None:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    modules_to_save=["echo"],
                    target_modules=[
                        "self_attention.q",
                        "self_attention.v",
                        "self_attention.k",
                        "self_attention.o",
                        "output_patch_embedding.output_layer",
                    ],
                )
            elif isinstance(lora_config, dict):
                lora_config = LoraConfig(**lora_config)
            model = get_peft_model(model, lora_config)
        elif finetune_mode != "echo_only":
            raise ValueError("finetune_mode must be one of 'echo_only', 'lora', or 'full'")

        self._mark_echo_trainable(model)
        return model

    def fit_echo(
        self,
        *,
        echo_config: Chronos2EchoConfig | dict | None = None,
        finetune_mode: Literal["echo_only", "lora", "full"] = "echo_only",
        lora_config: "LoraConfig | dict | None" = None,
    ) -> "Chronos2EchoPipeline":
        model = self._clone_as_echo_model(echo_config=echo_config)
        trainable_model = self._prepare_trainable_model(model, finetune_mode=finetune_mode, lora_config=lora_config)
        return Chronos2EchoPipeline(model=trainable_model)  # type: ignore[arg-type]

    @staticmethod
    def _unwrap_echo_model(model: torch.nn.Module) -> Chronos2EchoModel:
        return model.get_base_model() if hasattr(model, "get_base_model") else model  # type: ignore[return-value]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, echo_config: Chronos2EchoConfig | dict | None = None, **kwargs):
        if str(pretrained_model_name_or_path).startswith("s3://"):
            return BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        device_map = kwargs.pop("device_map", None)
        torch_dtype = kwargs.get("torch_dtype", "auto")
        if torch_dtype != "auto" and isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = cls.dtypes[torch_dtype]

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if not hasattr(config, "chronos_config"):
            raise ValueError("Not a Chronos config file")

        reset_new_echo = echo_config is not None or not hasattr(config, "echo_config")
        if echo_config is not None or not hasattr(config, "echo_config"):
            config.echo_config = cls._echo_config_to_dict(echo_config)
        config.chronos_pipeline_class = "Chronos2EchoPipeline"
        config.architectures = ["Chronos2EchoModel"]

        model = Chronos2EchoModel.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
        if reset_new_echo:
            model.reset_echo_residual_head_random()
            model.load_pretrained_echo_backbones()
        if device_map is not None:
            if not isinstance(device_map, str) or device_map == "auto":
                raise ValueError("Chronos2EchoPipeline.from_pretrained supports only string devices like 'cpu' or 'cuda'.")
            model.to(device_map)
        if reset_new_echo:
            model.reset_echo_safety_parameters(zero_residual_head=False)
        return cls(model=model)
