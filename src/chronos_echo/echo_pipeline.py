# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from chronos.base import BaseChronosPipeline
from chronos.chronos2.pipeline import Chronos2Pipeline

from .config import Chronos2EchoConfig
from .echo import Chronos2EchoModel
from .timemmd import TimeMMDBatchDataset, TimeMMDWindowDataset, create_timemmd_tokenizer

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers.trainer_callback import TrainerCallback


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

    def _create_timemmd_dataset(
        self,
        *,
        root_path: str | Path,
        data_path: str,
        flag: str,
        seq_len: int,
        pred_len: int,
        target: str,
        features: str,
        batch_size: int,
        shuffle: bool,
        repeat: bool,
        image_column: str | None = None,
        image_root_path: str | Path | None = None,
        image_size: int | None = None,
        echo_config: Chronos2EchoConfig | None = None,
        tokenizer=None,
        text_column: str = "fact",
        missing_text: str = "error",
    ) -> tuple[TimeMMDWindowDataset, TimeMMDBatchDataset]:
        echo_config = echo_config or self.model.echo_config
        if tokenizer is None:
            tokenizer = create_timemmd_tokenizer(
                echo_config.text_tokenizer_name_or_path or echo_config.text_model_name_or_path,
            )
        window_dataset = TimeMMDWindowDataset(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            seq_len=seq_len,
            pred_len=pred_len,
            target=target,
            features=features,
            tokenizer=tokenizer,
            max_text_length=echo_config.max_text_length,
            text_column=text_column,
            missing_text=missing_text,
            image_column=image_column,
            image_root_path=image_root_path,
            image_size=image_size or echo_config.vision_image_size,
        )
        batch_dataset = TimeMMDBatchDataset(
            window_dataset,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            shuffle=shuffle,
            repeat=repeat,
        )
        return window_dataset, batch_dataset

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

    def fit_timemmd(
        self,
        *,
        root_path: str | Path,
        data_path: str | list[str],
        target: str = "OT",
        seq_len: int,
        pred_len: int,
        features: str = "S",
        batch_size: int = 256,
        flag: str = "train",
        validation_flag: str | None = "val",
        echo_config: Chronos2EchoConfig | dict | None = None,
        finetune_mode: Literal["echo_only", "lora", "full"] = "echo_only",
        lora_config: "LoraConfig | dict | None" = None,
        learning_rate: float = 1e-4,
        num_steps: int = 50000,
        warmup_steps: int = 5000,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "constant",
        gradient_accumulation_steps: int = 1,
        output_dir: Path | str | None = None,
        finetuned_ckpt_name: str = "finetuned-ckpt",
        callbacks: list["TrainerCallback"] | None = None,
        remove_printer_callback: bool = False,
        disable_data_parallel: bool = True,
        image_column: str | None = None,
        image_root_path: str | Path | None = None,
        image_size: int | None = None,
        tokenizer=None,
        **extra_trainer_kwargs,
    ) -> "Chronos2EchoPipeline":
        from transformers.trainer_callback import PrinterCallback
        from transformers.training_args import TrainingArguments

        from .trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

        model = self._clone_as_echo_model(echo_config=echo_config)
        trainable_model = self._prepare_trainable_model(model, finetune_mode=finetune_mode, lora_config=lora_config)

        if tokenizer is None:
            tokenizer = create_timemmd_tokenizer(
                model.echo_config.text_tokenizer_name_or_path or model.echo_config.text_model_name_or_path,
            )

        # Support single-dataset (str) and multi-dataset (list[str]) paths,
        # mirroring Aurora's ConcatDataset pretraining strategy.
        if isinstance(data_path, list):
            from .multimodal_dataset import MultiTimeMMDDataset

            sources = [(root_path, dp) for dp in data_path]
            train_dataset: TimeMMDBatchDataset = MultiTimeMMDDataset(  # type: ignore[assignment]
                sources,
                flag=flag,
                seq_len=seq_len,
                pred_len=pred_len,
                target=target,
                features=features,
                batch_size=batch_size,
                output_patch_size=self.model_output_patch_size,
                shuffle=True,
                repeat=True,
                tokenizer=tokenizer,
                max_text_length=model.echo_config.max_text_length,
                image_column=image_column,
                image_root_path=image_root_path,
                image_size=image_size or model.echo_config.vision_image_size,
            )
        else:
            _, train_dataset = self._create_timemmd_dataset(
                root_path=root_path,
                data_path=data_path,
                flag=flag,
                seq_len=seq_len,
                pred_len=pred_len,
                target=target,
                features=features,
                batch_size=batch_size,
                shuffle=True,
                repeat=True,
                image_column=image_column,
                image_root_path=image_root_path,
                image_size=image_size,
                echo_config=model.echo_config,
                tokenizer=tokenizer,
            )

        eval_dataset = None
        callbacks = callbacks or []
        if validation_flag is not None:
            _, eval_dataset = self._create_timemmd_dataset(
                root_path=root_path,
                data_path=data_path if isinstance(data_path, str) else data_path[0],
                flag=validation_flag,
                seq_len=seq_len,
                pred_len=pred_len,
                target=target,
                features=features,
                batch_size=batch_size,
                shuffle=False,
                repeat=False,
                image_column=image_column,
                image_root_path=image_root_path,
                image_size=image_size,
                echo_config=model.echo_config,
                tokenizer=tokenizer,
            )
            callbacks.append(EvaluateAndSaveFinalStepCallback())

        if output_dir is None:
            output_dir = Path("chronos-2-echo-finetuned") / time.strftime("%Y-%m-%d_%H-%M-%S")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        use_cpu = str(model.device) == "cpu"
        has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        if use_cpu and torch.cuda.is_available():
            warnings.warn(
                "The model is being fine-tuned on the CPU, but a CUDA device is available. "
                "We recommend using the GPU for faster fine-tuning.",
                category=UserWarning,
                stacklevel=2,
            )

        training_kwargs: dict = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps if warmup_ratio == 0.0 else 0,
            warmup_ratio=warmup_ratio if warmup_steps == 0 else 0.0,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=100,
            disable_tqdm=False,
            report_to="none",
            max_steps=num_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=0,
            tf32=has_sm80 and not use_cpu,
            bf16=has_sm80 and not use_cpu,
            save_only_model=True,
            prediction_loss_only=True,
            save_total_limit=1,
            save_strategy="no",
            save_steps=None,
            eval_strategy="no",
            eval_steps=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            use_cpu=use_cpu,
            remove_unused_columns=False,
        )
        if eval_dataset is not None:
            training_kwargs["save_strategy"] = "steps"
            training_kwargs["save_steps"] = 100
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = 100
            training_kwargs["load_best_model_at_end"] = True
            training_kwargs["metric_for_best_model"] = "eval_loss"
            training_kwargs["label_names"] = ["future_target"]

        training_kwargs.update(extra_trainer_kwargs)
        training_args = TrainingArguments(**training_kwargs)
        if disable_data_parallel and not use_cpu:
            training_args._n_gpu = 1

        trainer = Chronos2Trainer(
            model=trainable_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )
        if remove_printer_callback:
            trainer.pop_callback(PrinterCallback)
        trainer.train()

        trained_model = trainer.model
        trained_model.chronos_config.context_length = max(trained_model.chronos_config.context_length, seq_len)
        trained_model.chronos_config.max_output_patches = max(
            trained_model.chronos_config.max_output_patches,
            (pred_len + self.model_output_patch_size - 1) // self.model_output_patch_size,
        )
        if hasattr(trained_model, "config"):
            trained_model.config.chronos_config = trained_model.chronos_config.__dict__
            trained_model.config.echo_config = model.echo_config.__dict__
            trained_model.config.chronos_pipeline_class = "Chronos2EchoPipeline"
            trained_model.config.architectures = ["Chronos2EchoModel"]

        if model.echo_config.guard_against_baseline and eval_dataset is not None:
            self._guard_against_baseline(trained_model, eval_dataset)

        finetuned_pipeline = Chronos2EchoPipeline(model=trained_model)  # type: ignore[arg-type]
        finetuned_path = output_dir / finetuned_ckpt_name
        finetuned_pipeline.save_pretrained(finetuned_path)
        return finetuned_pipeline

    @staticmethod
    def _unwrap_echo_model(model: torch.nn.Module) -> Chronos2EchoModel:
        return model.get_base_model() if hasattr(model, "get_base_model") else model  # type: ignore[return-value]

    @torch.no_grad()
    def _dataset_rmse(self, model: torch.nn.Module, dataset: TimeMMDBatchDataset, *, force_base_prediction: bool) -> float:
        base_model = self._unwrap_echo_model(model)
        device = base_model.device
        loader = DataLoader(dataset, batch_size=None, pin_memory=device.type == "cuda")
        errors = []
        model.eval()
        for batch in loader:
            future_target = batch["future_target"].to(device)
            model_inputs = {}
            for key, value in batch.items():
                if key in {"future_target", "target_idx_ranges"}:
                    continue
                model_inputs[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            output = model(**model_inputs, force_base_prediction=force_base_prediction)
            median_idx = min(range(len(base_model.chronos_config.quantiles)), key=lambda idx: abs(base_model.chronos_config.quantiles[idx] - 0.5))
            prediction = output.quantile_preds[:, median_idx, : future_target.shape[-1]]
            mask = torch.isfinite(future_target)
            errors.append((prediction[mask] - future_target[mask]).pow(2))
        return torch.cat(errors).mean().sqrt().item()

    def _guard_against_baseline(self, model: torch.nn.Module, eval_dataset: TimeMMDBatchDataset) -> None:
        echo_rmse = self._dataset_rmse(model, eval_dataset, force_base_prediction=False)
        base_rmse = self._dataset_rmse(model, eval_dataset, force_base_prediction=True)
        if echo_rmse > base_rmse:
            warnings.warn(
                f"Echo validation RMSE ({echo_rmse:.6f}) exceeded baseline RMSE ({base_rmse:.6f}); "
                "resetting residual_scale to preserve baseline behavior.",
                category=UserWarning,
                stacklevel=2,
            )
            self._unwrap_echo_model(model).echo.residual_scale.data.zero_()

    @torch.no_grad()
    def predict_timemmd(
        self,
        *,
        root_path: str | Path,
        data_path: str,
        target: str = "OT",
        seq_len: int,
        pred_len: int,
        features: str = "S",
        batch_size: int = 128,
        flag: str = "test",
        image_column: str | None = None,
        image_root_path: str | Path | None = None,
        image_size: int | None = None,
        tokenizer=None,
        return_base: bool = False,
        text_column: str = "fact",
        missing_text: str = "error",
    ) -> dict[str, torch.Tensor | list[float]]:
        if tokenizer is None:
            tokenizer = create_timemmd_tokenizer(
                self.model.echo_config.text_tokenizer_name_or_path or self.model.echo_config.text_model_name_or_path,
            )
        window_dataset, dataset = self._create_timemmd_dataset(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            seq_len=seq_len,
            pred_len=pred_len,
            target=target,
            features=features,
            batch_size=batch_size,
            shuffle=False,
            repeat=False,
            image_column=image_column,
            image_root_path=image_root_path,
            image_size=image_size,
            echo_config=self.model.echo_config,
            tokenizer=tokenizer,
            text_column=text_column,
            missing_text=missing_text,
        )
        loader = DataLoader(dataset, batch_size=None, pin_memory=self.model.device.type == "cuda")

        quantile_windows = []
        base_quantile_windows = []
        target_windows = []
        self.model.eval()
        for batch in loader:
            target_idx_ranges = batch.pop("target_idx_ranges")
            future_target = batch["future_target"]
            model_inputs = {}
            for key, value in batch.items():
                if key == "future_target":
                    continue
                model_inputs[key] = value.to(self.model.device) if isinstance(value, torch.Tensor) else value

            output = self.model(**model_inputs)
            quantile_preds = output.quantile_preds[..., :pred_len].cpu()
            base_quantile_preds = output.base_quantile_preds[..., :pred_len].cpu()
            for start, end in target_idx_ranges:
                quantile_windows.append(quantile_preds[start:end].permute(2, 0, 1))
                base_quantile_windows.append(base_quantile_preds[start:end].permute(2, 0, 1))
                target_windows.append(future_target[start:end].permute(1, 0))

        quantiles = torch.stack(quantile_windows, dim=0)
        base_quantiles = torch.stack(base_quantile_windows, dim=0)
        targets = torch.stack(target_windows, dim=0)
        target_columns = list(range(targets.shape[-1]))

        def inverse_targets(values: torch.Tensor) -> torch.Tensor:
            array = values.numpy()
            original_shape = array.shape
            array = window_dataset.inverse_transform(array.reshape(-1, original_shape[-1]), target_columns)
            return torch.from_numpy(array.reshape(original_shape)).to(values.dtype)

        def inverse_quantiles(values: torch.Tensor) -> torch.Tensor:
            array = values.numpy()
            w, h, n, q = array.shape
            array = array.transpose(0, 1, 3, 2).reshape(-1, n)
            array = window_dataset.inverse_transform(array, target_columns)
            array = array.reshape(w, h, q, n).transpose(0, 1, 3, 2)
            return torch.from_numpy(array).to(values.dtype)

        quantiles = inverse_quantiles(quantiles)
        base_quantiles = inverse_quantiles(base_quantiles)
        targets = inverse_targets(targets)
        median_idx = min(range(len(self.quantiles)), key=lambda idx: abs(self.quantiles[idx] - 0.5))
        predictions = quantiles[..., median_idx]
        output = {
            "quantiles": quantiles,
            "predictions": predictions,
            "targets": targets,
            "quantile_levels": self.quantiles,
        }
        if return_base:
            output["base_quantiles"] = base_quantiles
            output["base_predictions"] = base_quantiles[..., median_idx]
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, echo_config: Chronos2EchoConfig | dict | None = None, **kwargs):
        if str(pretrained_model_name_or_path).startswith("s3://"):
            return BaseChronosPipeline.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

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
            model.reset_echo_safety_parameters()
        return cls(model=model)
