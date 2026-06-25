# Chronos-2-ECHO: Event-guided Calibration with Heterogeneous Observations for Chronos-2

Chronos-2-ECHO adds an Aurora-inspired Event-Guided Echo Adapter on top of
[Chronos-2](https://huggingface.co/amazon/chronos-2) for multimodal financial
time-series forecasting. This repository only contains the ECHO additions; the
base Chronos-2 model code is provided by the `chronos-forecasting` package.

## Install

```sh
pip install -e .
```

The distribution package is `chronos2-echo`; the Python import package is
`chronos_echo`. It depends on `chronos-forecasting>=2.2.2,<3`. Use the upstream
`chronos` package for plain Chronos-2 APIs.

## Usage

```python
import torch

from chronos_echo import Chronos2EchoConfig, Chronos2EchoPipeline

echo_config = Chronos2EchoConfig(
    num_echo_layers=1,
    num_text_tokens=8,
    num_vision_tokens=8,
)

pipeline = Chronos2EchoPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    echo_config=echo_config,
)

output = pipeline.model(
    context=torch.rand(2, 256),
    num_output_patches=2,
    text_input_ids=torch.ones(2, 32, dtype=torch.long),
    text_attention_mask=torch.ones(2, 32, dtype=torch.long),
    risk_features=torch.rand(2, 24, 1),
)

print(output.quantile_preds.shape)
```

## TimeMMD Helpers

`chronos_echo.timemmd` contains the TimeMMD dataset and batching helpers used by
ECHO fine-tuning:

```python
from chronos_echo.timemmd import TimeMMDWindowDataset, build_timemmd_batch
```

## Citation

If you use the base Chronos-2 model, please cite the Chronos-2 report:

```bibtex
@article{ansari2025chronos2,
  title        = {Chronos-2: From Univariate to Universal Forecasting},
  author       = {Abdul Fatir Ansari and Oleksandr Shchur and Jaris Küken and Andreas Auer and Boran Han and Pedro Mercado and Syama Sundar Rangapuram and Huibin Shen and Lorenzo Stella and Xiyuan Zhang and Mononito Goswami and Shubham Kapoor and Danielle C. Maddix and Pablo Guerron and Tony Hu and Junming Yin and Nick Erickson and Prateek Mutalik Desai and Hao Wang and Huzefa Rangwala and George Karypis and Yuyang Wang and Michael Bohlke-Schneider},
  journal      = {arXiv preprint arXiv:2510.15821},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.15821}
}
```

## License

This project is licensed under the Apache-2.0 License.
