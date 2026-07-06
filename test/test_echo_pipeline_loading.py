import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_echo_pipeline_with_fakes(monkeypatch):
    package = types.ModuleType("chronos_echo")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "chronos_echo")]
    monkeypatch.setitem(sys.modules, "chronos_echo", package)

    config_module = types.ModuleType("chronos_echo.config")

    class Chronos2EchoConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config_module.Chronos2EchoConfig = Chronos2EchoConfig
    monkeypatch.setitem(sys.modules, "chronos_echo.config", config_module)

    echo_module = types.ModuleType("chronos_echo.echo")

    class Chronos2EchoModel:
        from_pretrained_called = False

        def __init__(self, config):
            self.config = config
            self.echo_config = SimpleNamespace()
            self.device = "cpu"
            self.reset_random_called = False
            self.backbones_loaded = False
            self.safety_reset = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_called = True
            raise AssertionError("base checkpoints must load through Chronos2Pipeline first")

        def to(self, device):
            self.device = device
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, state, strict):
            self.loaded_state = (state, strict)

        def reset_echo_residual_head_random(self):
            self.reset_random_called = True

        def load_pretrained_echo_backbones(self):
            self.backbones_loaded = True

        def reset_echo_safety_parameters(self, *, zero_residual_head=True):
            self.safety_reset = zero_residual_head

    echo_module.Chronos2EchoModel = Chronos2EchoModel
    monkeypatch.setitem(sys.modules, "chronos_echo.echo", echo_module)

    torch_module = types.ModuleType("torch")
    torch_module.nn = SimpleNamespace(Module=object)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    transformers_module = types.ModuleType("transformers")

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(chronos_config={})

    transformers_module.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    chronos_module = types.ModuleType("chronos")
    base_module = types.ModuleType("chronos.base")
    chronos2_module = types.ModuleType("chronos.chronos2")
    pipeline_module = types.ModuleType("chronos.chronos2.pipeline")

    class BaseChronosPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("s3 fallback not expected")

    class BaseModel:
        def __init__(self):
            self.config = SimpleNamespace(chronos_config={})
            self.device = "cuda"

        def state_dict(self):
            return {}

    class Chronos2Pipeline:
        dtypes = {}
        from_pretrained_calls = []

        def __init__(self, model):
            self.model = model

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_calls.append((args, kwargs))
            return cls(BaseModel())

    base_module.BaseChronosPipeline = BaseChronosPipeline
    pipeline_module.Chronos2Pipeline = Chronos2Pipeline
    monkeypatch.setitem(sys.modules, "chronos", chronos_module)
    monkeypatch.setitem(sys.modules, "chronos.base", base_module)
    monkeypatch.setitem(sys.modules, "chronos.chronos2", chronos2_module)
    monkeypatch.setitem(sys.modules, "chronos.chronos2.pipeline", pipeline_module)

    module_path = Path(__file__).resolve().parents[1] / "src" / "chronos_echo" / "echo_pipeline.py"
    spec = importlib.util.spec_from_file_location("chronos_echo.echo_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "chronos_echo.echo_pipeline", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, Chronos2EchoModel, Chronos2Pipeline


def test_base_checkpoint_loads_through_base_pipeline_before_echo_clone(monkeypatch):
    module, echo_model_cls, base_pipeline_cls = _load_echo_pipeline_with_fakes(monkeypatch)

    pipeline = module.Chronos2EchoPipeline.from_pretrained(
        "base-chronos",
        echo_config={"num_echo_layers": 1},
        device_map="cuda",
    )

    assert base_pipeline_cls.from_pretrained_calls == [(("base-chronos",), {"device_map": "cuda"})]
    assert echo_model_cls.from_pretrained_called is False
    assert pipeline.model.reset_random_called is True
    assert pipeline.model.backbones_loaded is True
    assert pipeline.model.safety_reset is False
