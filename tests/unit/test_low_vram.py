from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from kandinsky.core.components.text_embedder import Kandinsky6TextEmbedder
from kandinsky.pipeline import reconfigure
from kandinsky.runtime.offload.block import BlockOffload


def test_offload_wrapper_preserves_compiled_bound_forward_and_hooks():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = 3

        @torch.compile(backend=backend)
        def forward(self, value):
            return value + self.offset

    events = []

    @contextmanager
    def use_block(group, following):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    handle = BlockOffload.__new__(BlockOffload)
    handle._use_block = use_block
    block = Block()
    block.forward = handle._decorate(block.forward, object(), None)
    assert block(torch.ones(1)).item() == 4
    assert events == ["enter", "exit"]
    assert graphs, "Regional compile was disabled along with transfer orchestration"


def test_device_bound_text_encoder_release_and_reload(monkeypatch):
    encoder = Kandinsky6TextEmbedder.__new__(Kandinsky6TextEmbedder)
    encoder.qwen = object()
    encoder.clip = object()
    encoder._paths = ("qwen", "clip")
    encoder.max_length = 10
    encoder.device = torch.device("cpu")
    encoder.quantized_qwen = True
    encoder.text_token_padding = False
    encoder.release()
    assert encoder.qwen is encoder.clip is None
    loaded = []

    def init(self, *paths, **kwargs):
        loaded.append((paths, kwargs))
        self.qwen = object()
        self.clip = object()

    monkeypatch.setattr(Kandinsky6TextEmbedder, "__init__", init)
    encoder.ensure_loaded()
    encoder.ensure_loaded()
    assert len(loaded) == 1
    assert loaded[0][0] == ("qwen", "clip")
    assert loaded[0][1]["quantized_qwen"]


def test_profiler_does_not_keep_released_text_weights_alive():
    import gc
    import weakref

    from kandinsky.runtime.profile import NoOpProfile, TimeMemModuleProfile

    class Text:
        def __init__(self):
            self.qwen = self.clip = None
            self.ensure_loaded()

        def ensure_loaded(self):
            if self.qwen is None:
                self.qwen = torch.nn.Linear(1, 1)
                self.clip = torch.nn.Linear(1, 1)

        def release(self):
            self.qwen = self.clip = None

    text = Text()
    pipe = SimpleNamespace(text_embedder=text, release_text_encoder=True)
    timed = TimeMemModuleProfile()
    timed.apply(pipe)
    old_qwen = weakref.ref(text.qwen)
    pipe.profile = NoOpProfile()  # Same situation as benchmark warmup.
    text.release()
    gc.collect()
    assert old_qwen() is None, "Profiling hooks retained NF4 weights"
    text.ensure_loaded()
    assert "forward" in text.qwen.__dict__  # Reloaded weights are profiled too.
    text.release()
    timed.clear(pipe)


@pytest.mark.parametrize("warmup_enabled,expected_calls", [(False, 1), (True, 2)])
def test_cli_sr_no_grad_and_warmup_boolean(
    tmp_path, monkeypatch, warmup_enabled, expected_calls
):
    from kandinsky import cli
    from kandinsky_sr.core.utils.offload import NoOpOffload
    from kandinsky_sr.pipeline import components, warmup

    calls = []

    class SR:
        offload = NoOpOffload()
        dit = object()
        resolution_scale = 2

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            calls.append(kwargs["save_path"])
            return SimpleNamespace(
                frames=torch.zeros(1, 3, 1, 4, 4), path=kwargs["save_path"]
            )

    monkeypatch.setattr(cli, "load_sr_pipeline", lambda *args, **kwargs: SR())
    monkeypatch.setattr(components, "scale_factor_for", lambda _: (1, 1, 1))
    monkeypatch.setattr(warmup, "warmup", lambda *args: None)
    monkeypatch.setattr(warmup, "compile_vae_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(warmup, "clip_base_resolution", lambda *args: (2, 2))
    result = SimpleNamespace(frames=torch.zeros(1, 3, 1, 2, 2), audio=None)
    cli._super_resolve_if_enabled(
        "unused",
        "cpu",
        "block",
        result,
        tmp_path / "base.mp4",
        SimpleNamespace(note_sr=lambda _: None),
        fps=24,
        audio_fps=44100,
        warmup=warmup_enabled,
    )
    assert len(calls) == expected_calls


def test_reconfigure_does_not_register_releasable_nf4(monkeypatch):
    registered = []
    handle = SimpleNamespace(register=lambda name, module: registered.append(name))
    monkeypatch.setattr(reconfigure, "build_offload", lambda *args, **kwargs: handle)
    pipe = SimpleNamespace(
        offload=SimpleNamespace(strategy="none"),
        device=torch.device("cpu"),
        release_text_encoder=True,
        text_embedder=object(),
        dit=object(),
        vae=object(),
    )
    reconfigure._apply_offload(pipe, "block", True, force=True)
    assert registered == ["dit", "vae"]
