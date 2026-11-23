"""original/svd_run.py with the gradio_client Space faked (no network, no model)."""
import runpy
import sys
from pathlib import Path

import pytest

SCRIPT = str(Path(__file__).resolve().parent.parent / "original" / "svd_run.py")


@pytest.fixture()
def fake_gradio(tmp_path, monkeypatch):
    clip = tmp_path / "space_result.mp4"
    clip.write_bytes(b"CLIP")
    calls = {}

    class Client:
        def __init__(self, space):
            calls["space"] = space

        def predict(self, *args, **kw):
            calls["args"], calls["kw"] = args, kw
            return {"video": str(clip)}, 1234

    mod = type(sys)("gradio_client")
    mod.Client = Client
    mod.handle_file = lambda p: ("file", p)
    monkeypatch.setitem(sys.modules, "gradio_client", mod)
    return calls


def test_default_motion_and_copy(fake_gradio, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out.mp4"
    monkeypatch.setattr(sys, "argv", ["svd_run.py", "in.png", str(out)])
    runpy.run_path(SCRIPT, run_name="__main__")
    assert out.read_bytes() == b"CLIP"
    assert fake_gradio["space"] == "multimodalart/stable-video-diffusion"
    assert fake_gradio["args"] == (("file", "in.png"), 42, False, 100, 6)
    assert fake_gradio["kw"] == {"api_name": "/video"}
    assert capsys.readouterr().out == f"{out} seed 1234\n"


def test_explicit_motion_is_parsed_as_float(fake_gradio, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["svd_run.py", "in.png", str(tmp_path / "o.mp4"), "55.5"])
    runpy.run_path(SCRIPT, run_name="__main__")
    motion = fake_gradio["args"][3]
    assert motion == 55.5 and isinstance(motion, float)
