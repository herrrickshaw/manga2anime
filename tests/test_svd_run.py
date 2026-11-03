"""svd_run tests: the hosted Space and gradio_client are faked; no network, no weights."""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

import svd_run


@pytest.fixture
def fake_gradio(monkeypatch):
    mod = types.ModuleType("gradio_client")
    mod.handle_file = lambda p: {"path": p}
    mod.Client = mock.Mock(name="Client")
    monkeypatch.setitem(sys.modules, "gradio_client", mod)
    return mod


def test_parse_args_positional_and_default_motion(tmp_path):
    a = svd_run.parse_args(["in.png", "out.mp4"])
    assert a.src.name == "in.png" and a.out.name == "out.mp4" and a.motion == svd_run.DEFAULT_MOTION
    assert svd_run.parse_args(["in.png", "out.mp4", "60"]).motion == 60.0


def test_run_svd_calls_space_and_copies_clip(tmp_path, fake_gradio):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mp4")
    client = mock.Mock()
    client.predict.return_value = ({"video": str(clip)}, 42)
    factory = mock.Mock(return_value=client)

    seed = svd_run.run_svd(tmp_path / "in.png", tmp_path / "out.mp4", 80, client_factory=factory)

    factory.assert_called_once_with(svd_run.SPACE)
    client.predict.assert_called_once_with(
        {"path": str(tmp_path / "in.png")}, svd_run.SEED, False, 80, svd_run.FPS, api_name=svd_run.API_NAME
    )
    assert seed == 42
    assert (tmp_path / "out.mp4").read_bytes() == b"mp4"


def test_main_prints_out_and_seed(tmp_path, fake_gradio, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    fake_gradio.Client.return_value.predict.return_value = ({"video": str(clip)}, 7)
    svd_run.main([str(tmp_path / "in.png"), str(tmp_path / "out.mp4")])
    assert capsys.readouterr().out.strip() == f"{tmp_path / 'out.mp4'} seed 7"
