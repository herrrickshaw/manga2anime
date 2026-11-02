"""Unit tests for the pure helpers plus a one-frame smoke test (ffmpeg is mocked)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import pytest

import render_lantern_oracle as r


def test_scale_helpers_use_supersample_factor():
    assert r.s(10) == 10 * r.S
    assert r.pts([(1, 2), (3.5, 4)]) == [(r.S, 2 * r.S), (3.5 * r.S, 4 * r.S)]


def test_bez_endpoints_and_count():
    curve = r.bez((0, 0), (0, 10), (10, 10), (10, 0), n=8)
    assert len(curve) == 9
    assert curve[0] == (0, 0)
    assert curve[-1] == pytest.approx((10, 0))
    assert curve[4] == pytest.approx((5, 7.5))  # cubic midpoint


def test_make_sigil_is_deterministic_and_well_formed():
    a, b = r.make_sigil(123), r.make_sigil(123)
    assert a == b
    strokes, ring, dots = a
    assert 2 <= len(strokes) <= 4
    assert isinstance(ring, bool)
    assert 0 <= dots <= 2
    for (ax, ay), (bx, by) in strokes:
        assert {ax, ay, bx, by} <= {-1, 0, 1}
        assert (ax, ay) != (bx, by)


def test_sigils_and_scatter_sizes():
    assert len(r.sigils()) == r.N_SIGILS
    stars, motes = r.scatter()
    assert len(stars) == r.N_STARS and len(motes) == r.N_MOTES
    assert all(40 <= x <= r.W - 40 for x, _, _, _ in stars)


def test_hair_strands_shape_and_loop():
    strands = r.hair_strands(0.0)
    assert len(strands) == 16
    assert all(len(st) == 31 for st in strands)
    # phase 0 and phase 1 are the same frame (seamless loop)
    for p, q in zip(r.hair_strands(0.0)[3], r.hair_strands(1.0)[3]):
        assert p == pytest.approx(q)


def test_arch_outline_symmetric_and_closed_at_margin():
    o = r.arch_outline()
    assert o[0] == (r.ARCH_MARGIN, r.H - r.ARCH_MARGIN)
    assert o[-1] == (r.W - r.ARCH_MARGIN, r.H - r.ARCH_MARGIN)
    xs = [p[0] for p in o]
    assert min(xs) == pytest.approx(r.ARCH_MARGIN) and max(xs) == pytest.approx(r.W - r.ARCH_MARGIN)
    assert min(p[1] for p in o) == pytest.approx(r.ARCH_TOP - 120, abs=0.1)  # 80-sample arc


def test_render_single_frame_smoke():
    img = r.render(0.25)
    assert img.size == (r.W, r.H)
    assert img.mode == "RGB"
    # outside the arch is inked
    assert img.getpixel((2, 2)) == r.INK


def test_parse_args_defaults_and_overrides(tmp_path):
    a = r.parse_args([])
    assert a.out == r.OUT and a.poster == r.POSTER
    b = r.parse_args(["--out", str(tmp_path / "x.mp4"), "--poster", str(tmp_path / "p.png")])
    assert b.out == tmp_path / "x.mp4"


def test_main_writes_frames_and_calls_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "N", 4)  # keep the smoke test fast
    out, poster = tmp_path / "o.mp4", tmp_path / "p.png"
    seen: dict = {}

    def fake_run(cmd, check):
        seen["cmd"] = cmd
        frames_dir = Path(cmd[cmd.index("-i") + 1]).parent
        seen["frames"] = sorted(p.name for p in frames_dir.glob("f*.png"))

    with mock.patch.object(r.subprocess, "run", side_effect=fake_run):
        r.main(["--out", str(out), "--poster", str(poster)])
    assert seen["frames"] == [f"f{i:04d}.png" for i in range(4)]
    assert seen["cmd"][0] == "ffmpeg" and seen["cmd"][-1] == str(out)
    assert poster.exists()
    assert not Path(seen["cmd"][seen["cmd"].index("-i") + 1]).parent.exists()  # temp dir cleaned


def test_tau():
    assert r.TAU == pytest.approx(2 * math.pi)
