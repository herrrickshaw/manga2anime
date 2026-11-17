"""Pure logic of original/render_lantern_oracle.py: Bezier sampling, supersample scaling, sigil determinism,
and the docstring's '6 s seamless loop' claim (frame at t=1 must equal frame at t=0)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "original"))
import render_lantern_oracle as r  # noqa: E402


def _layer(draw_fn, t):
    img = Image.new("RGB", (r.W * r.S, r.H * r.S), (0, 0, 0))
    draw_fn(ImageDraw.Draw(img), t)
    return np.asarray(img, dtype=int)


def _same_frame(a, b, max_px=300):
    """Equal up to rasterisation rounding: rotating by exactly one tick spacing moves float endpoints by ~1e-13,
    which can flip a few hundred edge pixels out of 2.76M. A real discontinuity changes thousands."""
    return int(np.any(a != b, axis=2).sum()) <= max_px


def test_bezier_hits_endpoints_and_midpoint():
    p0, p1, p2, p3 = (0, 0), (0, 10), (10, 10), (10, 0)
    c = r.bez(p0, p1, p2, p3, n=4)
    assert len(c) == 5 and c[0] == p0 and c[-1] == p3
    # B(0.5) = (p0 + 3p1 + 3p2 + p3) / 8
    assert c[2] == pytest.approx((5.0, 7.5))


def test_supersample_scaling():
    assert r.s(3) == 3 * r.S and r.pts([(1, 2)]) == [(r.S, 2 * r.S)]


def test_frame_count_and_gradient_size():
    assert r.N == r.FPS * r.SECONDS == 144
    assert r.BG.size == (r.W * r.S, r.H * r.S)


def test_sigils_are_deterministic_and_well_formed():
    a, b = r.make_sigil(100), r.make_sigil(100)
    assert a == b
    for strokes, ring, dots in r.SIGILS:
        assert 2 <= len(strokes) <= 4 and isinstance(ring, bool) and 0 <= dots <= 2
        for (ax, ay), (bx, by) in strokes:
            assert (ax, ay) != (bx, by) and {ax, ay, bx, by} <= {-1, 0, 1}


@pytest.mark.parametrize("name,fn", [("stars", r.draw_stars), ("astrolabe", r.draw_astrolabe), ("halo", r.draw_halo),
                                     ("figure", r.draw_figure)])
def test_periodic_layers_loop_seamlessly(name, fn):
    assert _same_frame(_layer(fn, 0.0), _layer(fn, 1.0)), name


def test_motes_loop_seamlessly():
    def f(d, t):
        r.draw_motes(d, 500, 214, t)
    assert _same_frame(_layer(f, 0.0), _layer(f, 1.0))


@pytest.mark.xfail(strict=True, reason="BUG: sigil ring turns 60 deg per loop (TAU*t/6) but the 12 sigils are distinct, "
                                       "so at the loop point every sigil jumps to its neighbour's shape")
def test_sigil_ring_loops_seamlessly():
    assert _same_frame(_layer(r.draw_sigils, 0.0), _layer(r.draw_sigils, 1.0))


@pytest.mark.xfail(strict=True, reason="BUG: lantern rays turn 30 deg per loop but ray k's length uses sin(..+k), "
                                       "so rays change length at the loop point")
def test_lantern_rays_loop_seamlessly():
    def f(d, t):
        r.draw_lantern(d, d, 500, 214, t)
    assert _same_frame(_layer(f, 0.0), _layer(f, 1.0))
