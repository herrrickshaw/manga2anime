"""Geometry-level checks of the drawing routines using a recording ImageDraw stand-in (no rasterisation needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "original"))
import render_lantern_oracle as r  # noqa: E402

S, TAU = r.S, math.tau


class Rec:
    """Records every drawing call as (name, args, kwargs)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))

    def of(self, name):
        return [(a, k) for n, a, k in self.calls if n == name]


def _canvas():
    return Image.new("RGB", (r.W * S, r.H * S), (0, 0, 0))


# ---------- seeded tables are reproducible (rng seeds are part of the output) ----------
def test_seeded_tables_are_pinned():
    assert r.STARS[0] == pytest.approx((440.06109862698685, 609.0948461933802, 1.5308228282942324, 1.4150185072200883))
    assert r.STARS[109][:3] == pytest.approx((329.8045715561009, 446.21662224194, 1.2608963415219019))
    assert r.MOTES[0] == pytest.approx((0.5932283931856456, -16.66818688636414, 1.2286717304403094, 2.7816435925457763))


def test_make_sigil_depends_on_its_seed():
    assert r.make_sigil(100) != r.make_sigil(101)
    assert r.SIGILS[3] == r.make_sigil(103)


# ---------- helpers ----------
def test_bezier_matches_cubic_formula_for_generic_points():
    p0, p1, p2, p3 = (1.0, 2.0), (7.0, -3.0), (-4.0, 9.0), (10.0, 5.0)
    n = 8
    c = r.bez(p0, p1, p2, p3, n=n)
    assert len(c) == n + 1
    for i, (x, y) in enumerate(c):
        t = i / n
        ex = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t * t * p2[0] + t**3 * p3[0]
        ey = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t * t * p2[1] + t**3 * p3[1]
        assert (x, y) == pytest.approx((ex, ey))


def test_bezier_default_resolution_is_40_segments():
    assert len(r.bez((0, 0), (1, 1), (2, 2), (3, 3))) == 41


def test_font_order_prefers_didot_then_supplemental_georgia(monkeypatch):
    chosen = []
    monkeypatch.setattr(r.ImageFont, "truetype", lambda f, size: chosen.append(f) or f)

    class P:
        def __init__(self, f):
            self.f = str(f)

        def exists(self):
            return True          # every candidate exists: the first in the list must win
    monkeypatch.setattr(r, "Path", P)
    assert r.font(3) == "/System/Library/Fonts/Supplemental/Didot.ttc"


def test_gradient_blends_between_the_three_stops():
    px, h = r.BG.load(), r.H * r.S
    assert px[0, int(0.275 * h)] == pytest.approx((44, 18, 71), abs=2)    # halfway top -> mid
    assert px[0, int(0.775 * h)] == pytest.approx((45, 48, 94), abs=2)    # halfway mid -> bottom
    assert px[0, int(0.55 * h) - 3][2] == pytest.approx(96, abs=1)


# ---------- stars ----------
def test_star_radius_is_scaled_by_twinkle(monkeypatch):
    monkeypatch.setattr(r, "STARS", [(100.0, 100.0, 10.0, 0.0)])
    img = _canvas()
    r.draw_stars(ImageDraw.Draw(img), 0.0)               # a = 0.5 -> rr = 10 * (0.7 + 0.25) = 9.5
    ys, xs = np.nonzero(np.asarray(img).sum(axis=2))
    assert (xs.max() - xs.min() + 1) / S == pytest.approx(2 * 9.5, abs=1.0 / S + 0.01)
    assert (ys.max() - ys.min() + 1) / S == pytest.approx(2 * 9.5, abs=1.0 / S + 0.01)
    img = _canvas()
    r.draw_stars(ImageDraw.Draw(img), 0.125)             # sin(pi/2) = 1 -> a = 1 -> rr = 12
    ys, xs = np.nonzero(np.asarray(img).sum(axis=2))
    assert (xs.max() - xs.min() + 1) / S == pytest.approx(24, abs=1.0)
    assert img.getpixel((100 * S, 100 * S)) == (255, 255, 255)


# ---------- astrolabe ----------
def test_astrolabe_rings_and_ticks_geometry():
    d = Rec()
    r.draw_astrolabe(d, 0.0)
    ells = d.of("ellipse")
    cx, cy = r.HALO_C
    radii = [(b[2] - b[0]) / (2 * S) for (b,), k in ells[:3]]
    assert radii == [250, 205, 300]
    assert [k["outline"] for _, k in ells[:3]] == [(120, 96, 170), (120, 96, 170), (90, 70, 140)]
    lines = d.of("line")
    ticks, ecliptic = lines[:-1], lines[-1]
    assert len(ticks) == 72 + 36 + 24
    # per ring, long ticks (every 6th) are 10 long and the rest 5 long
    start = 0
    for rad, n in [(250, 72), (205, 36), (300, 24)]:
        for k, ((pts,), kw) in enumerate(ticks[start:start + n]):
            (x0, y0), (x1, y1) = pts
            assert math.hypot(x0 - x1, y0 - y1) / S == pytest.approx(10 if k % 6 == 0 else 5, abs=1e-6)
            assert math.hypot(x0 / S - cx, y0 / S - cy) == pytest.approx(rad)
        start += n


@pytest.mark.parametrize("ring_start,n,speed,rad", [(0, 72, 1, 250), (72, 36, -1, 205), (108, 24, 0, 300)])
def test_astrolabe_ring_rotation_direction(ring_start, n, speed, rad):
    t = 0.5
    d = Rec()
    r.draw_astrolabe(d, t)
    (pts,), _ = d.of("line")[ring_start]
    x0, y0 = pts[0]
    ang = math.atan2(y0 / S - r.HALO_C[1], x0 / S - r.HALO_C[0])
    want = TAU * t * speed / 6 if speed else 0.0
    assert math.cos(ang) == pytest.approx(math.cos(want), abs=1e-6)
    assert math.sin(ang) == pytest.approx(math.sin(want), abs=1e-6)


@pytest.mark.parametrize("t", [0.0, 0.5, 0.9])
def test_astrolabe_planet_rides_the_tilted_ecliptic(t):
    cx, cy = r.HALO_C
    img = _canvas()
    r.draw_astrolabe(ImageDraw.Draw(img), t)
    k = int((t % 1) * 120) % 120
    a = np.linspace(0, TAU, 120)[k]
    ex, ey = cx + 270 * math.cos(a), cy + 90 * math.sin(a)
    tilt = math.radians(-18)
    px = cx + (ex - cx) * math.cos(tilt) - (ey - cy) * math.sin(tilt)
    py = cy + (ex - cx) * math.sin(tilt) + (ey - cy) * math.cos(tilt)
    assert img.getpixel((round(px * S), round(py * S))) == r.TEAL
    assert img.getpixel((round(px * S), round((py + 12) * S))) != r.TEAL


# ---------- halo ----------
def test_halo_wedges_rotate_alternate_colours_and_leave_gaps():
    t = 0.3
    d = Rec()
    r.draw_halo(d, t)
    wedges = d.of("pieslice")
    assert len(wedges) == 24
    rot = TAU * t / 6 * 0.5
    for k, ((box, a0, a1), kw) in enumerate(wedges):
        assert a0 == pytest.approx(math.degrees(rot + TAU * k / 24))
        assert a1 - a0 == pytest.approx(360 / 24 - 2)
        assert kw["fill"] == ((226, 150, 70) if k % 2 else (176, 70, 110))
        assert box == [r.s(360 - 128), r.s(300 - 128), r.s(360 + 128), r.s(300 + 128)]
    assert [k["fill"] for _, k in d.of("ellipse")[:3]] == [(64, 36, 104), r.GOLD_D, (90, 46, 120)]


# ---------- sigils ----------
def test_sigil_positions_colours_and_glyph_strokes():
    t = 0.2
    d = Rec()
    r.draw_sigils(d, t)
    cx, cy = r.HALO_C
    lines, ells = d.of("line"), d.of("ellipse")
    li = ei = 0
    for i, (strokes, ring, dots) in enumerate(r.SIGILS):
        a = -TAU * t / 6 + TAU * i / 12
        x, y = cx + 228 * math.cos(a), cy + 228 * math.sin(a)
        glow = 0.5 + 0.5 * math.sin(TAU * t * 2 - i * 0.6)
        col = tuple(int(c * (0.55 + 0.45 * glow)) for c in r.LIGHT)
        if ring:
            (box,), kw = ells[ei]
            ei += 1
            assert box == pytest.approx([r.s(x - 12.6), r.s(y - 12.6), r.s(x + 12.6), r.s(y + 12.6)])
            assert kw["outline"] == col and kw["width"] == r.s(1)
        for (ax, ay), (bx, by) in strokes:
            (pts,), kw = lines[li]
            li += 1
            assert np.allclose(pts, [(r.s(x + ax * 9), r.s(y + ay * 9)), (r.s(x + bx * 9), r.s(y + by * 9))])
            assert kw["fill"] == col and kw["width"] == r.s(2)
        for k in range(dots):
            (box,), kw = ells[ei]
            ei += 1
            dx = x + (k - (dots - 1) / 2) * 6
            assert box == pytest.approx([r.s(dx - 1.5), r.s(y + 9 + 4 - 1.5), r.s(dx + 1.5), r.s(y + 9 + 4 + 1.5)])
            assert kw["fill"] == col
    assert li == len(lines) and ei == len(ells)


def test_some_sigil_has_a_ring_and_some_do_not():
    assert {ring for _, ring, _ in r.SIGILS} == {True, False}
    assert {dots for _, _, dots in r.SIGILS} >= {0, 1, 2}


# ---------- hair / figure ----------
def test_hair_strand_reach_varies_with_index_mod_five():
    t = 0.1
    strands = r.hair_strands(t)
    for i, st in enumerate(strands):
        ph = i * 0.7
        a = math.radians(200 + i * 9)
        x0, y0 = 360 + 40 * math.cos(a), 300 + 46 * math.sin(a) - 8
        reach = 170 + 14 * (i % 5)
        w2 = math.sin(TAU * t * 2 + ph * 1.3)
        w = math.sin(TAU * t + ph)
        assert st[-1] == pytest.approx((x0 - reach + 18 * w, y0 + 150 + i * 6 + 20 * w2))


def test_figure_strand_pen_styles():
    d = Rec()
    r.draw_figure(d, 0.0)
    strands = d.of("line")[:16]
    for i, ((pts,), kw) in enumerate(strands):
        assert kw["fill"] == (r.HAIR_L if i % 3 == 0 else r.HAIR)
        assert kw["width"] == r.s(7 if i % 2 else 9)
        assert len(pts) == 31


# ---------- lantern ----------
@pytest.mark.parametrize("t,radius", [(0.0, 79.0), (0.125, 88.0), (0.375, 70.0)])
def test_lantern_glow_radius_pulses_between_70_and_88(t, radius):
    img, glow = Rec(), Rec()
    r.draw_lantern(img, glow, 500, 214, t)
    (box,), kw = glow.of("ellipse")[0]
    assert (box[2] - box[0]) / (2 * S) == pytest.approx(radius)
    assert kw["fill"] == (150, 120, 60)


def test_lantern_has_twelve_rotating_rays_with_breathing_length():
    t = 0.07
    img, glow = Rec(), Rec()
    r.draw_lantern(img, glow, 500, 214, t)
    rays = glow.of("line")
    assert len(rays) == 12
    for k, ((pts,), kw) in enumerate(rays):
        a = TAU * k / 12 + TAU * t / 6
        l = 40 + 14 * math.sin(TAU * t * 2 + k)
        (x0, y0), (x1, y1) = pts
        assert (x0 / S, y0 / S) == pytest.approx((500 + 22 * math.cos(a), 220 + 22 * math.sin(a)))
        assert (x1 / S, y1 / S) == pytest.approx((500 + (22 + l) * math.cos(a), 220 + (22 + l) * math.sin(a)))
        assert kw["fill"] == (200, 170, 90) and kw["width"] == r.s(2)


# ---------- frame ----------
def test_frame_arch_crown_and_bottom_inner_edge():
    img = _canvas()
    r.draw_frame(ImageDraw.Draw(img))
    crown_y = 150 - 120                       # arch_top - 120 at the centre line
    assert img.getpixel((r.W // 2 * S, crown_y * S)) == r.GOLD
    assert img.getpixel((r.W // 2 * S, (crown_y + 12) * S)) == r.GOLD_D   # inner arch line, 12 px inside
    assert img.getpixel((r.W // 2 * S, (r.H - 22 - 12) * S)) == r.GOLD_D  # horizontal inner bottom edge
    assert img.getpixel((r.W // 2 * S, (crown_y - 10) * S)) == (0, 0, 0)  # nothing above the arch


def test_cartouche_outline_is_three_px_wide():
    img = _canvas()
    r.draw_frame(ImageDraw.Draw(img))
    cy = (r.H - 70) * S
    assert img.getpixel((170 * S + 5, cy)) == r.GOLD
    assert img.getpixel((170 * S + 8, cy)) == r.INK


# ---------- render pipeline wiring ----------
def test_render_blurs_glow_with_fourteen_px_radius_and_screens_it(monkeypatch):
    seen = {}
    real_blur, real_screen = r.ImageFilter.GaussianBlur, r.ImageChops.screen
    monkeypatch.setattr(r.ImageFilter, "GaussianBlur", lambda rad: seen.__setitem__("blur", rad) or real_blur(rad))
    monkeypatch.setattr(r.ImageChops, "screen", lambda a, b: seen.__setitem__("screen", True) or real_screen(a, b))
    out = r.render(0.0)
    assert seen == {"blur": r.s(14), "screen": True} and out.size == (r.W, r.H)


def test_render_pastes_ink_outside_the_arch():
    img = r.render(0.25)
    assert img.getpixel((2, 2)) == r.INK and img.getpixel((r.W - 3, r.H - 3)) == r.INK


def test_main_poster_is_frame_n_over_four(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "N", 10)           # 10 // 4 == 2 but 10 // 3 == 3
    monkeypatch.setattr(r, "OUT", tmp_path / "o.mp4")
    monkeypatch.setattr(r, "POSTER", tmp_path / "p.png")
    monkeypatch.setattr(r, "render", lambda t: Image.new("RGB", (2, 2), (round(t * 10), 0, 0)))
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: None)
    r.main()
    assert Image.open(tmp_path / "p.png").getpixel((0, 0))[0] == 2
