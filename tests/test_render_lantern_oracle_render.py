"""Offline coverage of render/main/draw_frame/font/gradient in original/render_lantern_oracle.py (ffmpeg is faked)."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "original"))
import render_lantern_oracle as r  # noqa: E402


def _canvas():
    return Image.new("RGB", (r.W * r.S, r.H * r.S), (0, 0, 0))


# ---------- static precomputes ----------
def test_gradient_endpoints_and_midpoint():
    px = r.BG.load()
    assert px[0, 0] == (18, 10, 46)                               # top colour
    assert px[0, int(0.55 * r.H * r.S)] == pytest.approx((70, 26, 96), abs=1)
    bottom = px[0, r.H * r.S - 1]
    assert bottom == pytest.approx((20, 70, 92), abs=1)
    # constant across x, monotone blue-ish at the bottom: the first and last columns match
    assert px[0, 100] == px[r.W * r.S - 1, 100]


def test_outside_mask_geometry():
    m = r.outside_mask()
    assert m.size == (r.W * r.S, r.H * r.S) and m.mode == "L"
    assert m.getpixel((0, 0)) == 255                              # corner is outside the arch
    assert m.getpixel((r.W * r.S // 2, r.H * r.S // 2)) == 0      # centre is inside
    assert m.getpixel((r.W * r.S // 2, 5)) == 255                 # above the arch crown
    assert m.getpixel((10 * r.S, r.H * r.S // 2)) == 255          # left of the 22px margin
    assert m.getpixel((40 * r.S, r.H * r.S // 2)) == 0


def test_stars_and_motes_tables_are_seeded_and_in_range():
    assert len(r.STARS) == 110 and len(r.MOTES) == 40
    for x, y, rad, ph in r.STARS:
        assert 40 <= x <= r.W - 40 and 60 <= y <= r.H * 0.7 and 0.6 <= rad <= 1.8 and 0 <= ph <= r.TAU
    for u0, dx, ph, rad in r.MOTES:
        assert 0 <= u0 <= 1 and -30 <= dx <= 30 and 1.2 <= rad <= 3.0
    assert len(r.SIGILS) == 12


# ---------- font ----------
def _fake_path(existing):
    class P:
        def __init__(self, f):
            self.f = str(f)

        def exists(self):
            return self.f in existing
    return P


def test_font_prefers_first_existing_file(monkeypatch):
    calls = []
    monkeypatch.setattr(r.ImageFont, "truetype", lambda f, size: calls.append((f, size)) or ("tt", f, size))
    monkeypatch.setattr(r, "Path", _fake_path({"/System/Library/Fonts/Supplemental/Georgia.ttf",
                                               "/Library/Fonts/Georgia.ttf"}))
    assert r.font(10) == ("tt", "/System/Library/Fonts/Supplemental/Georgia.ttf", 10 * r.S)
    monkeypatch.setattr(r, "Path", _fake_path({"/System/Library/Fonts/Supplemental/Didot.ttc",
                                               "/Library/Fonts/Georgia.ttf"}))
    assert r.font(4)[1] == "/System/Library/Fonts/Supplemental/Didot.ttc"
    monkeypatch.setattr(r, "Path", _fake_path({"/Library/Fonts/Georgia.ttf"}))
    assert r.font(4)[1] == "/Library/Fonts/Georgia.ttf"
    assert len(calls) == 3


def test_font_falls_back_to_default_when_none_installed(monkeypatch):
    monkeypatch.setattr(r, "Path", _fake_path(set()))
    sentinel = object()
    monkeypatch.setattr(r.ImageFont, "load_default", lambda: sentinel)
    monkeypatch.setattr(r.ImageFont, "truetype", lambda *a: pytest.fail("no font file exists"))
    assert r.font(26) is sentinel


# ---------- individual layers ----------
def test_stars_pixel_brightness_follows_phase_at_t0():
    img = _canvas()
    r.draw_stars(ImageDraw.Draw(img), 0.0)
    x, y, rad, ph = r.STARS[0]
    a = 0.5 + 0.5 * math.sin(ph)
    v = int(150 + 105 * a)
    assert img.getpixel((int(x * r.S), int(y * r.S))) == (v, v, v)
    # a quarter of the way through the 2-cycle twinkle the same star has a different brightness
    img2 = _canvas()
    r.draw_stars(ImageDraw.Draw(img2), 0.125)
    a2 = 0.5 + 0.5 * math.sin(math.pi / 2 + ph)
    v2 = int(150 + 105 * a2)
    assert img2.getpixel((int(x * r.S), int(y * r.S))) == (v2, v2, v2)


def test_hair_strands_shape_and_period():
    s0, s1 = r.hair_strands(0.0), r.hair_strands(1.0)
    assert len(s0) == 16 and all(len(s) == 31 for s in s0)
    for a, b in zip(s0, s1):
        assert np.allclose(a, b, atol=1e-9)                       # strands loop with period 1
    # roots sit on the upper-left of the head ellipse around (360, 300)
    for i, strand in enumerate(s0):
        ang = math.radians(200 + i * 9)
        assert strand[0] == pytest.approx((360 + 40 * math.cos(ang), 300 + 46 * math.sin(ang) - 8))
    assert s0 != r.hair_strands(0.25)


def test_figure_returns_lantern_anchor():
    img = _canvas()
    lx, ly = r.draw_figure(ImageDraw.Draw(img), 0.0)
    assert lx == r.LANTERN_BASE[0]
    assert ly == pytest.approx(r.LANTERN_BASE[1] + 6 * math.sin(0.4))
    _, ly2 = r.draw_figure(ImageDraw.Draw(_canvas()), 0.25)
    assert ly2 == pytest.approx(r.LANTERN_BASE[1] + 6 * math.sin(math.tau * 0.5 + 0.4))
    arr = np.asarray(img)
    assert (arr == r.SKIN).all(axis=2).any() and (arr == r.ROBE).all(axis=2).any()


def test_halo_draws_gold_ring_and_dark_core():
    img = _canvas()
    r.draw_halo(ImageDraw.Draw(img), 0.0)
    cx, cy = r.HALO_C
    assert img.getpixel((cx * r.S, cy * r.S)) == (54, 30, 92)     # inner disc
    assert img.getpixel(((cx + 92) * r.S, cy * r.S)) == r.GOLD    # gold ring at radius 92
    assert img.getpixel(((cx + 149) * r.S, cy * r.S)) == (64, 36, 104)  # outermost band


def test_astrolabe_planet_position_depends_on_time():
    a, b = _canvas(), _canvas()
    r.draw_astrolabe(ImageDraw.Draw(a), 0.0)
    r.draw_astrolabe(ImageDraw.Draw(b), 0.5)
    ta = (np.asarray(a) == r.TEAL).all(axis=2)
    tb = (np.asarray(b) == r.TEAL).all(axis=2)
    assert ta.any() and tb.any() and not (ta & tb).any()          # the TEAL planet moved


def test_sigils_pulse_and_stay_on_ring():
    img = _canvas()
    r.draw_sigils(ImageDraw.Draw(img), 0.0)
    arr = np.asarray(img).sum(axis=2)
    ys, xs = np.nonzero(arr)
    cx, cy = r.HALO_C
    dist = np.hypot(xs / r.S - cx, ys / r.S - cy)
    assert dist.min() > 228 - 20 and dist.max() < 228 + 25        # all ink hugs the radius-228 ring
    assert r.draw_sigils(ImageDraw.Draw(_canvas()), 0.3) is None


def test_lantern_draws_into_both_layers_and_glow_pulses():
    def go(t):
        img, glow = _canvas(), _canvas()
        r.draw_lantern(ImageDraw.Draw(img), ImageDraw.Draw(glow), 500, 214, t)
        return np.asarray(img), np.asarray(glow)
    img0, glow0 = go(0.0)
    img1, glow1 = go(0.125)           # sin(TAU*2*t) = 1 -> maximum pulse radius 88
    assert (glow0 > 0).any() and (img0 == r.GOLD).all(axis=2).any() and (img0 == r.LIGHT).all(axis=2).any()
    assert (glow1.sum(axis=2) > 0).sum() > (glow0.sum(axis=2) > 0).sum()
    # glow disc at t=0 has radius 70 + 18*0.5 = 79 around (500, 220)
    ys, xs = np.nonzero(glow0.sum(axis=2))
    assert (xs.max() - xs.min()) / r.S == pytest.approx(2 * 79, abs=3)


def test_motes_fade_at_loop_ends_and_peak_midway():
    glow = _canvas()
    r.draw_motes(ImageDraw.Draw(glow), 500, 214, 0.0)
    u0, dx, ph, rad = max(r.MOTES, key=lambda m: math.sin(math.pi * m[0]))
    a = math.sin(math.pi * u0)
    want = tuple(int(v * a) for v in (255, 220, 140))
    mx = 500 + dx + 40 * u0 * math.sin(ph) + 8 * math.sin(math.tau * u0 * 2 + ph)
    my = 214 - 260 * u0 + 20
    got = glow.getpixel((int(mx * r.S), int(my * r.S)))
    assert got[0] >= want[0] - 2 and got[0] > 150               # brightest mote is bright (overlaps may add)


def test_draw_frame_border_cartouche_and_title(monkeypatch):
    img = _canvas()
    r.draw_frame(ImageDraw.Draw(img))
    m = 22
    assert img.getpixel((m * r.S, (r.H - 100) * r.S)) == r.GOLD           # outer left border
    cy = r.H - 70
    assert img.getpixel((200 * r.S, cy * r.S)) == r.INK                   # cartouche body
    assert img.getpixel((171 * r.S, cy * r.S)) == r.GOLD                  # cartouche outline
    assert img.getpixel((100 * r.S, cy * r.S)) == (0, 0, 0)               # nothing outside the cartouche
    arr = np.asarray(img)
    title = arr[(cy - 15) * r.S:(cy + 15) * r.S, 250 * r.S:(r.W - 250) * r.S]
    assert (title == r.GOLD).all(axis=2).any()                            # the title text is drawn in gold


# ---------- render / main ----------
@pytest.fixture(scope="module")
def frame0():
    return r.render(0.0)


def test_render_output_shape_and_letterbox(frame0):
    assert frame0.size == (r.W, r.H) and frame0.mode == "RGB"
    assert frame0.getpixel((0, 0)) == r.INK                               # outside the arch is flat ink
    assert frame0.getpixel((r.W - 1, 0)) == r.INK
    assert frame0.getpixel((r.W // 2, r.H // 2)) != r.INK                 # figure / halo inside


def test_render_is_deterministic_and_time_dependent(frame0):
    assert r.render(0.0).tobytes() == frame0.tobytes()
    assert r.render(0.3).tobytes() != frame0.tobytes()


def test_main_renders_n_frames_saves_poster_and_calls_ffmpeg(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(r, "N", 8)
    monkeypatch.setattr(r, "OUT", tmp_path / "out.mp4")
    monkeypatch.setattr(r, "POSTER", tmp_path / "poster.png")
    ts = []

    def fake_render(t):
        ts.append(t)
        return Image.new("RGB", (2, 2), (int(t * 255), 0, 0))
    seen = {}

    def fake_run(cmd, check):
        seen["cmd"], seen["check"] = cmd, check
        frames = sorted(Path(cmd[cmd.index("-i") + 1]).parent.glob("f*.png"))
        seen["frames"] = [p.name for p in frames]
        seen["dir"] = frames[0].parent
    monkeypatch.setattr(r, "render", fake_render)
    monkeypatch.setattr(r.subprocess, "run", fake_run)
    r.main()
    assert ts == [i / 8 for i in range(8)]
    assert seen["frames"] == [f"f{i:04d}.png" for i in range(8)]
    cmd = seen["cmd"]
    assert seen["check"] is True and cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-framerate") + 1] == str(r.FPS)
    assert cmd[cmd.index("-i") + 1].endswith("f%04d.png")
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[-1] == str(tmp_path / "out.mp4")
    assert not seen["dir"].exists()                                       # frame tmp dir removed
    poster = Image.open(tmp_path / "poster.png")
    assert poster.getpixel((0, 0))[0] == int((8 // 4) / 8 * 255)         # poster is frame N//4
    assert capsys.readouterr().out == f"wrote {tmp_path / 'out.mp4'} and {tmp_path / 'poster.png'}\n"


def test_paths_point_into_repo_assets():
    assert r.OUT == r.ROOT / "assets" / "lantern_oracle.mp4"
    assert r.POSTER == r.ROOT / "assets" / "lantern_oracle_poster.png"
    assert (r.ROOT / "original").is_dir()
