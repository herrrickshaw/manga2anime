"""Render "The Lantern Oracle" — an original Art Nouveau / tarot-style animated panel.

Original character and composition; style vocabulary only (Art Nouveau arch frame,
tarot-card layout, astrolabe geometry, procedural sigils). No source panels used.

Output: assets/lantern_oracle.mp4 (6 s seamless loop) + assets/lantern_oracle_poster.png
Run:    python3 original/render_lantern_oracle.py [--out PATH] [--poster PATH]

The scene is drawn procedurally with Pillow at ``S``x supersampling, downsampled with
Lanczos, and the PNG frame sequence is encoded with ``ffmpeg`` (must be on ``PATH``).
Every frame is a pure function of ``t`` in [0, 1) so the clip loops seamlessly.
"""

from __future__ import annotations

import argparse
import functools
import math
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ---------- configuration ----------
W, H, S = 720, 960, 2  # output size (px), supersample factor
FPS, SECONDS = 24, 6
N = FPS * SECONDS  # frames per loop
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "lantern_oracle.mp4"
POSTER = ROOT / "assets" / "lantern_oracle_poster.png"
TAU = math.pi * 2

SCATTER_SEED = 7  # stars + motes share one RNG stream
SIGIL_SEED_BASE = 100
N_STARS, N_MOTES, N_SIGILS = 110, 40, 12
ARCH_MARGIN, ARCH_TOP = 22, 150  # frame border margin and arch spring line (px)
TITLE = "THE LANTERN ORACLE"
TITLE_SIZE = 26
GLOW_BLUR = 14
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Didot.ttc",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/Library/Fonts/Georgia.ttf",
)

INK = (24, 14, 40)
GOLD = (232, 182, 78)
GOLD_D = (168, 112, 40)
ROBE = (196, 64, 92)
ROBE_D = (128, 34, 70)
SKIN = (236, 200, 172)
HAIR = (44, 30, 86)
HAIR_L = (92, 70, 168)
LIGHT = (255, 236, 170)
TEAL = (64, 200, 196)

HALO_C = (360, 300)
LANTERN_BASE = (500, 214)


# ---------- pure geometry helpers ----------
def s(v: float) -> float:
    """Scale a design-space value (px) to supersampled canvas units."""
    return v * S


def pts(seq: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Scale a sequence of design-space points to canvas units."""
    return [(s(x), s(y)) for x, y in seq]


def bez(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n: int = 40,
) -> list[tuple[float, float]]:
    """Sample a cubic Bezier curve at ``n + 1`` evenly spaced parameter values."""
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return out


def arch_outline(margin: float = ARCH_MARGIN, arch_top: float = ARCH_TOP) -> list[tuple[float, float]]:
    """Design-space outline of the arch-topped frame, bottom-left to bottom-right."""
    return (
        [(margin, H - margin), (margin, arch_top)]
        + [
            (W / 2 + (W / 2 - margin) * math.cos(a), arch_top - 120 * math.sin(a))
            for a in np.linspace(math.pi, 0, 80)
        ]
        + [(W - margin, arch_top), (W - margin, H - margin)]
    )


def make_sigil(seed: int) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], bool, int]:
    """Procedural sigil: (strokes on a 3x3 grid, has_ring, dot_count), deterministic per seed."""
    r = np.random.default_rng(seed)
    grid = [(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)]
    strokes = []
    for _ in range(r.integers(2, 5)):
        a, b = r.choice(9, 2, replace=False)
        strokes.append((grid[a], grid[b]))
    return strokes, bool(r.integers(0, 2)), int(r.integers(0, 3))


def hair_strands(t: float) -> list[list[tuple[float, float]]]:
    """Sixteen wind-blown hair strands (Bezier samples) for loop phase ``t`` in [0, 1)."""
    strands = []
    for i in range(16):
        ph = i * 0.7
        a = math.radians(200 + i * 9)  # roots around upper-left of head
        x0, y0 = 360 + 40 * math.cos(a), 300 + 46 * math.sin(a) - 8
        reach = 170 + 14 * (i % 5)
        w = math.sin(TAU * t + ph)
        w2 = math.sin(TAU * t * 2 + ph * 1.3)
        p1 = (x0 - 60, y0 - 40 + 20 * w)
        p2 = (x0 - reach * 0.7, y0 + 60 + 30 * w2)
        p3 = (x0 - reach + 18 * w, y0 + 150 + i * 6 + 20 * w2)
        strands.append(bez((x0, y0), p1, p2, p3, 30))
    return strands


# ---------- static precomputes (built lazily, once) ----------
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """First available serif font from ``FONT_CANDIDATES`` at ``size`` px (design space)."""
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, s(size))
    return ImageFont.load_default()


@functools.cache
def title_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return font(TITLE_SIZE)


@functools.cache
def background() -> Image.Image:
    """Vertical three-stop night-sky gradient at canvas resolution."""
    y = np.linspace(0, 1, H * S)[:, None]
    top = np.array([18, 10, 46])
    mid = np.array([70, 26, 96])
    bot = np.array([20, 70, 92])
    c = np.where(y < 0.55, top + (mid - top) * (y / 0.55), mid + (bot - mid) * ((y - 0.55) / 0.45))
    arr = np.repeat(c[:, None, :], W * S, axis=1).reshape(H * S, W * S, 3)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


@functools.cache
def scatter() -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Star field (x, y, radius, phase) and lantern motes (u0, dx, phase, radius) from one seeded RNG."""
    rng = np.random.default_rng(SCATTER_SEED)
    stars = [
        (rng.uniform(40, W - 40), rng.uniform(60, H * 0.7), rng.uniform(0.6, 1.8), rng.uniform(0, TAU))
        for _ in range(N_STARS)
    ]
    motes = [
        (rng.uniform(0, 1), rng.uniform(-30, 30), rng.uniform(0, TAU), rng.uniform(1.2, 3.0))
        for _ in range(N_MOTES)
    ]
    return stars, motes


@functools.cache
def sigils() -> list[tuple[list[tuple[tuple[int, int], tuple[int, int]]], bool, int]]:
    return [make_sigil(SIGIL_SEED_BASE + i) for i in range(N_SIGILS)]


@functools.cache
def outside_mask() -> Image.Image:
    """L-mode mask: 255 outside the arch (to be inked over), 0 inside."""
    m = Image.new("L", (W * S, H * S), 255)
    ImageDraw.Draw(m).polygon(pts(arch_outline()), fill=0)
    return m


# ---------- drawing ----------
def draw_stars(d: ImageDraw.ImageDraw, t: float) -> None:
    for x, y, r, ph in scatter()[0]:
        a = 0.5 + 0.5 * math.sin(TAU * t * 2 + ph)
        c = tuple(int(150 + 105 * a) for _ in range(3))
        rr = r * (0.7 + 0.5 * a)
        d.ellipse([s(x - rr), s(y - rr), s(x + rr), s(y + rr)], fill=c)


def draw_astrolabe(d: ImageDraw.ImageDraw, t: float) -> None:
    cx, cy = HALO_C
    for i, (rad, speed, ticks) in enumerate([(250, 1, 72), (205, -1, 36), (300, 0, 24)]):
        col = (120, 96, 170) if i != 2 else (90, 70, 140)
        d.ellipse([s(cx - rad), s(cy - rad), s(cx + rad), s(cy + rad)], outline=col, width=s(1))
        rot = TAU * t * speed / 6 if speed else 0
        for k in range(ticks):
            a = rot + TAU * k / ticks
            length = 10 if k % 6 == 0 else 5
            d.line(
                [
                    (s(cx + rad * math.cos(a)), s(cy + rad * math.sin(a))),
                    (s(cx + (rad - length) * math.cos(a)), s(cy + (rad - length) * math.sin(a))),
                ],
                fill=col,
                width=s(1),
            )
    # tilted ecliptic ellipse
    e = [(cx + 270 * math.cos(a), cy + 90 * math.sin(a)) for a in np.linspace(0, TAU, 120)]
    ang = math.radians(-18)
    e = [
        (
            cx + (x - cx) * math.cos(ang) - (y - cy) * math.sin(ang),
            cy + (x - cx) * math.sin(ang) + (y - cy) * math.cos(ang),
        )
        for x, y in e
    ]
    d.line(pts(e), fill=(110, 150, 190), width=s(1))
    # planet riding the ellipse
    k = int((t % 1) * len(e)) % len(e)
    px, py = e[k]
    d.ellipse([s(px - 5), s(py - 5), s(px + 5), s(py + 5)], fill=TEAL)


def draw_halo(d: ImageDraw.ImageDraw, t: float) -> None:
    cx, cy = HALO_C
    for rad, col in [(150, (64, 36, 104)), (138, GOLD_D), (132, (90, 46, 120))]:
        d.ellipse([s(cx - rad), s(cy - rad), s(cx + rad), s(cy + rad)], fill=col)
    # rotating mosaic wedges
    rot = TAU * t / 6 * 0.5
    for k in range(24):
        a0 = math.degrees(rot + TAU * k / 24)
        col = (226, 150, 70) if k % 2 else (176, 70, 110)
        d.pieslice([s(cx - 128), s(cy - 128), s(cx + 128), s(cy + 128)], a0, a0 + 360 / 24 - 2, fill=col)
    d.ellipse([s(cx - 96), s(cy - 96), s(cx + 96), s(cy + 96)], fill=(54, 30, 92))
    d.ellipse([s(cx - 92), s(cy - 92), s(cx + 92), s(cy + 92)], outline=GOLD, width=s(2))


def draw_sigils(d: ImageDraw.ImageDraw, t: float) -> None:
    cx, cy = HALO_C
    rot = -TAU * t / 6
    for i, (strokes, ring, dots) in enumerate(sigils()):
        a = rot + TAU * i / 12
        x, y = cx + 228 * math.cos(a), cy + 228 * math.sin(a)
        glow = 0.5 + 0.5 * math.sin(TAU * t * 2 - i * 0.6)
        col = tuple(int(c * (0.55 + 0.45 * glow)) for c in LIGHT)
        sz = 9
        if ring:
            d.ellipse(
                [s(x - sz * 1.4), s(y - sz * 1.4), s(x + sz * 1.4), s(y + sz * 1.4)], outline=col, width=s(1)
            )
        for (ax, ay), (bx, by) in strokes:
            d.line([(s(x + ax * sz), s(y + ay * sz)), (s(x + bx * sz), s(y + by * sz))], fill=col, width=s(2))
        for k in range(dots):
            dx = x + (k - (dots - 1) / 2) * 6
            d.ellipse([s(dx - 1.5), s(y + sz + 4 - 1.5), s(dx + 1.5), s(y + sz + 4 + 1.5)], fill=col)


def draw_figure(d: ImageDraw.ImageDraw, t: float) -> tuple[float, float]:
    """Draw the oracle; return the lantern anchor (design-space x, y) for this frame."""
    sway = math.sin(TAU * t)
    sway2 = math.sin(TAU * t * 2 + 0.4)
    # hair back mass + strands (wind blows left)
    d.ellipse([s(306), s(240), s(414), s(372)], fill=HAIR)
    for i, st in enumerate(hair_strands(t)):
        d.line(pts(st), fill=HAIR_L if i % 3 == 0 else HAIR, width=s(7 if i % 2 else 9), joint="curve")
    # robe
    left = bez((318, 398), (296, 560 + 8 * sway), (238, 780), (176 + 10 * sway, 960))
    hem = [
        (176 + 10 * sway + (544 - 176) * k / 30, 960 - 10 * abs(math.sin(k / 30 * TAU * 1.5 + TAU * t)))
        for k in range(31)
    ]
    right = bez((544 - 8 * sway, 960), (470, 780), (432, 560 - 6 * sway), (402, 398))
    d.polygon(pts(left + hem + right), fill=ROBE)
    # Art Nouveau folds
    for off in (-50, -12, 30, 70):
        fold = bez(
            (360 + off * 0.3, 430),
            (350 + off * 0.6 + 6 * sway2, 600),
            (360 + off + 8 * sway, 760),
            (360 + off * 1.6 + 10 * sway, 950),
        )
        d.line(pts(fold), fill=ROBE_D, width=s(3), joint="curve")
    # gold collar + belt with spiral clasp
    d.line(pts(bez((318, 400), (338, 430), (382, 430), (402, 400))), fill=GOLD, width=s(5))
    d.line(pts(bez((300, 520), (340, 534), (390, 534), (432, 518))), fill=GOLD, width=s(6))
    spiral = [
        (366 + (2 + 1.6 * a) * math.cos(a + TAU * t), 530 + (2 + 1.6 * a) * math.sin(a + TAU * t))
        for a in np.linspace(0, 4 * math.pi, 60)
    ]
    d.line(pts(spiral), fill=LIGHT, width=s(2))
    # neck + head
    d.polygon(pts([(348, 340), (372, 340), (376, 402), (344, 402)]), fill=SKIN)
    d.ellipse([s(322), s(256), s(398), s(352)], fill=SKIN)
    # front hair fringe framing face
    d.chord([s(316), s(244), s(404), s(330)], 180, 360, fill=HAIR)
    d.line(pts(bez((320, 290), (310, 320), (316, 350), (330, 372))), fill=HAIR, width=s(8))
    # closed eyes, brows, mouth
    blink = 1 + 0.3 * max(0, math.sin(TAU * t * 1))  # gentle lid breathing
    for ex in (343, 377):
        d.arc([s(ex - 9), s(304 - 4 * blink), s(ex + 9), s(304 + 4 * blink)], 10, 170, fill=INK, width=s(2))
        d.arc([s(ex - 11), s(290), s(ex + 11), s(300)], 200, 340, fill=HAIR, width=s(2))
    d.arc([s(352), s(326), s(368), s(336)], 20, 160, fill=(170, 70, 80), width=s(2))
    # circlet with teal gem
    d.arc([s(322), s(262), s(398), s(298)], 200, 340, fill=GOLD, width=s(3))
    d.ellipse([s(355), s(258), s(365), s(268)], fill=TEAL)
    # raised arm: sleeve drape then forearm
    lx, ly = LANTERN_BASE[0], LANTERN_BASE[1] + 6 * sway2
    d.polygon(pts([(398, 404), (432, 330), (474, 268 + 3 * sway2), (450, 360), (420, 440)]), fill=ROBE_D)
    d.line([(s(430), s(336)), (s(lx - 18), s(ly + 46))], fill=SKIN, width=s(14))
    d.ellipse([s(lx - 28), s(ly + 32), s(lx - 8), s(ly + 54)], fill=SKIN)
    # lantern chain
    d.line([(s(lx - 18), s(ly + 40)), (s(lx), s(ly + 18))], fill=GOLD, width=s(2))
    return lx, ly


def crescent(d: ImageDraw.ImageDraw, x: float, y: float, r: float, col: tuple[int, int, int]) -> None:
    d.ellipse([s(x - r), s(y - r), s(x + r), s(y + r)], fill=col)


def draw_lantern(d: ImageDraw.ImageDraw, glow_d: ImageDraw.ImageDraw, x: float, y: float, t: float) -> None:
    pulse = 0.5 + 0.5 * math.sin(TAU * t * 2)
    # cage
    d.polygon(pts([(x - 14, y - 6), (x + 14, y - 6), (x + 10, y + 20), (x - 10, y + 20)]), outline=GOLD, width=s(2))
    d.line([(s(x - 16), s(y - 6)), (s(x + 16), s(y - 6))], fill=GOLD, width=s(3))
    # crescent flame (outer moon minus offset moon)
    crescent(d, x, y + 6, 9, LIGHT)
    crescent(d, x + 4, y + 3, 7, (60, 30, 80))
    # glow layer
    gr = 70 + 18 * pulse
    glow_d.ellipse([s(x - gr), s(y + 6 - gr), s(x + gr), s(y + 6 + gr)], fill=(150, 120, 60))
    glow_d.ellipse([s(x - 20), s(y - 14), s(x + 20), s(y + 26)], fill=(255, 230, 160))
    # light rays
    for k in range(12):
        a = TAU * k / 12 + TAU * t / 6
        length = 40 + 14 * math.sin(TAU * t * 2 + k)
        glow_d.line(
            [
                (s(x + 22 * math.cos(a)), s(y + 6 + 22 * math.sin(a))),
                (s(x + (22 + length) * math.cos(a)), s(y + 6 + (22 + length) * math.sin(a))),
            ],
            fill=(200, 170, 90),
            width=s(2),
        )


def draw_motes(glow_d: ImageDraw.ImageDraw, x: float, y: float, t: float) -> None:
    for u0, dx, ph, r in scatter()[1]:
        u = (u0 + t) % 1  # loops exactly once per clip
        mx = x + dx + 40 * u * math.sin(ph) + 8 * math.sin(TAU * u * 2 + ph)
        my = y - 260 * u + 20
        a = math.sin(math.pi * u)
        c = tuple(int(v * a) for v in (255, 220, 140))
        glow_d.ellipse([s(mx - r), s(my - r), s(mx + r), s(my + r)], fill=c)


def draw_frame(d: ImageDraw.ImageDraw) -> None:
    """Arch-topped gold border, whiplash corner curls and the title cartouche."""
    m, arch_top = ARCH_MARGIN, ARCH_TOP
    outer = arch_outline(m, arch_top)
    d.line(pts(outer + [(m, H - m)]), fill=GOLD, width=s(6), joint="curve")
    inner = [
        (
            p[0] + (12 if p[0] < W / 2 else -12) if p[1] > arch_top - 1 else p[0],
            p[1] + 12 if p[1] <= arch_top else p[1],
        )
        for p in outer
    ]
    inner[0] = (m + 12, H - m - 12)
    inner[-1] = (W - m - 12, H - m - 12)
    d.line(pts(inner + [inner[0]]), fill=GOLD_D, width=s(2), joint="curve")
    # whiplash corner curls
    for sx in (1, -1):
        cx = W / 2 + sx * (W / 2 - 70)
        for k in range(3):
            sp = [
                (cx + sx * (6 + 3 * a) * math.cos(a + k), 210 + k * 230 + (6 + 3 * a) * math.sin(a + k))
                for a in np.linspace(0, 3.2 * math.pi, 70)
            ]
            d.line(pts(sp), fill=GOLD_D, width=s(2))
    # title cartouche
    cy = H - 70
    d.rounded_rectangle(
        [s(170), s(cy - 24), s(W - 170), s(cy + 24)], radius=s(22), fill=INK, outline=GOLD, width=s(3)
    )
    d.text((s(W / 2), s(cy)), TITLE, fill=GOLD, font=title_font(), anchor="mm")


def render(t: float) -> Image.Image:
    """Render one ``W``x``H`` RGB frame at loop phase ``t`` in [0, 1)."""
    img = background().copy()
    d = ImageDraw.Draw(img)
    glow = Image.new("RGB", img.size, (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    draw_stars(d, t)
    draw_astrolabe(d, t)
    draw_halo(d, t)
    draw_sigils(d, t)
    lx, ly = draw_figure(d, t)
    draw_lantern(d, gd, lx, ly, t)
    draw_motes(gd, lx, ly, t)
    glow = glow.filter(ImageFilter.GaussianBlur(s(GLOW_BLUR)))
    img = ImageChops.screen(img, glow)
    img.paste(INK, mask=outside_mask())
    draw_frame(ImageDraw.Draw(img))
    return img.resize((W, H), Image.LANCZOS)


# ---------- I/O edges ----------
def write_frames(frames_dir: Path, poster: Path, n_frames: int = N) -> None:
    """Write ``n_frames`` PNGs as ``f%04d.png`` into ``frames_dir``; frame ``n/4`` also becomes the poster."""
    for i in range(n_frames):
        frame = render(i / n_frames)
        frame.save(frames_dir / f"f{i:04d}.png")
        if i == n_frames // 4:
            frame.save(poster)


def encode_video(frames_dir: Path, out: Path, fps: int = FPS) -> None:
    """Encode the ``f%04d.png`` sequence in ``frames_dir`` to an H.264 MP4 with ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out),
        ],
        check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT, help=f"output mp4 (default: {OUT})")
    p.add_argument("--poster", type=Path, default=POSTER, help=f"poster png (default: {POSTER})")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tmp = Path(tempfile.mkdtemp())
    write_frames(tmp, args.poster, N)
    encode_video(tmp, args.out)
    shutil.rmtree(tmp)
    print(f"wrote {args.out} and {args.poster}")


if __name__ == "__main__":
    main()
