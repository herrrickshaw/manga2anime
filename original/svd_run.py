"""Send a 1024x576 still to the public SVD Space and save the clip.

Usage: svd_run.py IN.png OUT.mp4 [motion]

Calls the hosted Gradio demo ``multimodalart/stable-video-diffusion`` (Stable Video
Diffusion img2vid-XT) via ``gradio_client``; nothing runs locally. ``motion`` is the
SVD motion-bucket id (default 100). Seed 42 and 6 fps are fixed to match the recorded run.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

SPACE = "multimodalart/stable-video-diffusion"
API_NAME = "/video"
SEED = 42
RANDOMIZE_SEED = False
FPS = 6
DEFAULT_MOTION = 100.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="input still (1024x576 PNG)")
    p.add_argument("out", type=Path, help="output mp4 path")
    p.add_argument("motion", type=float, nargs="?", default=DEFAULT_MOTION, help="motion bucket id")
    return p.parse_args(argv)


def run_svd(
    src: Path,
    out: Path,
    motion: float = DEFAULT_MOTION,
    client_factory: Callable[[str], Any] | None = None,
) -> Any:
    """Submit ``src`` to the SVD Space, copy the returned clip to ``out``, return the seed used.

    ``client_factory`` lets tests inject a fake client; by default it is ``gradio_client.Client``.
    """
    from gradio_client import handle_file

    if client_factory is None:
        from gradio_client import Client

        client_factory = Client
    res, seed = client_factory(SPACE).predict(
        handle_file(str(src)), SEED, RANDOMIZE_SEED, motion, FPS, api_name=API_NAME
    )
    shutil.copy(res["video"], out)
    return seed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seed = run_svd(args.src, args.out, args.motion)
    print(args.out, "seed", seed)


if __name__ == "__main__":
    main()
