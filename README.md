# manga2anime

Experiments animating a static manga/anime-style panel into short video clips using **open-source** image-to-video models, run on free hosted GPUs (Hugging Face Spaces / Colab) — never on local hardware.

## Source art

`assets/frame1_source.png` — cropped from a CC0 (public domain) stock manga-style illustration ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Manga_page_publicdomainq.png), via publicdomainq.net). Not derived from any copyrighted manga/anime series.

`assets/frame2_generated.png` — a locally-generated second keyframe (slight head tilt + blink), used as the "end frame" for two-frame interpolation models like ToonCrafter.

## Results

| Model | Type | Output |
|---|---|---|
| [Stable Video Diffusion (img2vid-XT 1.1)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) | single-image → video | `assets/svd_output.mp4` |
| [ToonCrafter](https://github.com/Doubiiu/ToonCrafter) | two-keyframe interpolation, cartoon/anime-aware | `assets/tooncrafter_output.mp4` (pending) |

Both were run via each project's public hosted demo (Hugging Face Space), driven through a browser — not executed locally, since the dev Mac (M1, 8GB RAM) has no CUDA GPU.

## Why these tools

Commercial closed tools like DomoAI, Anifusion, and Kling AI offer similar manga-panel animation, but aren't open source. This repo tracks open-source equivalents for the same use case.
