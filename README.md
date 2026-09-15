# manga2anime

Experiments animating a static manga/anime-style panel into short video clips using **open-source** image-to-video models, run on free hosted GPUs (Hugging Face Spaces / Colab) — never on local hardware.

## Source art

`assets/frame1_source.png` — cropped from a CC0 (public domain) stock manga-style illustration ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Manga_page_publicdomainq.png), via publicdomainq.net). Not derived from any copyrighted manga/anime series.

`assets/frame2_generated.png` — a locally-generated second keyframe (slight head tilt + blink), used as the "end frame" for two-frame interpolation models like ToonCrafter.

## Results

| Model | Type | Output |
|---|---|---|
| [Stable Video Diffusion (img2vid-XT 1.1)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) | single-image → video | `assets/svd_output.mp4` |
| [ToonCrafter](https://github.com/Doubiiu/ToonCrafter) | two-keyframe interpolation, cartoon/anime-aware | inconclusive — see below |

Both were run via each project's public hosted demo/notebook, driven through a browser — not executed locally, since the dev Mac (M1, 8GB RAM) has no CUDA GPU.

### ToonCrafter status (inconclusive)

- The Hugging Face Space (`Doubiiu/tooncrafter`) hit its anonymous ZeroGPU quota (shared across all Spaces, exhausted by the SVD run above; resets ~24h, or requires a signed-in HF account for a bigger quota).
- Fell back to `camenduru/ToonCrafter-jupyter`'s Colab notebook on a free T4. Its `requirements.txt` pins 2023-era exact versions (`torch==2.0.0`, `pytorch_lightning==1.9.3`, etc.) with no wheels for Colab's current Python 3.13 — several packages failed to build, which silently aborted the rest of the `pip install -r requirements.txt` batch (so `decord` and `pytorch_lightning` never installed). Fixed by installing `decord` directly and stripping the stale version pins (`sed -i 's/==.*//' requirements.txt`) so pip resolves Python-3.13-compatible versions.
- After that fix, the app started loading the model and downloaded the ~3.94GB CLIP checkpoint successfully (now cached in that Colab runtime). The browser tab driving the notebook then became unresponsive (repeated screenshot timeouts, stalled output, one stray interrupt) before the Gradio server's launch could be confirmed.
- Net: the two real dependency bugs are identified and fixed; whether the server actually comes up needs a re-run in a stable browser session against the same (still-provisioned) Colab runtime.

## Why these tools

Commercial closed tools like DomoAI, Anifusion, and Kling AI offer similar manga-panel animation, but aren't open source. This repo tracks open-source equivalents for the same use case.
