"""Send a 1024x576 still to the public SVD Space and save the clip. Usage: svd_run.py IN.png OUT.mp4 [motion]"""
import shutil, sys
from gradio_client import Client, handle_file

src, out = sys.argv[1], sys.argv[2]
motion = float(sys.argv[3]) if len(sys.argv) > 3 else 100
res, seed = Client("multimodalart/stable-video-diffusion").predict(
    handle_file(src), 42, False, motion, 6, api_name="/video")
shutil.copy(res["video"], out)
print(out, "seed", seed)
