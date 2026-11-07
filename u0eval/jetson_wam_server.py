#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    cv2.resize(np.zeros((8, 8, 3), np.uint8), (4, 4))
except Exception:
    from PIL import Image
    cv2.resize = lambda im, size, interpolation=None: np.asarray(Image.fromarray(im).resize(size, Image.BILINEAR))
    print("[jetson] cv2.resize unusable -> PIL resize", flush=True)
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
sys.argv = [str(here / "wam_policy_server.py")] + sys.argv[1:]
runpy.run_path(str(here / "wam_policy_server.py"), run_name="__main__")
