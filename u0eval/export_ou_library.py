#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.data import load_ou_split
src, dst = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/data/ou_explore", sys.argv[2] if len(sys.argv) > 2 else "/hy-tmp/models/uwam/ou_library_pwm.npy"
lib = np.concatenate([ep.pwm for ep in load_ou_split(Path(src))], axis=0).astype(np.float32)
np.save(dst, lib); print(f"{lib.shape} -> {dst} ({Path(dst).stat().st_size/1e6:.1f} MB)")
