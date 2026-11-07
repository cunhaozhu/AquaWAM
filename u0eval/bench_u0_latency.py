#!/usr/bin/env python3
import json, platform, sys, time
from pathlib import Path

import numpy as np
import torch

model_dir = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20; out = sys.argv[3] if len(sys.argv) > 3 else ""
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

data_config = load_data_config("u0_bot")
t0 = time.time()
policy = Gr00tPolicy(model_path=model_dir, modality_config=data_config.modality_config(), modality_transform=data_config.transform(),
                     embodiment_tag="new_embodiment", denoising_steps=4, device="cuda")
load_s = time.time() - t0
rng = np.random.default_rng(0)
obs = {
    "video.ego": rng.integers(0, 255, (1, 240, 320, 3), dtype=np.uint8),
    "video.wrist": rng.integers(0, 255, (1, 240, 320, 3), dtype=np.uint8),
    "state.joint_pos": np.zeros((1, 5), np.float32), "state.pwm": np.zeros((1, 8), np.float32),
    "state.joint_v": np.zeros((1, 5), np.float32), "state.dvl_v": np.zeros((1, 3), np.float32),
    "state.imu_av": np.zeros((1, 3), np.float32), "state.imu_la": np.zeros((1, 3), np.float32),
    "state.pressure": np.zeros((1, 1), np.float32), "state.dvl_h": np.zeros((1, 1), np.float32),
    "annotation.human.action.task_description": ["Go to the charging station"],
}
for _ in range(3):
    policy.get_action(obs)
torch.cuda.synchronize()
ts = []
for _ in range(n):
    torch.cuda.synchronize(); t = time.perf_counter(); a = policy.get_action(obs); torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t))
arr = np.asarray(ts)
n_params = sum(p.numel() for p in policy.model.parameters())
act = a[0] if isinstance(a, (tuple, list)) else a
keys = sorted(act.keys()) if isinstance(act, dict) else []
res = {"model": model_dir, "params_B": n_params / 1e9, "load_s": load_s, "median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
       "n": n, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "action_keys": keys,
       "chunk": int(np.asarray(act[keys[0]]).shape[0]) if keys else None, "device": torch.cuda.get_device_name(0), "host": platform.node(), "torch": torch.__version__}
print("[bench-u0]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
if out:
    Path(out).write_text(json.dumps(res, indent=1))
