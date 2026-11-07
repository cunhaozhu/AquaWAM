#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image

CHUNK, N_JOINTS = 16, 5


class OpenVLAPolicy:
    def __init__(self, ckpt: str, device: str = "cuda", unnorm_key: str = "usim"):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                          low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()
        self.device, self.unnorm_key = device, unnorm_key
        assert unnorm_key in self.vla.norm_stats, f"norm_stats has no key {unnorm_key!r}: {list(self.vla.norm_stats)[:5]}"
        self.dim = self.vla.get_action_dim(unnorm_key)
        self._n, self._last = 0, None
        print(f"[openvla-server] loaded {ckpt} in {time.time() - t0:.0f} s, action dim {self.dim}", flush=True)

    @torch.no_grad()
    def act(self, obs):
        instr = obs.get("annotation.human.action.task_description", ["go to the goal"])
        instr = (instr[0] if isinstance(instr, (list, tuple, np.ndarray)) else str(instr))
        instr = str(instr).strip().lower().rstrip(".")
        im = obs.get("video.ego")
        if im is None:
            im = obs.get("video.wrist")
        img = Image.fromarray(np.asarray(im, np.uint8).reshape(240, 320, 3))
        prompt = f"In: What action should the robot take to {instr}?\nOut:"
        inputs = self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        action = np.asarray(action, np.float64).ravel()
        if action.size < 13:
            action = np.concatenate([action, np.zeros(13 - action.size)])
        chunk = np.repeat(action[None, :13], CHUNK, axis=0)
        joints = np.clip(chunk[:, :N_JOINTS], -0.1, 2.0)
        pwm = np.clip(chunk[:, N_JOINTS:N_JOINTS + 8], -1.0, 1.0)
        self._n += 1
        return [{"action.pwm": pwm, "action.joint_pos": joints}, {}]


def make_handler(policy):
    import json_numpy
    json_numpy.patch()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"healthy")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            obs = payload.get("observation", payload) if isinstance(payload, dict) else payload
            t0 = time.time()
            try:
                out = policy.act(obs)
            except Exception as e:
                print(f"[openvla-server] act failed: {type(e).__name__}: {e}", flush=True)
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode()); return
            dt = (time.time() - t0) * 1000
            if policy._n % 50 == 1:
                print(f"[openvla-server] act#{policy._n} {dt:.0f} ms |pwm| {np.abs(out[0]['action.pwm']).mean():.3f}", flush=True)
            body = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--ckpt", default="/hy-tmp/baselines/ft/openvla_usim/merged")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--unnorm-key", default="usim")
    a = ap.parse_args()
    policy = OpenVLAPolicy(a.ckpt, unnorm_key=a.unnorm_key)
    srv = HTTPServer((a.host, a.port), make_handler(policy))
    print(f"[openvla-server] serving on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
