#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

STATE_ORDER = ("joint_pos", "pwm", "joint_v", "dvl_v", "imu_av", "imu_la", "pressure", "dvl_h")
STATE_DIMS = (5, 8, 5, 3, 3, 3, 1, 1)
CHUNK, N_JOINTS = 16, 5
CAM_MAP = {"video.ego": "observation.images.base_0_rgb", "video.wrist": "observation.images.left_wrist_0_rgb"}


def _arr(obs, key, dim):
    v = obs.get(key)
    if v is None:
        return np.zeros(dim, np.float32)
    a = np.asarray(v, np.float32).ravel()
    if a.size < dim:
        a = np.concatenate([a, np.zeros(dim - a.size, np.float32)])
    return a[:dim]


class Policy:
    def __init__(self, ckpt: str, device: str = "cuda"):
        from lerobot.policies.factory import get_policy_class
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cls = get_policy_class(cfg.type)
        self.policy = cls.from_pretrained(ckpt, config=cfg)
        self.policy.to(device).eval()
        self.device, self.cfg = device, cfg
        try:
            from lerobot.policies.factory import make_pre_post_processors
            self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=ckpt)
        except Exception as e:
            print(f"[lerobot-server] saved processors unavailable ({type(e).__name__}: {str(e)[:120]}); using defaults", flush=True)
            try:
                from lerobot.policies.factory import make_pre_post_processors
                self.pre, self.post = make_pre_post_processors(cfg)
            except Exception as e2:
                print(f"[lerobot-server] default processors unavailable ({type(e2).__name__}); raw model I/O", flush=True)
                self.pre, self.post = None, None
        self.cam_map = dict(CAM_MAP)
        env_map = os.environ.get("LEROBOT_CAM_MAP")
        if env_map:
            self.cam_map = json.loads(env_map)
        else:
            feats = getattr(cfg, "input_features", {}) or {}
            img_keys = [k for k, f in feats.items() if str(getattr(f, "type", "")).endswith("VISUAL")]
            if len(img_keys) >= 2 and not all(k in img_keys for k in CAM_MAP.values()):
                self.cam_map = {"video.ego": img_keys[0], "video.wrist": img_keys[1]}
        print(f"[lerobot-server] loaded {cfg.type} from {ckpt}; chunk {getattr(cfg, 'n_action_steps', '?')}; "
              f"cameras {self.cam_map}", flush=True)
        self.n_act = int(getattr(cfg, "n_action_steps", CHUNK) or CHUNK)
        self.queue: list[np.ndarray] = []
        self.state_dim = None
        if os.environ.get("LEROBOT_ZERO_SHOT") == "1":
            st = (getattr(cfg, "input_features", {}) or {}).get("observation.state")
            self.state_dim = int(st.shape[0]) if st is not None and getattr(st, "shape", None) else None
            if self.state_dim and self.state_dim != sum(STATE_DIMS):
                print(f"[lerobot-server] zero-shot adapter: 29-d state -> {self.state_dim}-d positional", flush=True)

    def _batch(self, obs):
        state = np.concatenate([_arr(obs, f"state.{k}", d) for k, d in zip(STATE_ORDER, STATE_DIMS)])
        if self.state_dim and self.state_dim != state.size:
            state = np.concatenate([state, np.zeros(max(0, self.state_dim - state.size), np.float32)])[: self.state_dim]
        b = {"observation.state": torch.from_numpy(state).float()[None].to(self.device)}
        for src, dst in self.cam_map.items():
            im = obs.get(src)
            if im is None:
                continue
            a = np.asarray(im, np.uint8).reshape(240, 320, 3).astype(np.float32) / 255.0
            b[dst] = torch.from_numpy(a.transpose(2, 0, 1))[None].to(self.device)
        task = obs.get("annotation.human.action.task_description") or [""]
        b["task"] = [task[0] if isinstance(task, (list, tuple)) else str(task)]
        return b

    @torch.no_grad()
    def act(self, obs):
        dump = os.environ.get("LEROBOT_DUMP_OBS")
        if dump:
            os.makedirs(dump, exist_ok=True)
            k = len([f for f in os.listdir(dump) if f.endswith(".npz")])
            if k < 5:
                np.savez(os.path.join(dump, f"obs_{k}.npz"),
                         **{key.replace(".", "_"): np.asarray(v) for key, v in obs.items() if key.startswith(("video.", "state."))},
                         task=str(obs.get("annotation.human.action.task_description", "")))
        batch = self._batch(obs)
        if self.pre is not None:
            batch = self.pre(batch)
        if hasattr(self.policy, "predict_action_chunk"):
            out = self.policy.predict_action_chunk(batch)
            if self.post is not None:
                out = torch.stack([self.post(out[:, i]) for i in range(out.shape[1])], dim=1)
            chunk = out.squeeze(0).float().cpu().numpy()
        else:
            acts = []
            for _ in range(self.n_act):
                a = self.policy.select_action(batch)
                if self.post is not None:
                    a = self.post(a)
                acts.append(a.squeeze(0).float().cpu().numpy())
            chunk = np.stack(acts)
        chunk = np.atleast_2d(chunk)
        if chunk.shape[1] < 13:
            chunk = np.concatenate([chunk, np.zeros((len(chunk), 13 - chunk.shape[1]), chunk.dtype)], axis=1)
        chunk = chunk[:, :13]
        if len(chunk) < CHUNK:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], CHUNK - len(chunk), 0)])
        chunk = chunk[:CHUNK]
        joints = np.clip(chunk[:, :N_JOINTS], -0.1, 2.0).astype(np.float64)
        pwm = np.clip(chunk[:, N_JOINTS:N_JOINTS + 8], -1.0, 1.0).astype(np.float64)
        return [{"action.pwm": pwm, "action.joint_pos": joints}, {}]


def make_handler(policy):
    import json_numpy

    json_numpy.patch()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(b"healthy")

        def do_POST(self):
            import json
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            obs = payload.get("observation", payload)
            t0 = time.time()
            try:
                out = policy.act(obs)
            except Exception as e:
                import traceback; traceback.print_exc()
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode()); return
            dt = (time.time() - t0) * 1e3
            policy._n = getattr(policy, "_n", 0) + 1
            if policy._n % 20 == 1:
                print(f"[lerobot-server] act#{policy._n} {dt:.0f} ms |pwm| {np.abs(out[0]['action.pwm']).mean():.3f}", flush=True)
            body = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    pol = Policy(args.ckpt, args.device)
    srv = HTTPServer(("0.0.0.0", args.port), make_handler(pol))
    print(f"[lerobot-server] listening on {args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
