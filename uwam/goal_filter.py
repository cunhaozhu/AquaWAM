from __future__ import annotations

from collections import deque

import numpy as np


def _rot(yaw: float, d_body: np.ndarray) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([cy * d_body[0] - sy * d_body[1], sy * d_body[0] + cy * d_body[1], d_body[2]])


class GoalFilter:
    def __init__(self, n_keep: int = 6, reset_k: int = 3, gate_deg: float = 35.0, age_decay: float = 0.75):
        self.n_keep = n_keep
        self.reset_k = reset_k
        self.gate = np.radians(gate_deg)
        self.age_decay = age_decay
        self.reset()

    def reset(self) -> None:
        self.rays: deque = deque(maxlen=self.n_keep)
        self.x: np.ndarray | None = None
        self.pending: list = []
        self.n = 0

    def _solve(self, p_now: np.ndarray) -> np.ndarray:
        m = len(self.rays)
        acc = np.zeros(3)
        for i, (p_i, u_i, r_i) in enumerate(self.rays):
            w = self.age_decay ** (m - 1 - i)
            pt = p_i + r_i * u_i
            v = pt - p_now
            nv = float(np.linalg.norm(v))
            acc += w * (u_i if (r_i > 2.0 or nv < 0.3) else v / nv)
        u = acc / max(float(np.linalg.norm(acc)), 1e-9)
        recent = [r for _, _, r in list(self.rays)[-3:]]
        r_new = recent[-1]
        if len(recent) >= 2 and (max(recent) - min(recent)) < 0.3 * max(r_new, 0.5):
            r_use = float(np.median(recent))
        else:
            r_use = r_new
        return p_now + r_use * u

    def update(self, pos, yaw: float, d_body) -> np.ndarray:
        p = np.asarray(pos, np.float64).reshape(3)
        d = np.asarray(d_body, np.float64).reshape(3)
        d_w = _rot(yaw, d)
        r = float(np.linalg.norm(d_w))
        if r < 1e-3:
            return (self.x.copy() if self.x is not None else p.copy())
        u = d_w / r
        if self.x is not None and self.n >= 2:
            v = self.x - p
            nv = float(np.linalg.norm(v))
            ang = float(np.arccos(np.clip(u @ (v / nv), -1.0, 1.0))) if nv > 0.3 else 0.0
            if ang > self.gate:
                self.pending.append((p.copy(), u.copy(), r))
                if len(self.pending) >= self.reset_k:
                    pend = self.pending
                    self.reset()
                    for tup in pend:
                        self.rays.append(tup)
                    self.n = len(pend)
                    self.x = self._solve(p)
                return self.x.copy()
            self.pending = []
        self.rays.append((p.copy(), u.copy(), r))
        self.n += 1
        self.x = self._solve(p)
        return self.x.copy()

    def position_fix(self, landmark_w, yaw: float, d_body, gate_m: float = 1.5, pos_guess=None):
        d = np.asarray(d_body, np.float64).reshape(3)
        fix = np.asarray(landmark_w, np.float64) - _rot(yaw, d)
        if pos_guess is not None and np.linalg.norm(fix - np.asarray(pos_guess, np.float64)) > gate_m:
            return None
        return fix
