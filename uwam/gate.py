
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GateCfg:
    kappa: float = 1.0
    h: float = 5.0
    floor: float = 0.05
    decay: float = 0.9
    alpha_base: float = 0.0

    @classmethod
    def from_calib(cls, path) -> Optional["GateCfg"]:
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(kappa=float(d["kappa"]), h=float(d["h"]),
                   floor=float(d.get("floor", 0.05)), decay=float(d.get("decay", 0.9)),
                   alpha_base=float(d.get("alpha_base", 0.0)))


class CusumGate:

    def __init__(self, cfg: Optional[GateCfg] = None):
        self.cfg = cfg or GateCfg()
        self.reset()

    def reset(self):
        self.S = 0.0
        self.anchor: Optional[np.ndarray] = None
        self.latched = False

    def start_blind(self, v_anchor: np.ndarray):
        self.reset()
        self.anchor = np.asarray(v_anchor, np.float64)

    def step(self, v_est: np.ndarray, sigma: np.ndarray, goal_changed: bool = False) -> float:
        if goal_changed:
            self.latched = True
        if self.latched:
            return 1.0
        c = self.cfg
        if self.anchor is None:
            return float(c.alpha_base)
        denom = float(np.linalg.norm(np.asarray(sigma, np.float64) + c.floor))
        z = float(np.linalg.norm(np.asarray(v_est, np.float64) - self.anchor)) / max(denom, 1e-9)
        self.S = max(0.0, c.decay * self.S + z - c.kappa)
        return float(max(c.alpha_base, min(1.0, self.S / c.h)))


def _cusum_trace(z: np.ndarray, kappa: float, h: float, decay: float) -> np.ndarray:
    S = 0.0
    alpha = np.empty(len(z))
    for i, zi in enumerate(z):
        S = max(0.0, decay * S + float(zi) - kappa)
        alpha[i] = min(1.0, S / h)
    return alpha


def calibrate(
    z_null: "list[np.ndarray]",
    z_shift: "list[tuple[np.ndarray, int]]",
    dt: float,
    far_target: float = 0.05,
    open_level: float = 0.5,
    decay: float = 0.9,
    kappa_quantiles: "tuple[float, ...]" = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995),
    h_grid: "tuple[float, ...]" = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0),
    shift_sizes: "Optional[list]" = None,
) -> dict:
    pooled = np.concatenate([np.asarray(z) for z in z_null])
    if shift_sizes is not None and len(shift_sizes) == len(z_shift):
        med = float(np.median(shift_sizes))
        big = [i for i, s in enumerate(shift_sizes) if s >= med]
    else:
        big = list(range(len(z_shift)))
    results = []
    for q in kappa_quantiles:
        kappa = float(np.quantile(pooled, q))
        for h in h_grid:
            fa = [bool((_cusum_trace(np.asarray(z), kappa, h, decay) >= open_level).any())
                  for z in z_null]
            far = float(np.mean(fa))
            delays, missed = [], []
            for z, tc in z_shift:
                a = _cusum_trace(np.asarray(z), kappa, h, decay)
                idx = np.nonzero(a[tc:] >= open_level)[0]
                if len(idx):
                    delays.append(float(idx[0]) * dt)
                    missed.append(False)
                else:
                    delays.append(float(len(z) - tc) * dt)
                    missed.append(True)
            d_delay = float(np.mean([delays[i] for i in big])) if big else None
            d_miss = float(np.mean([missed[i] for i in big])) if big else None
            results.append({
                "kappa": round(kappa, 4), "kappa_q": q, "h": h, "far": round(far, 4),
                "delay_s": round(d_delay, 3) if d_delay is not None else None,
                "miss": round(d_miss, 3) if d_miss is not None else None,
                "miss_all": round(float(np.mean(missed)), 3) if missed else None,
            })
    feasible = [r for r in results if r["far"] <= far_target and r["delay_s"] is not None]
    if feasible:
        best = min(feasible, key=lambda r: (r["miss"], round(r["delay_s"] * 2) / 2,
                                            -r["kappa"], -r["h"]))
    else:
        best = min(results, key=lambda r: (r["far"], r["delay_s"] or 1e9))
    return {
        "kappa": best["kappa"], "h": best["h"], "decay": decay,
        "far_target": far_target, "open_level": open_level, "dt": dt,
        "achieved_far": best["far"], "decision_delay_s": best["delay_s"],
        "decision_miss": best["miss"], "miss_all_shifts": best["miss_all"],
        "kappa_quantile": best["kappa_q"],
        "n_null": len(z_null), "n_shift": len(z_shift), "n_decision_shift": len(big),
        "shift_size_median": (round(float(np.median(shift_sizes)), 4)
                              if shift_sizes is not None and len(shift_sizes) else None),
        "null_z_stats": {
            "mean": round(float(pooled.mean()), 4), "std": round(float(pooled.std()), 4),
            "q90": round(float(np.quantile(pooled, 0.9)), 4),
            "q99": round(float(np.quantile(pooled, 0.99)), 4),
        },
        "grid": results,
    }
