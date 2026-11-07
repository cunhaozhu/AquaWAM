#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np


def load(p: Path) -> dict[str, np.ndarray]:
    z = np.load(p, allow_pickle=True)
    rows, cols, lay = z["rows"], list(z["cols"]), list(z["layout"])
    off, out = 0, {}
    for c, n in zip(cols, lay):
        out[str(c)] = rows[:, off:off + int(n)]
        off += int(n)
    return out


def segments(t: np.ndarray, max_gap: float, min_len: int):
    cuts = np.where(np.diff(t) > max_gap)[0] + 1
    s = 0
    for c in list(cuts) + [len(t)]:
        if c - s >= min_len:
            yield s, c
        s = c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--out", default="/hy-tmp/data/closedloop_sensors")
    ap.add_argument("--max-per-block", type=int, default=40)
    ap.add_argument("--holdout-mod", type=int, default=10, help="episode index %% this == 0 is held out")
    ap.add_argument("--exclude", default="_eta,_archive,_naive", help="substrings of arm dirs to skip")
    ap.add_argument("--min-sec", type=float, default=5.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    excl = [e for e in args.exclude.split(",") if e]
    files = sorted(glob.glob(f"{args.runs}/*/*/logs/episode_*_sensors.npz"))
    per_block: dict[str, int] = {}
    n_files = n_ticks = n_skip_hold = 0
    for f in files:
        parts = f.split("/")
        arm, task = parts[-4], parts[-3]
        if any(e in arm for e in excl):
            continue
        m = re.search(r"episode_(\d+)_sensors", f)
        ep = int(m.group(1)) if m else 0
        if args.holdout_mod > 0 and ep % args.holdout_mod == 0:
            n_skip_hold += 1
            continue
        key = f"{arm}/{task}"
        if per_block.get(key, 0) >= args.max_per_block:
            continue
        try:
            d = load(Path(f))
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        t = d["t"][:, 0]
        ok = (d["pressure"][:, 0] > 0) & np.all(np.isfinite(d["dvl_true"]), axis=1) & np.all(np.isfinite(d["imu_la"]), axis=1)
        if ok.sum() < args.min_sec * 10:
            continue
        first = int(np.argmax(ok))
        d = {k: v[first:] for k, v in d.items()}
        t = t[first:]
        wrote = False
        for k, (s, e) in enumerate(segments(t, 0.25, int(args.min_sec * 10))):
            sl = slice(s, e)
            n = e - s
            name = f"{arm}__{task}__episode{ep}" + (f"_s{k}" if k else "") + ".npz"
            np.savez_compressed(
                out / name,
                pwm=d["pwm_cmd"][sl].astype(np.float32),
                eta=np.ones((n, 8), np.float32),
                pwm_is_commanded=np.bool_(True),
                dvl=d["dvl_true"][sl].astype(np.float32),
                imu_av=d["imu_av"][sl].astype(np.float32),
                imu_la=d["imu_la"][sl].astype(np.float32),
                pressure=(d["pressure"][sl] * 1e4).astype(np.float32).reshape(n, 1),
                dvl_h=d["alt_true"][sl].astype(np.float32).reshape(n, 1),
                timestamp=(t[sl] - t[s]).astype(np.float64),
                dt=np.float64(0.1),
            )
            n_files += 1
            n_ticks += n
            wrote = True
        if wrote:
            per_block[key] = per_block.get(key, 0) + 1
    print(f"wrote {n_files} files, {n_ticks} ticks ({n_ticks / 36000:.1f} h) from {len(per_block)} blocks; "
          f"held out {n_skip_hold} trials (index % {args.holdout_mod} == 0)")


if __name__ == "__main__":
    main()
