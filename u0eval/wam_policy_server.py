#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg, enable_arm, enable_object
from uwam.control import AXIS_SIGN, SamplingMPC, attitude_pwm, vel_track_pwm
from uwam.data import RunningNorm, load_ou_split
from uwam.gate import CusumGate, GateCfg
from uwam.models import DynamicsWAM

L = 16
DT = 0.1
CHUNK = 16
N_JOINTS = 5

MODE_BY_PHRASE = (
    ("charge station", "nav"), ("water tower", "nav"),
    ("scan the ship", "scan"), ("inspect the pipeline", "inspect"),
    ("follow the boat", "follow"),
    ("transfer it", "transfer"), ("pick up", "grasp"),
)
ADVANCE_TOL = {"nav": 0.9, "scan": 3.5, "inspect": 1.5, "follow": 2.5,
               "grasp": 0.12, "transfer": 0.15}
ADVANCE_YAW = {"nav": 3.14, "scan": 0.80, "inspect": 0.40, "follow": 1.0,
               "grasp": 0.25, "transfer": 0.35}
V_MAX = {"nav": 0.25, "scan": 0.22, "inspect": 0.20, "follow": 0.25,
         "grasp": 0.18, "transfer": 0.18}
KP_POS = 0.6
FOLLOW_CATCHUP = float(os.environ.get("WAM_FOLLOW_CATCHUP", "1.0"))
ARMED_JOINTS = np.array([0.0, 0.4965, 0.4965, 0.5056, 0.0], np.float32)
GRIPPER_CLOSE = 0.015
GRASP_STANDOFF = {
    "search": np.array([0.45, -0.08, 0.35], np.float32),
    "approach": np.array([0.32, -0.08, 0.27], np.float32),
    "grasp": np.array([0.30, -0.08, 0.20], np.float32),
    "lift": np.array([0.30, -0.08, 0.45], np.float32),
}


def _load_model(ckpt_path: Path, device: str):
    import torch

    cfg = Cfg()
    cfg.model.use_language = False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" in ckpt:
        c = ckpt["cfg"]
        if "disturbance_dim" in c:
            cfg.model.disturbance_dim = int(c["disturbance_dim"])
        if "hidden" in c:
            cfg.model.hidden = int(c["hidden"])
        cfg.model.use_dt = bool(c.get("use_dt", False))
        cfg.model.no_token = bool(c.get("no_token", False))
        if c.get("use_object"):
            enable_object(cfg)
        elif c.get("use_arm"):
            enable_arm(cfg)
    model = DynamicsWAM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    dyn_norm = RunningNorm()
    pwm_norm = RunningNorm()
    dyn_norm.load_state_dict(ckpt["dyn_norm"])
    pwm_norm.load_state_dict(ckpt["pwm_norm"])
    return model, dyn_norm, pwm_norm, cfg


def _newest_traj(base: Path):
    cands = glob.glob(str(base / "logs" / "episode_*_traj.npy"))
    if not cands:
        return None, None
    now = time.time()
    fresh = [p for p in cands if now - os.path.getmtime(p) < 120]
    p = max(fresh or cands, key=os.path.getmtime)
    try:
        arr = np.load(p, allow_pickle=True)
        wps = np.asarray([[float(x) for x in row[:4]] for row in arr], np.float64)
        return p, wps
    except Exception:
        return p, None


def _yaw_wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class WamPolicy:

    def __init__(self, args):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.goal_source = getattr(args, "goal_source", "privileged")
        self.percept = None
        if self.goal_source == "percept":
            from percept.goal_head import PerceptGoal

            e2e = getattr(args, "percept_e2e", "") or None
            if e2e:
                from percept.goal_head import PerceptGoalE2E
                self.percept = PerceptGoalE2E(e2e_ckpt=e2e, device=self.device)
                print(f"[wam-server] goal source: PERCEPT-E2E (fine-tuned DINOv2, intent head {e2e}; "
                      "grasp goals from the frozen-feature head)", flush=True)
            else:
                nav_ckpt = getattr(args, "percept_nav_head", "/hy-tmp/models/uwam/percept_nav_head.pt") or None
                self.percept = PerceptGoal(device=self.device, nav_head_ckpt=nav_ckpt)
                print("[wam-server] goal source: PERCEPT (DINOv2-base head, no privileged reads "
                      f"on grasp/transfer; locomotion head: "
                      f"{'ON ' + str(nav_ckpt) if self.percept.nav_head is not None else 'off (waypoint file)'})",
                      flush=True)
        self.model, self.dyn_norm, self.pwm_norm, self.cfg = _load_model(Path(args.ckpt), self.device)
        n_samples = int(getattr(args, "n_samples", 0) or 0)
        cem_iters = int(getattr(args, "cem_iters", 0) or 0)
        if n_samples > 0:
            self.cfg.control.n_samples = n_samples
        if cem_iters > 0:
            self.cfg.control.cem_iters = cem_iters
        self.mpc = SamplingMPC(self.model, self.dyn_norm, self.pwm_norm, self.cfg.control, device=self.device)
        self.action_source = getattr(args, "action_source", "mpc") or "mpc"
        self.direct = None
        if self.action_source != "mpc":
            from uwam.direct import DirectPolicy

            self.direct = DirectPolicy.load(getattr(args, "direct_ckpt", ""), self.model, self.dyn_norm,
                                            self.pwm_norm, device=self.device)
            print(f"[wam-server] action source: {self.action_source.upper()} "
                  f"(head {getattr(args, 'direct_ckpt', '')}, {self.direct.n_params / 1e3:.0f}k params)",
                  flush=True)
        self.grasp_planner = None
        if getattr(args, "grasp_planner", "primitive") == "wam":
            from uwam.arm_kin import load as load_arm_kin
            from uwam.grasp_planner import GraspCfg, GraspPlanner

            gm, gdn, gpn, gcfg = _load_model(Path(args.grasp_ckpt), self.device)
            assert gcfg.model.dyn_state_dim == 35, f"grasp core must be the 35-d arm+object model, got {gcfg.model.dyn_state_dim}"
            self.grasp_core, self.grasp_dyn_norm, self.grasp_pwm_norm = gm, gdn, gpn
            self.arm_kin = load_arm_kin(args.arm_kin, self.device)
            self.grasp_planner = GraspPlanner(gm, gdn, gpn, self.arm_kin, GraspCfg(), device=self.device)
            self.arm_frozen = False
            self.wrist_head = None
            wp = getattr(args, "wrist_pose_ckpt", "") or ""
            if wp:
                from percept.wrist_pose_model import WristPoseHead
                self.wrist_head = WristPoseHead(wp, device=self.device)
                self.obj_est = None
                self.obj_yaw_rel = None
                self.obj_source = os.environ.get("WAM_OBJ_SOURCE", "privileged")
                print(f"[wam-server] wrist-camera head {wp}: object source = {self.obj_source.upper()}"
                      f"{' (shadow: estimate logged, privileged pose drives)' if self.obj_source != 'wrist' else ''}", flush=True)
            self.box_head = None
            bh = os.environ.get("WAM_BOX_HEAD", "")
            if bh:
                from percept.wrist_pose_model import WristPoseHead
                self.box_head = WristPoseHead(bh, device=self.device)
                self.dest_est = None
                print(f"[wam-server] transfer destination source: FORWARD CAMERA head {bh} (the harness's destination "
                      "file is not read)", flush=True)
            self.direct_manip = None
            dm = os.environ.get("WAM_DIRECT_MANIP", "")
            if dm:
                from uwam.direct_manip import DirectManipPolicy
                self.direct_manip = DirectManipPolicy(dm, device=self.device)
                print(f"[wam-server] manipulation actions: WAM-DIRECT single pass ({dm}); no imagination search, no stage "
                      "program, no close gate -- the 13-D chunk (thrusters + joints + jaw) is the network output", flush=True)
            self.close_gate = None
            if os.environ.get("WAM_CLOSE_GATE", "hand") == "learned":
                import json as _json
                gp = os.environ.get("WAM_CLOSE_GATE_CKPT", "/hy-tmp/models/uwam/close_outcome.json")
                self.close_gate = _json.load(open(gp))
                self.close_gate.setdefault("threshold", float(os.environ.get("WAM_CLOSE_GATE_P", "0.7")))
                print(f"[wam-server] close gate: LEARNED outcome model {gp} (p >= {self.close_gate['threshold']}, cv AUROC "
                      f"{self.close_gate.get('cv_auroc', float('nan')):.3f}, n={self.close_gate.get('n')})", flush=True)
            print(f"[wam-server] grasp planner: WAM (core {args.grasp_ckpt}, fk {args.arm_kin}); "
                  "search/approach by the hull primitive, fine alignment + arm by imagination", flush=True)
        print(f"[wam-server] planner budget: n_samples={self.cfg.control.n_samples} "
              f"cem_iters={self.cfg.control.cem_iters} action_source={self.action_source}", flush=True)
        self._plan_ms = []
        ok = self.mpc.load_vel_ensemble(args.vel_ens) if args.vel_ens else False
        print(f"[wam-server] vel ensemble: {'loaded' if ok else 'MISSING (legacy gate)'}", flush=True)
        lib = []
        for d in (args.ou or "").split(","):
            if d and d.endswith(".npy") and Path(d).exists():
                lib.append(np.load(d).astype(np.float32))
            elif d and Path(d).exists():
                lib += [ep.pwm for ep in load_ou_split(Path(d))]
        if lib:
            self.mpc.set_library(np.concatenate(lib, axis=0))
            print(f"[wam-server] action library: {sum(len(x) for x in lib)} frames from {args.ou}", flush=True)
        self.vmax_scale = float(getattr(args, "vmax_scale", 1.0))
        if self.vmax_scale != 1.0:
            print(f"[wam-server] V_MAX scale {self.vmax_scale}: "
                  f"{ {k: round(v * self.vmax_scale, 2) for k, v in V_MAX.items()} }", flush=True)
        self.gate_cfg = GateCfg.from_calib(args.gate_calib) if args.gate_calib else None
        self.mpc.gate_cfg = self.gate_cfg
        print(f"[wam-server] gate: {self.gate_cfg}", flush=True)
        self.eval_root = Path(args.eval_root)
        self.task_code = None
        self.hold_anchor = getattr(args, "hold_anchor", "mixer")
        self.blind_est = os.environ.get("WAM_BLIND_EST", "model")
        if os.environ.get("WAM_PULSE_V"):
            self.PULSE_V = float(os.environ["WAM_PULSE_V"])
            print(f"[wam-server] primitive pulse speed PULSE_V = {self.PULSE_V:.3f} m/s (override)", flush=True)
        self.imu_vw = None
        if self.blind_est == "imu":
            print("[wam-server] blind estimator = IMU strapdown integration (ablation)", flush=True)
        self.dump_dir = Path(args.dump_dir) if getattr(args, "dump_dir", "") else None
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._reset_episode(None)

    def _reset_episode(self, traj_path):
        self.traj_path = traj_path
        self.waypoints = None
        self.judge = None
        self.wp_idx = 0
        self.mode = "nav"
        self.task_str = ""
        self.hist_s: list = []
        self.hist_a: list = []
        self.last_u = np.zeros(8, np.float32)
        self.est_sighted: list = []
        self.est_hist: list = []
        self.v_anchor = None
        self.alpha = 0.0
        self.cusum = CusumGate(self.gate_cfg)
        self.u_trans_hist: list = []
        self.u_trans_hold = None
        self.blind_prev = False
        self.imu_vw = None
        self.dr_pos = None
        self.dr_yaw = None
        self.last_pose = None
        self.gyro_bias_samples: list = []
        self.gyro_bias = 0.0
        self.prev_yaw_odom = None
        self.pz_pairs: list = []
        self.pz_fit = None
        self.brake_done = False
        self.coast_disp = None
        self.grasp_stage = "search"
        self.grasp_hold_ticks = 0
        self.joint_cmd = ARMED_JOINTS.copy()
        self.arm_frozen = False
        self.gp_acts = 0
        self.arm_move_ticks = 0
        self.obj_est = None
        self.obj_yaw_rel = None
        self._wo_act = -1
        self.lift_ticks = 0; self.jaw_eff_hist = []; self.grasp_retries = 0; self.lift_anchor = None; self.gate_hist = []; self.dest_est = None
        if getattr(self, "grasp_planner", None) is not None:
            self.grasp_planner.reset()
        self.pg_near_ticks = 0
        self.pg_grasped = False
        self.pg_release_ticks = 0
        self.pg_lost_ticks = 0
        self.pg_search = False
        self.pg_act_n = 0
        self.pg_mem = None
        self.pg_mem_ttl = 0
        self.pg_axis = None
        self.pg_side = 0
        self.pg_side_votes = 0
        self.pg_nav_mem = None
        self.pg_nav_mem_used = 0
        self.comp_parsed = False
        self.comp_depth = None
        self.comp_return = False
        self.comp_leg = 0
        self.comp_crumbs = []
        self.comp_near = 0
        self.comp_ret_idx = None
        self.att_recover = 0
        self.att_cool = 0
        self.grasp_near_ticks = 0
        self.grasp_pulse_steps = None
        if self.percept is not None:
            self.percept.reset()
        self._flush_dump(traj_path)
        self.dump_rec = {k: [] for k in ("dvl", "imu_av", "imu_la", "pressure", "dvl_h", "pwm")}
        vel_std = self.dyn_norm.std[0:3].astype(np.float64)
        self.vel_std = np.where(vel_std > 1e-6, vel_std, 1.0)
        if hasattr(self.mpc, "reset"):
            self.mpc.reset()

    def _newest_meta_csv(self):
        cands = glob.glob(str(self._scope() / "logs" / "episode_*_data.csv"))
        if not cands:
            return None
        now = time.time()
        fresh = [c for c in cands if now - os.path.getmtime(c) < 120]
        return max(fresh or cands, key=os.path.getmtime)

    def _scope(self) -> Path:
        tc = getattr(self, "task_code", None)
        return self.eval_root / tc if tc else self.eval_root / "*"

    def _maybe_new_episode(self, task_str):
        p, wps = _newest_traj(self._scope())
        mt = os.path.getmtime(p) if p else None
        key = (p, mt, self._newest_meta_csv())
        if key != getattr(self, "traj_key", None) or task_str != self.task_str:
            self.traj_key = key
            print(f"[wam-server] new episode: traj={p} task={task_str!r}", flush=True)
            self._reset_episode(p)
            self.task_str = task_str
            low = task_str.lower()
            for phrase, mode in MODE_BY_PHRASE:
                if phrase in low:
                    self.mode = mode
                    break
            self.waypoints = None if self._uses_live_target() else wps
            self.judge = self._read_judge_sidecar(p)
        elif self.waypoints is None and p is not None and not self._uses_live_target():
            self.traj_path, self.waypoints = p, wps
            self.judge = self._read_judge_sidecar(p)

    @staticmethod
    def _read_judge_sidecar(traj_path):
        if not traj_path:
            return None
        side = Path(traj_path).with_suffix(".json")
        if not side.exists() or time.time() - os.path.getmtime(side) > 600:
            return None
        try:
            j = json.loads(side.read_text())
        except Exception:
            return None
        pos = float(j.get("pos_tol", -1.0))
        yaw = float(j.get("yaw_tol", -1.0))
        if pos <= 0 and yaw <= 0 and not j.get("sequential", False):
            return None
        print(f"[wam-server] judge side-car: pos_tol={pos} yaw_tol={yaw} sequential={j.get('sequential')}",
              flush=True)
        return {"pos": pos if pos > 0 else None, "yaw": yaw if yaw > 0 else None}

    def _advance_limits(self):
        yaw_lim = ADVANCE_YAW.get(self.mode, 3.14)
        pos_lim = ADVANCE_TOL.get(self.mode, 0.9)
        j = getattr(self, "judge", None)
        if j:
            if j.get("pos"):
                pos_lim = min(pos_lim, 0.8 * j["pos"])
            if j.get("yaw"):
                yaw_lim = min(yaw_lim, 0.8 * j["yaw"])
        return pos_lim, yaw_lim

    def _uses_live_target(self):
        return self.mode in ("grasp", "transfer", "follow")

    def _percept_nav_active(self):
        return (self.percept is not None and getattr(self.percept, "nav_head", None) is not None
                and self.mode in ("nav", "scan", "inspect", "follow"))

    def _task_goal(self, pos, yaw, obs):
        if self.mode in ("grasp", "transfer"):
            if self.percept is not None:
                return self._percept_goal(obs)
            return self._grasp_goal(pos, yaw, obs)
        if self.percept is not None and getattr(self.percept, "nav_head", None) is not None:
            return self._percept_nav_goal(obs, pos, yaw)
        if self.mode == "follow":
            return self._follow_goal(pos, yaw, obs)
        return self._goal_from_waypoints(pos, yaw)

    def _percept_nav_goal(self, obs, pos=None, yaw=None):
        ego = self._get(obs, "video.ego")
        wrist = self._get(obs, "video.wrist")
        jp = self._get(obs, "state.joint_pos")
        pr = self._get(obs, "state.pressure")
        al = self._get(obs, "state.dvl_h")
        if ego is None or wrist is None:
            return np.zeros(3, np.float32), 0.0
        g, raw = self.percept.predict_nav(
            np.asarray(ego, np.uint8).reshape(240, 320, 3),
            np.asarray(wrist, np.uint8).reshape(240, 320, 3),
            np.zeros(5, np.float32) if jp is None else np.ravel(jp)[:5],
            0.0 if pr is None else float(np.ravel(pr)[0]),
            0.0 if al is None else float(np.ravel(al)[0]),
            self.task_str, yaw=None if yaw is None else float(yaw),
        )
        d = g[:3].astype(np.float64)
        if self.mode == "scan" and pos is not None and yaw is not None and getattr(self.percept, "kind", "") != "e2e":
            if self.pg_axis is None:
                self.pg_axis = (np.asarray(pos[:2], np.float64).copy(), float(yaw))
            ax0, ayaw = self.pg_axis
            cy, sy = np.cos(yaw), np.sin(yaw)
            gw = np.asarray(pos[:2], np.float64) + np.array([cy * d[0] - sy * d[1], sy * d[0] + cy * d[1]])
            lat = -np.sin(ayaw) * (gw[0] - ax0[0]) + np.cos(ayaw) * (gw[1] - ax0[1])
            if abs(lat) > 1.0:
                self.pg_side_votes += 1 if lat > 0 else -1
            if self.pg_side == 0 and abs(self.pg_side_votes) >= 3:
                self.pg_side = 1 if self.pg_side_votes > 0 else -1
                print(f"[wam-server] percept-nav: scan side latched ({'+' if self.pg_side > 0 else '-'})",
                      flush=True)
            if self.pg_side != 0 and lat * self.pg_side < -1.0:
                ex, ey = np.cos(ayaw), np.sin(ayaw)
                along = ex * (gw[0] - ax0[0]) + ey * (gw[1] - ax0[1])
                gw_m = ax0 + along * np.array([ex, ey]) - lat * np.array([-ey, ex])
                dw = gw_m - np.asarray(pos[:2], np.float64)
                d = d.copy()
                d[0], d[1] = cy * dw[0] + sy * dw[1], -sy * dw[0] + cy * dw[1]
                g = g.copy()
                g[5] = _yaw_wrap(2.0 * ayaw - (yaw + g[5])) - yaw
        d, comp_owned = self._composed_subgoal(d, pos, yaw, pr)
        dist = float(np.linalg.norm(d))
        if pos is not None and yaw is not None and self.mode == "nav" and not comp_owned:
            p_w = np.asarray(pos, np.float64)
            cy, sy = np.cos(yaw), np.sin(yaw)
            g_w = p_w + np.array([cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]])
            mem = self.pg_nav_mem
            arrive_r = 1.2 if self.mode in ("scan", "inspect") else 0.5
            if mem is not None:
                mem_d = float(np.linalg.norm(mem["pt"] - p_w))
                self.pg_nav_mem["ttl"] -= 1
                if self.pg_nav_mem["ttl"] <= 0 or mem_d < arrive_r:
                    self.pg_nav_mem = mem = None
            near_r = 1.0 if getattr(self.percept, "kind", "") == "e2e" else 2.0
            if mem is not None and dist > mem["r"] + 1.5:
                dw = mem["pt"] - p_w
                d = np.array([cy * dw[0] + sy * dw[1], -sy * dw[0] + cy * dw[1], dw[2]])
                dist = float(np.linalg.norm(d))
                self.pg_nav_mem_used += 1
            elif dist < near_r:
                self.pg_nav_mem = {"pt": g_w, "r": dist, "ttl": 15}
        vmax = V_MAX.get(self.mode, 0.25) * self.vmax_scale
        if dist < 1.5 and self.mode != "follow":
            vmax = min(vmax, max(0.06, 0.30 * dist))
        d_eff = d * max(1.0, 1.0 / dist) if dist > 0.25 else d
        v = KP_POS * d_eff
        if self.mode == "follow" and FOLLOW_CATCHUP > 1.0:
            vmax *= FOLLOW_CATCHUP
            v_now = getattr(self, "_v_now", None)
            if dist < 1.0 and v_now is not None:
                v = KP_POS * d + np.array([max(0.0, float(v_now[0])), 0.0, 0.0])
        n = float(np.linalg.norm(v))
        if n > vmax:
            v *= vmax / n
        v_goal = np.array([v[0], AXIS_SIGN[1] * v[1], AXIS_SIGN[2] * v[2]], np.float32)
        if self.mode in ("scan", "inspect"):
            yaw_err = float(np.clip(g[5], -1.0, 1.0))
        else:
            yaw_err = float(np.arctan2(d[1], d[0])) if dist > 1.0 else 0.0
        v_goal, yaw_err = self._turn_first(v_goal, yaw_err)
        self.pg_act_n += 1
        if self.pg_act_n % 5 == 1:
            print(f"[wam-server] percept-nav#{self.pg_act_n} mode={self.mode} |g|={dist:.2f} "
                  f"bearing={np.degrees(np.arctan2(d[1], d[0])):.0f}deg dz={d[2]:+.2f} "
                  f"yaw_err={yaw_err:+.2f} raw|g|={np.linalg.norm(raw[:3]):.2f} "
                  f"mem={'on' if self.pg_nav_mem is not None else 'off'}/{self.pg_nav_mem_used}", flush=True)
        return v_goal, yaw_err

    def _composed_subgoal(self, d, pos, yaw, pr):
        if self.mode != "nav":
            return d, False
        if not self.comp_parsed:
            low = self.task_str.lower()
            m = re.search(r"at a depth of ([0-9]+(?:\.[0-9]+)?) ?m", low)
            self.comp_depth = float(m.group(1)) if m else None
            self.comp_return = "come back to where you started" in low or "return to where you started" in low
            self.comp_parsed = True
            if self.comp_depth is not None or self.comp_return:
                print(f"[wam-server] composed goto: depth={self.comp_depth} return={self.comp_return}", flush=True)
        d = np.asarray(d, np.float64).copy()
        if self.comp_depth is not None and pr is not None:
            d[2] = self.comp_depth - float(np.ravel(pr)[0])
        if not self.comp_return or pos is None or yaw is None:
            return d, False
        p_w = np.asarray(pos, np.float64)
        if self.comp_leg == 0:
            if not self.comp_crumbs or np.linalg.norm(p_w - self.comp_crumbs[-1]) > 0.4:
                self.comp_crumbs.append(p_w.copy())
            travelled = float(np.linalg.norm(p_w[:2] - self.comp_crumbs[0][:2]))
            self.comp_near = self.comp_near + 1 if (np.linalg.norm(d) < 0.6 and travelled > 5.0) else 0
            if self.comp_near < 3:
                return d, False
            self.comp_leg = 1
            self.comp_ret_idx = len(self.comp_crumbs) - 1
            print(f"[wam-server] composed goto: sub-goal reached after {travelled:.1f} m, "
                  f"retracing {len(self.comp_crumbs)} breadcrumbs", flush=True)
        crumbs = self.comp_crumbs
        i = self.comp_ret_idx
        while i > 0 and np.linalg.norm(crumbs[i] - p_w) < 0.6:
            i -= 1
        self.comp_ret_idx = i
        j, acc = i, 0.0
        while j > 0 and acc < 1.5:
            acc += float(np.linalg.norm(crumbs[j] - crumbs[j - 1]))
            j -= 1
        dw = crumbs[j] - p_w
        cy, sy = np.cos(yaw), np.sin(yaw)
        d_ret = np.array([cy * dw[0] + sy * dw[1], -sy * dw[0] + cy * dw[1], dw[2]])
        if self.comp_depth is not None and pr is not None:
            d_ret[2] = self.comp_depth - float(np.ravel(pr)[0])
        return d_ret, True

    def _follow_goal(self, pos, yaw, obs):
        box = self._get(obs, "state.box_pos")
        box_rpy = self._get(obs, "state.box_rpy")
        if box is None:
            return self._goal_from_waypoints(pos, yaw)
        b = box.ravel().astype(np.float64)
        byaw = float(box_rpy.ravel()[2]) if box_rpy is not None else yaw
        off = np.array([3.0, 0.0, -0.5])
        cy, sy = np.cos(byaw), np.sin(byaw)
        tgt = b - np.array([cy * off[0] - sy * off[1], sy * off[0] + cy * off[1], off[2]])
        d_world = tgt - pos
        prev = getattr(self, "_boat_prev", None)
        v_boat = np.zeros(3)
        if prev is not None and 0.5 <= (time.time() - prev[1]) <= 5.0:
            v_boat = (b - prev[0]) / (time.time() - prev[1])
            v_boat[2] = 0.0
        self._boat_prev = (b.copy(), time.time())
        vmax = V_MAX["follow"] * self.vmax_scale
        v_world = KP_POS * d_world + v_boat
        n = float(np.linalg.norm(v_world))
        if n > vmax:
            v_world *= vmax / n
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        v_goal = np.array([cy_ * v_world[0] + sy_ * v_world[1], AXIS_SIGN[1] * (-sy_ * v_world[0] + cy_ * v_world[1]),
                           AXIS_SIGN[2] * v_world[2]], np.float32)
        bearing = float(np.arctan2(b[1] - pos[1], b[0] - pos[0]))
        yaw_err = float(np.clip(_yaw_wrap(bearing - yaw), -1.0, 1.0))
        return v_goal, yaw_err

    @staticmethod
    def _get(obs, key, default=None):
        v = obs.get(key, default)
        return None if v is None else np.asarray(v, np.float32)

    def _frames_from_obs(self, obs):
        hd = self._get(obs, "state.hist_dvl")
        if hd is None:
            return None, None, None
        hav = self._get(obs, "state.hist_imu_av")
        hla = self._get(obs, "state.hist_imu_la")
        hpr = self._get(obs, "state.hist_pressure") * 1e4
        hal = self._get(obs, "state.hist_dvl_h")
        hpw = self._get(obs, "state.hist_pwm")
        frames = np.concatenate([hd, hav, hla, hpr, hal, hpw], axis=-1).astype(np.float32)
        hist_a = np.vstack([hpw[1:], self.last_u[None]]).astype(np.float32)
        dvl_rows_zero = np.all(np.abs(hd) < 1e-9, axis=1)
        return frames, hist_a, dvl_rows_zero

    def _flush_dump(self, traj_path):
        dump_dir = getattr(self, "dump_dir", None)
        rec = getattr(self, "dump_rec", None)
        if dump_dir is None or rec is None or not rec.get("pwm"):
            return
        n = len(rec["pwm"])
        path = dump_dir / f"task_{int(time.time())}_{n}.npz"
        np.savez_compressed(
            path,
            pwm=np.stack(rec["pwm"]).astype(np.float32),
            eta=np.ones((n, 8), np.float32),
            pwm_is_commanded=np.bool_(True),
            dvl=np.stack(rec["dvl"]).astype(np.float32),
            imu_av=np.stack(rec["imu_av"]).astype(np.float32),
            imu_la=np.stack(rec["imu_la"]).astype(np.float32),
            pressure=np.stack(rec["pressure"]).astype(np.float32).reshape(n, 1),
            dvl_h=np.stack(rec["dvl_h"]).astype(np.float32).reshape(n, 1),
            timestamp=np.arange(n, dtype=np.float64) * DT,
            dt=np.float64(DT),
        )
        print(f"[wam-server] dumped {n} ticks -> {path} (prev={traj_path})", flush=True)

    def _record_dump(self, frames, hist_a):
        if getattr(self, "dump_dir", None) is None:
            return
        for i in range(len(frames)):
            if not np.any(frames[i, 0:9]):
                continue
            self.dump_rec["dvl"].append(frames[i, 0:3].copy())
            self.dump_rec["imu_av"].append(frames[i, 3:6].copy())
            self.dump_rec["imu_la"].append(frames[i, 6:9].copy())
            self.dump_rec["pressure"].append(float(frames[i, 9]))
            self.dump_rec["dvl_h"].append(float(frames[i, 10]))
            self.dump_rec["pwm"].append(hist_a[i, :8].copy())

    @staticmethod
    def _rot_zyx(r, p, y):
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])

    def _imu_integrate(self, frames, dvl_zero, imu_yaw, obs):
        att = self._attitude(obs) if obs is not None else None
        roll, pitch = (float(att[0]), float(att[1])) if att is not None else (0.0, 0.0)
        yaw_now = float(imu_yaw) if imu_yaw is not None else (
            float(att[2]) if att is not None else (self.dr_yaw if self.dr_yaw is not None else 0.0))
        L = len(frames)
        gz = frames[:, 5].astype(np.float64) - self.gyro_bias
        gx = frames[:, 3].astype(np.float64)
        gy = frames[:, 4].astype(np.float64)
        yaws, rolls, pitches = np.empty(L), np.empty(L), np.empty(L)
        yaws[-1], rolls[-1], pitches[-1] = yaw_now, roll, pitch
        for i in range(L - 2, -1, -1):
            yaws[i] = yaws[i + 1] - gz[i + 1] * DT
            rolls[i] = rolls[i + 1] - gx[i + 1] * DT
            pitches[i] = pitches[i + 1] - gy[i + 1] * DT
        hr = self._get(obs, "state.hist_imu_rpy") if obs is not None else None
        if hr is not None and len(hr) == L:
            ok = np.abs(hr).sum(axis=1) > 1e-9
            rolls[ok], pitches[ok], yaws[ok] = hr[ok, 0], hr[ok, 1], hr[ok, 2]
        g_w = np.array([0.0, 0.0, 9.81])
        start = 0
        if self.imu_vw is None:
            valid = np.nonzero(~np.asarray(dvl_zero, bool))[0] if dvl_zero is not None else np.array([], int)
            if len(valid) == 0:
                v0 = self.last_dvl_v if getattr(self, "last_dvl_v", None) is not None else np.zeros(3)
                k = -1
            else:
                k = int(valid[-1]); v0 = frames[k, 0:3].astype(np.float64)
            v0 = np.array([v0[0], AXIS_SIGN[1] * v0[1], AXIS_SIGN[2] * v0[2]])
            kk = max(k, 0)
            self.imu_vw = self._rot_zyx(rolls[kk], pitches[kk], yaws[kk]) @ v0
            start = k + 1
        for i in range(start, L):
            a_w = self._rot_zyx(rolls[i], pitches[i], yaws[i]) @ frames[i, 6:9].astype(np.float64) + g_w
            self.imu_vw = self.imu_vw + a_w * DT
        vb = self._rot_zyx(roll, pitch, yaw_now).T @ self.imu_vw
        return np.array([vb[0], AXIS_SIGN[1] * vb[1], AXIS_SIGN[2] * vb[2]], np.float32)

    def _v_est(self, frames, hist_a):
        if self.blind_est == "imu":
            v = self.imu_v_body if self.imu_vw is not None else frames[-1, 0:3]
            return np.asarray(v, np.float32), None, None
        ens = self.mpc.estimate_velocity_ens(frames, hist_a)
        if ens is not None:
            return ens[0].astype(np.float32), ens[1], ens[2]
        v = self.mpc.estimate_velocity(frames, hist_a)
        return (None if v is None else v.astype(np.float32)), None, None

    def _world_to_vgoal(self, d_world, yaw, vmax):
        v_world = KP_POS * d_world
        n = float(np.linalg.norm(v_world))
        if n > vmax:
            v_world *= vmax / n
        cy, sy = np.cos(yaw), np.sin(yaw)
        vx = cy * v_world[0] + sy * v_world[1]
        vy = -sy * v_world[0] + cy * v_world[1]
        vz = v_world[2]
        return np.array([vx, AXIS_SIGN[1] * vy, AXIS_SIGN[2] * vz], np.float32)

    def _box_in_frame(self, obs, pos, yaw):
        if pos is None:
            return obs
        rel = ryaw = None
        if getattr(self, "wrist_head", None) is not None and getattr(self, "obj_source", "") == "wrist":
            if self.obj_est is None:
                o = dict(obs); o.pop("state.box_pos", None); o.pop("state.box_rpy", None)
                return o
            rel = np.asarray(self.obj_est[0], np.float64).copy()
            ryaw = self.obj_yaw_rel
            k = int(getattr(self, "grasp_retries", 0))
            if k > 0:
                seq = [(0.0, 0.01), (0.0, -0.01), (0.0, 0.02), (0.0, -0.02), (0.015, 0.0), (-0.015, 0.0)]
                dx, dy = seq[(k - 1) % len(seq)]
                rel[0] += dx; rel[1] += dy
        else:
            r = obs.get("state.box_rel")
            if r is None:
                return obs
            rel = np.ravel(np.asarray(r, np.float64))[:3]
            ry = obs.get("state.box_rel_yaw")
            ryaw = None if ry is None else float(np.ravel(ry)[0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        box = np.asarray(pos, np.float64) + np.array([cy * rel[0] - sy * rel[1], sy * rel[0] + cy * rel[1], rel[2]])
        o = dict(obs)
        o["state.box_pos"] = box.astype(np.float32).reshape(1, -1)
        if ryaw is not None:
            rpy = obs.get("state.box_rpy")
            r = np.ravel(np.asarray(rpy, np.float64)).copy() if rpy is not None else np.zeros(3)
            r[2] = _yaw_wrap(yaw + float(ryaw))
            o["state.box_rpy"] = r.astype(np.float32).reshape(1, -1)
        return o

    def _gripper_world(self, obs, pos, yaw):
        ee = self._get(obs, "state.ee_pose")
        if ee is None:
            return None
        e = np.ravel(ee).astype(np.float64)[:3]
        body = np.array([e[2] + 0.196, -e[1] - 0.084, e[0] + 0.145 + 0.05])
        cy, sy = np.cos(yaw), np.sin(yaw)
        return pos + np.array([cy * body[0] - sy * body[1],
                               sy * body[0] + cy * body[1],
                               body[2]])

    def _standoff_pose(self, box_xyz, box_yaw, key):
        off = GRASP_STANDOFF[key]
        cy, sy = np.cos(box_yaw), np.sin(box_yaw)
        world = np.array([
            cy * off[0] - sy * off[1],
            sy * off[0] + cy * off[1],
            off[2],
        ], np.float64)
        return box_xyz - world, box_yaw

    def _update_dest_estimate(self, obs, pos, yaw):
        import torch
        ego = self._get(obs, "video.ego")
        if ego is None:
            return
        pred = self.box_head.predict(np.asarray(ego, np.uint8).reshape(240, 320, 3), None, np.zeros(5, np.float32))
        rel = np.asarray(pred["obj_ee"], np.float64); sig = np.asarray(pred["sigma"], np.float64) + 0.02
        if not np.isfinite(rel).all() or np.linalg.norm(rel) > 6.0 or sig.max() > 1.5:
            return
        cy, sy = np.cos(yaw), np.sin(yaw)
        meas = np.asarray(pos, np.float64) + np.array([cy * rel[0] - sy * rel[1], sy * rel[0] + cy * rel[1], rel[2]])
        dyaw = _yaw_wrap(yaw + float(pred.get("yaw_mod_pi", 0.0)))
        if self.dest_est is None:
            self.dest_est = (meas, sig, dyaw)
        else:
            m0, s0, y0 = self.dest_est
            s0 = s0 + 0.005
            w = s0 ** 2 / (s0 ** 2 + sig ** 2)
            a0, a1 = 2 * y0, 2 * dyaw
            v = 0.7 * np.array([np.sin(a0), np.cos(a0)]) + 0.3 * np.array([np.sin(a1), np.cos(a1)])
            self.dest_est = ((1 - w) * m0 + w * meas, np.sqrt((1 - w) * s0 ** 2), 0.5 * float(np.arctan2(v[0], v[1])))
        self._n_de = getattr(self, "_n_de", 0) + 1
        if self._n_de % 10 == 1:
            truth = self._dest_from_file()
            err = "" if truth is None else f" err_vs_file=({self.dest_est[0][0]-truth[0]:+.2f},{self.dest_est[0][1]-truth[1]:+.2f},{self.dest_est[0][2]-truth[2]:+.2f}) m"
            print(f"[wam-server] dest-est#{self._n_de} sigma={np.round(self.dest_est[1], 2).tolist()} dist={np.linalg.norm(rel):.2f}{err}", flush=True)

    def _newest_dest(self):
        if getattr(self, "box_head", None) is not None:
            if getattr(self, "dest_est", None) is None or self.dest_est[1].max() > 0.6:
                return None
            m, _, y = self.dest_est
            return np.array([m[0], m[1], m[2], y], np.float64)
        return self._dest_from_file()

    def _dest_from_file(self):
        cands = glob.glob(str(self._scope() / "logs" / "episode_*_desti.npy"))
        if not cands:
            return None
        now = time.time()
        fresh = [p for p in cands if now - os.path.getmtime(p) < 180]
        p = max(fresh or cands, key=os.path.getmtime)
        try:
            arr = np.load(p, allow_pickle=True)
            return np.asarray(arr, np.float64).reshape(-1)[:4]
        except Exception:
            return None

    def _goal_from_waypoints(self, pos, yaw):
        if self.waypoints is None or len(self.waypoints) == 0:
            return np.zeros(3, np.float32), 0.0
        pos_lim, yaw_lim = self._advance_limits()
        single = self.mode in ("scan", "inspect")
        while self.wp_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.wp_idx]
            pos_ok = np.linalg.norm(wp[:3] - pos) <= pos_lim
            yaw_ok = abs(_yaw_wrap(float(wp[3]) - yaw)) <= yaw_lim
            if pos_ok and yaw_ok:
                self.wp_idx += 1
                if single:
                    break
            else:
                break
        wp = self.waypoints[self.wp_idx]
        d_world = wp[:3] - pos
        dist = float(np.linalg.norm(d_world))
        vmax = V_MAX.get(self.mode, 0.25) * self.vmax_scale
        v_goal = self._world_to_vgoal(d_world, yaw, vmax)
        if self.mode in ("inspect", "scan", "grasp", "transfer"):
            yaw_ref = float(wp[3])
        else:
            yaw_ref = np.arctan2(d_world[1], d_world[0]) if dist > 1.0 else yaw
        return self._turn_first(v_goal, float(_yaw_wrap(yaw_ref - yaw)))

    SADDLE, TURN_ONLY = 2.6, 1.6

    def _turn_first(self, v_goal, yaw_err_raw):
        e = float(_yaw_wrap(float(yaw_err_raw)))
        if e < -self.SADDLE:
            e += 2.0 * np.pi
        if abs(e) > self.TURN_ONLY:
            return np.zeros(3, np.float32), float(np.clip(e, -1.0, 1.0))
        return v_goal, float(np.clip(e, -1.0, 1.0))

    def _imu_yaw(self, obs):
        r = self._get(obs, "state.imu_rpy")
        return None if r is None else float(np.ravel(r)[2])

    def _update_sighted_calib(self, frames, pos, yaw):
        gz = float(np.mean(frames[:, 5]))
        if self.prev_yaw_odom is not None:
            true_rate = _yaw_wrap(yaw - self.prev_yaw_odom) / (DT * CHUNK)
            if abs(true_rate) < 1.0:
                self.gyro_bias_samples.append(gz - true_rate)
                self.gyro_bias_samples = self.gyro_bias_samples[-40:]
                if len(self.gyro_bias_samples) >= 3:
                    self.gyro_bias = float(np.median(self.gyro_bias_samples))
        self.prev_yaw_odom = float(yaw)
        p = float(frames[-1, 9]) / 1e4
        self.pz_pairs.append((p, float(pos[2])))
        self.pz_pairs = self.pz_pairs[-80:]
        if len(self.pz_pairs) >= 5:
            ps = np.array([q[0] for q in self.pz_pairs])
            zs = np.array([q[1] for q in self.pz_pairs])
            if ps.std() > 0.02:
                k, b = np.polyfit(ps, zs, 1)
                if 0.8 < k < 1.2:
                    self.pz_fit = (float(k), float(b))
            if self.pz_fit is None:
                self.pz_fit = (0.989, float(zs[-1] - 0.989 * ps[-1]))

    def _transport_mem(self, obs):
        if self.pg_mem is None:
            return
        hd = self._get(obs, "state.hist_dvl")
        hav = self._get(obs, "state.hist_imu_av")
        if hd is None or hav is None:
            return
        v_body = np.stack([hd[:, 0], AXIS_SIGN[1] * hd[:, 1], AXIS_SIGN[2] * hd[:, 2]], 1)
        disp = v_body.sum(axis=0) * DT
        dyaw = float(hav[:, 2].mean()) * DT * len(hav)
        g = self.pg_mem
        p = g[:3] - disp
        cy, sy = np.cos(-dyaw), np.sin(-dyaw)
        self.pg_mem[:3] = np.array([cy * p[0] - sy * p[1], sy * p[0] + cy * p[1], p[2]])
        self.pg_mem[5] = _yaw_wrap(g[5] - dyaw)

    def _percept_goal(self, obs):
        ego = self._get(obs, "video.ego")
        wrist = self._get(obs, "video.wrist")
        jp = self._get(obs, "state.joint_pos")
        pr = self._get(obs, "state.pressure")
        al = self._get(obs, "state.dvl_h")
        if ego is None or wrist is None:
            return np.zeros(3, np.float32), 0.0
        self._transport_mem(obs)
        g, grip_p = self.percept.predict(
            np.asarray(ego, np.uint8).reshape(240, 320, 3),
            np.asarray(wrist, np.uint8).reshape(240, 320, 3),
            np.zeros(5, np.float32) if jp is None else np.ravel(jp)[:5],
            0.0 if pr is None else float(np.ravel(pr)[0]),
            0.0 if al is None else float(np.ravel(al)[0]),
            self.task_str,
        )
        dist = float(np.linalg.norm(g[:3]))
        if not self.pg_grasped:
            self.pg_near_ticks = self.pg_near_ticks + 1 if grip_p > 0.5 else 0
            if self.pg_near_ticks >= 2:
                self.joint_cmd = ARMED_JOINTS.copy()
                self.joint_cmd[0] = GRIPPER_CLOSE
                self.pg_grasped = True
                print(f"[wam-server] percept: grip_p={grip_p:.2f} dist={dist:.3f} -> close",
                      flush=True)
        elif self.mode == "transfer":
            self.pg_release_ticks = self.pg_release_ticks + 1 if grip_p < 0.3 else 0
            if self.pg_release_ticks >= 3 and self.joint_cmd[0] > 0.0:
                self.joint_cmd[0] = 0.0
                print(f"[wam-server] percept: grip_p={grip_p:.2f} -> release", flush=True)
        fresh_ok = dist >= 0.15 or grip_p >= 0.15
        if fresh_ok:
            self.pg_mem = g.copy()
            self.pg_mem_ttl = 12
        track = g
        using_mem = False
        if not self.pg_grasped:
            lost_now = dist < 0.12 and grip_p < 0.10
            if lost_now and self.pg_mem is not None and self.pg_mem_ttl > 0:
                self.pg_mem_ttl -= 1
                track = self.pg_mem
                using_mem = True
                self.pg_lost_ticks = 0
            else:
                self.pg_lost_ticks = self.pg_lost_ticks + 1 if lost_now else 0
            if not self.pg_search and self.pg_lost_ticks >= 3:
                self.pg_search = True
                self.pg_search_ticks = 0
                print(f"[wam-server] percept: LOST (|g|={dist:.3f}, grip_p={grip_p:.2f}) "
                      "-> back up + search", flush=True)
            elif self.pg_search and fresh_ok:
                self.pg_search = False
                self.pg_lost_ticks = 0
                print(f"[wam-server] percept: reacquired (|g|={dist:.3f}, grip_p={grip_p:.2f})",
                      flush=True)
        else:
            self.pg_search = False
        self.pg_act_n += 1
        if self.pg_act_n % 5 == 1:
            print(f"[wam-server] percept#{self.pg_act_n} |g|={dist:.3f} grip_p={grip_p:.2f} "
                  f"mem={using_mem}/{self.pg_mem_ttl} search={self.pg_search} "
                  f"grasped={self.pg_grasped}", flush=True)
        if self.pg_search:
            self.pg_search_ticks = getattr(self, "pg_search_ticks", 0) + 1
            climb = AXIS_SIGN[2] * -0.04
            if self.pg_search_ticks <= 4:
                return np.array([-0.08, 0.0, climb], np.float32), 0.0
            return np.array([0.0, 0.0, climb], np.float32), 0.30
        tdist = float(np.linalg.norm(track[:3]))
        vmax = min(V_MAX.get(self.mode, 0.18), max(0.03, 0.35 * tdist))
        v = KP_POS * track[:3]
        n = float(np.linalg.norm(v))
        if n > vmax:
            v *= vmax / n
        v_goal = np.array([v[0], AXIS_SIGN[1] * v[1], AXIS_SIGN[2] * v[2]], np.float32)
        yaw_err = float(np.clip(track[5], -1.0, 1.0))
        return v_goal, yaw_err

    PULSE_V = 0.03

    def _grasp_goal(self, pos, yaw, obs):
        box = self._get(obs, "state.box_pos")
        box_rpy = self._get(obs, "state.box_rpy")
        if box is None:
            return np.zeros(3, np.float32), 0.0
        box_xyz = box.ravel().astype(np.float64)
        box_yaw = float(box_rpy.ravel()[2]) if box_rpy is not None else yaw
        cands = [box_yaw, _yaw_wrap(box_yaw + np.pi)]
        box_yaw = min(cands, key=lambda a: abs(_yaw_wrap(a - yaw)))
        stage = self.grasp_stage
        key = {"search": "search", "approach": "approach", "grasp": "grasp",
               "lift": "lift", "carry": "lift"}.get(stage, "search")
        if stage != "carry":
            self.carry_yaw = None
        if stage == "carry":
            dest = self._newest_dest()
            if dest is not None:
                grip_now = self._gripper_world(obs, pos, yaw)
                off_w = (grip_now - pos) if grip_now is not None else np.zeros(3)
                cy_, sy_ = np.cos(yaw), np.sin(yaw)
                off_b = np.array([cy_ * off_w[0] + sy_ * off_w[1], -sy_ * off_w[0] + cy_ * off_w[1], off_w[2]])
                if getattr(self, "carry_yaw", None) is None:
                    d0 = dest[:2] - (np.asarray(pos, np.float64)[:2] + off_w[:2])
                    self.carry_yaw = float(np.arctan2(d0[1], d0[0])) if np.hypot(*d0) > 0.3 else float(yaw)
                    print(f"[wam-server] carry: heading frozen at {self.carry_yaw:+.2f} rad "
                          f"(hull yaw {yaw:+.2f}, destination {np.hypot(*d0):.2f} m away)", flush=True)
                yaw_ref = self.carry_yaw
                cf, sf = np.cos(yaw_ref), np.sin(yaw_ref)
                hull_off = np.array([cf * off_b[0] - sf * off_b[1], sf * off_b[0] + cf * off_b[1], off_b[2]])
                tgt = dest[:3] - np.array([0.0, 0.0, 0.5]) - hull_off
                if float(np.hypot(*(tgt[:2] - np.asarray(pos, np.float64)[:2]))) > 0.25:
                    tgt[2] -= 0.25
            else:
                tgt, yaw_ref = self._standoff_pose(box_xyz, box_yaw, "lift")
                if getattr(self, "box_head", None) is not None:
                    if getattr(self, "lift_anchor", None) is not None:
                        tgt, _ = self._standoff_pose(self.lift_anchor[0], self.lift_anchor[1], "lift")
                    yaw_ref = _yaw_wrap(yaw + 0.35)
        else:
            if stage in ("lift", "carry") and getattr(self, "lift_anchor", None) is not None:
                box_xyz, box_yaw = self.lift_anchor
            tgt, yaw_ref = self._standoff_pose(box_xyz, box_yaw, key)
        err = tgt - pos
        grip_w = self._gripper_world(obs, pos, yaw)
        grip_d = None
        if grip_w is not None and stage in ("approach", "grasp"):
            hover = 0.06 if stage == "approach" else 0.0
            err = (box_xyz - np.array([0.0, 0.0, hover])) - grip_w
            grip_d = float(np.linalg.norm(box_xyz - grip_w))
        dist = float(np.linalg.norm(err))
        yaw_err_abs = abs(_yaw_wrap(yaw_ref - yaw))
        arrive = 0.12 if stage in ("search", "lift", "carry") else 0.08
        aligned = False
        near = False
        dxy_b = None
        if grip_w is not None:
            dw = box_xyz - grip_w
            cy_, sy_ = np.cos(yaw), np.sin(yaw)
            dxy_b = np.array([cy_ * dw[0] + sy_ * dw[1], -sy_ * dw[0] + cy_ * dw[1]])
            dxy = np.abs(dxy_b)
            v_now = self._get(obs, "state.dvl_v")
            settled = float(np.linalg.norm(np.zeros(3) if v_now is None else np.ravel(v_now)[:3])) < 0.03
            aligned = bool(dxy[0] < 0.030 and dxy[1] < 0.007
                           and grip_d is not None and grip_d < 0.035 and yaw_err_abs < 0.2 and settled)
            near = bool(dxy[0] < 0.030 and dxy[1] < 0.010
                        and grip_d is not None and grip_d < 0.038 and yaw_err_abs < 0.25)
            if getattr(self, "close_gate", None) is not None:
                g = self.close_gate
                e = -dw
                dist_now = float(np.linalg.norm(e))
                self.gate_hist = (getattr(self, "gate_hist", []) + [(dist_now, float(e[0]), float(e[1]))])[-3:]
                if len(self.gate_hist) >= 3:
                    d1, x1, y1 = self.gate_hist[0]
                else:
                    d1, x1, y1 = self.gate_hist[0]
                feats = np.array([abs(e[0]), abs(e[1]), e[2], dist_now, abs(dist_now - d1), float(np.hypot(e[0] - x1, e[1] - y1))])
                z = (feats - np.asarray(g["mu"])) / np.asarray(g["sd"])
                p_grip = float(1.0 / (1.0 + np.exp(-(np.dot(z, np.asarray(g["w"])[:-1]) + g["w"][-1]))))
                self.p_grip = p_grip
                aligned = bool(p_grip >= g.get("threshold", 0.7) and settled and yaw_err_abs < 0.25 and dist_now < 0.06)
                near = bool(p_grip >= 0.5 and yaw_err_abs < 0.25 and dist_now < 0.06)
        if stage in ("lift", "carry"):
            eff = self._get(obs, "state.joint_effort")
            e0 = float(np.ravel(eff)[0]) if eff is not None and np.ravel(eff).size else float("nan")
            self.jaw_eff_hist = (getattr(self, "jaw_eff_hist", []) + [e0])[-8:]
            self.lift_ticks = getattr(self, "lift_ticks", 0) + 1
            if self.lift_ticks >= 8 and len(self.jaw_eff_hist) >= 8 and np.isfinite(self.jaw_eff_hist).all() \
                    and float(np.max(self.jaw_eff_hist)) < 0.02:
                self.grasp_retries = getattr(self, "grasp_retries", 0) + 1
                print(f"[wam-server] grasp: jaw effort max {np.max(self.jaw_eff_hist):.3f} over 4 s after close -> EMPTY, "
                      f"reopen and retry (#{self.grasp_retries})", flush=True)
                self.joint_cmd[0] = 0.0
                self.arm_frozen = False
                self.grasp_stage, self.grasp_hold_ticks, self.grasp_near_ticks = "approach", 0, 0
                self.lift_anchor = None
                self.lift_ticks = 0; self.jaw_eff_hist = []
                if getattr(self, "grasp_planner", None) is not None:
                    self.grasp_planner.reset()
                stage = "approach"
        else:
            self.lift_ticks = 0; self.jaw_eff_hist = []
        self.grasp_near_ticks = (self.grasp_near_ticks + 1) if (stage == "grasp" and near) else 0
        if stage == "grasp" and (aligned or self.grasp_near_ticks >= 3):
            self.grasp_hold_ticks += 1
            if self.grasp_hold_ticks >= 1:
                if getattr(self, "grasp_planner", None) is None:
                    self.joint_cmd = ARMED_JOINTS.copy()
                self.joint_cmd[0] = GRIPPER_CLOSE
                self.arm_frozen = True
                print(f"[wam-server] grasp: aligned (d={grip_d:.3f} dxy={dxy[0]:.3f},"
                      f"{dxy[1]:.3f}{', p_grip=%.2f' % self.p_grip if getattr(self, 'close_gate', None) is not None else ''}) -> close", flush=True)
                self.lift_anchor = (box_xyz.copy(), box_yaw)
            self.grasp_stage, self.grasp_hold_ticks = "lift", 0
        elif dist < arrive and yaw_err_abs < 0.25:
            self.grasp_hold_ticks += 1
            if stage == "search":
                self.grasp_stage, self.grasp_hold_ticks = "approach", 0
            elif stage == "approach":
                self.grasp_stage, self.grasp_hold_ticks = "grasp", 0
                self.gp_acts = 0
                if getattr(self, "grasp_planner", None) is not None:
                    self.grasp_planner.reset()
            elif stage == "grasp" and self.grasp_hold_ticks >= 12:
                self.grasp_stage, self.grasp_hold_ticks = "approach", 0
            elif stage == "lift" and self.mode == "transfer":
                self.grasp_stage, self.grasp_hold_ticks = "carry", 0
            elif stage == "carry" and self.grasp_hold_ticks >= 2:
                self.joint_cmd[0] = 0.0
        elif stage == "grasp":
            self.grasp_hold_ticks += 1
            if self.grasp_hold_ticks >= 12:
                self.grasp_stage, self.grasp_hold_ticks = "approach", 0
        jaw = float(self.joint_cmd[0])
        if not getattr(self, "arm_frozen", False):
            self.joint_cmd = 0.7 * self.joint_cmd + 0.3 * ARMED_JOINTS
        self.joint_cmd[0] = jaw
        vmax = V_MAX.get(self.mode, 0.18)
        self.grasp_pulse_steps = None
        if self.grasp_stage in ("approach", "grasp"):
            if dist < 0.10:
                vmax = self.PULSE_V
                self.grasp_pulse_steps = int(np.clip(round(dist / (self.PULSE_V * 0.1)), 1, CHUNK - 4))
            else:
                vmax = min(vmax, max(0.03, 0.30 * dist))
        err_cmd = err
        if self.grasp_stage == "carry" and getattr(self, "carry_yaw", None) is not None and err[2] < -0.08:
            err_cmd = np.array([0.25 * err[0], 0.25 * err[1], err[2]])
        v_goal = self._world_to_vgoal(err_cmd, yaw, vmax)
        yaw_err = float(_yaw_wrap(yaw_ref - yaw))
        if self.grasp_stage == "carry" and getattr(self, "carry_yaw", None) is not None:
            v_goal, yaw_err = self._turn_first(v_goal, yaw_err)
        else:
            yaw_err = float(np.clip(yaw_err, -1.0, 1.0))
        self._n_grasp = getattr(self, "_n_grasp", 0) + 1
        if self._n_grasp % 5 == 1:
            gd = "n/a" if grip_d is None else f"{grip_d:.3f}"
            print(f"[wam-server] grasp#{self._n_grasp} stage={self.grasp_stage} "
                  f"hull_err={dist:.3f} grip_d={gd} vmax={vmax:.3f} "
                  f"jaw={self.joint_cmd[0]:.3f}", flush=True)
        return v_goal, yaw_err

    GRASP_CLOSE_TICKS = 2
    GRASP_RETRY_ACTS = 60

    def _grasp_frames(self, obs, frames, hist_a, obj_body):
        L = frames.shape[0]
        hjp = self._get(obs, "state.hist_joint_pos")
        hjv = self._get(obs, "state.hist_joint_v")
        jp = self._get(obs, "state.joint_pos")
        jp = np.zeros(5, np.float32) if jp is None else np.ravel(jp)[:5]
        hjp = np.tile(jp[None], (L, 1)) if hjp is None else np.asarray(hjp, np.float32).reshape(L, -1)[:, :5]
        hjv = np.zeros((L, 5), np.float32) if hjv is None else np.asarray(hjv, np.float32).reshape(L, -1)[:, :5]
        ob = np.zeros((L, 6), np.float32); ob[:, :3] = obj_body; ob[:, 3] = 1.0; ob[:, 5] = 1.0
        hs = np.concatenate([frames, hjp, hjv, ob], axis=-1).astype(np.float32)
        jc = np.tile(np.asarray(self.joint_cmd, np.float32)[None], (L, 1))
        ha = np.concatenate([hist_a, jc], axis=-1).astype(np.float32)
        return hs, ha

    def _wrist_object(self, obs, obj_priv=None):
        import torch

        if getattr(self, "_wo_act", -1) == getattr(self, "_n_act_total", 0) and self.obj_est is not None:
            return self.obj_est[0]
        wrist = self._get(obs, "video.wrist"); ego = self._get(obs, "video.ego")
        jp = self._get(obs, "state.joint_pos")
        if wrist is None or jp is None:
            return obj_priv if self.obj_est is None else self.obj_est[0]
        self._wo_act = getattr(self, "_n_act_total", 0)
        q = np.ravel(np.asarray(jp, np.float32))[:5]
        pred = self.wrist_head.predict(np.asarray(wrist, np.uint8).reshape(240, 320, 3),
                                       None if ego is None else np.asarray(ego, np.uint8).reshape(240, 320, 3), q)
        with torch.no_grad():
            ee_pos, ee_R = self.arm_kin(torch.as_tensor(q, device=self.device).reshape(1, 5))
        ee_pos = ee_pos[0].cpu().numpy(); ee_R = ee_R[0].cpu().numpy()
        meas = (ee_pos + ee_R @ pred["obj_ee"]).astype(np.float32)
        sig = np.asarray(pred["sigma"], np.float32)
        sig_b = np.abs(ee_R) @ sig + 1e-3
        if self.obj_est is None:
            self.obj_est = (meas, sig_b)
        else:
            m0, s0 = self.obj_est
            s0 = s0 + 0.01
            w = s0 ** 2 / (s0 ** 2 + sig_b ** 2)
            self.obj_est = ((1 - w) * m0 + w * meas, np.sqrt((1 - w) * s0 ** 2))
        psi = float(pred.get("yaw_mod_pi", 0.0))
        if getattr(self, "obj_yaw_rel", None) is None:
            self.obj_yaw_rel = psi
        else:
            a0, a1 = 2 * self.obj_yaw_rel, 2 * psi
            m = 0.6 * np.array([np.sin(a0), np.cos(a0)]) + 0.4 * np.array([np.sin(a1), np.cos(a1)])
            self.obj_yaw_rel = 0.5 * float(np.arctan2(m[0], m[1]))
        self._n_wo = getattr(self, "_n_wo", 0) + 1
        if self._n_wo % 3 == 1 and obj_priv is not None:
            err = self.obj_est[0] - obj_priv
            merr = meas - obj_priv
            dist = float(np.linalg.norm(obj_priv - ee_pos))
            print(f"[wam-server] wrist-object#{self._n_wo} meas_sigma={np.round(sig_b, 3).tolist()} "
                  f"est_err_vs_truth=({err[0]:+.3f},{err[1]:+.3f},{err[2]:+.3f}) m "
                  f"meas_err=({merr[0]:+.3f},{merr[1]:+.3f},{merr[2]:+.3f}) dist={dist:.3f} stage={self.grasp_stage}", flush=True)
        return self.obj_est[0]

    def _grasp_arm_act(self, frames, hist_a, obs, pos, rpy, att):
        box = self._get(obs, "state.box_pos")
        if box is None:
            return None
        yaw = float(rpy[2])
        v_goal, yaw_err = self._grasp_goal(pos, yaw, obs)
        s_t = frames[-1].copy()
        chunk, u1 = self._plan_chunk(frames, hist_a, s_t, v_goal, yaw_err, rpy=att)
        k = getattr(self, "grasp_pulse_steps", None)
        if k is not None and att is not None:
            hold = self._level_chunk(att, frames[-1, 3:6])
            chunk = chunk.copy(); chunk[k:] = hold[k:] if len(hold) == CHUNK else 0.0
        grip_w = self._gripper_world(obs, pos, yaw)
        d_grip = float(np.linalg.norm(grip_w - box.ravel())) if grip_w is not None else 1.0
        v_now = self._get(obs, "state.dvl_v")
        settled = float(np.linalg.norm(np.zeros(3) if v_now is None else np.ravel(v_now)[:3])) < 0.03
        arm_busy = getattr(self, "arm_move_ticks", 0) > 0
        if arm_busy:
            self.arm_move_ticks -= 1
            if att is not None:
                chunk = self._level_chunk(att, frames[-1, 3:6]).astype(np.float32)
        mode = os.environ.get("WAM_GRASP_MODE", "pulse")
        if mode == "pulse" and self.grasp_stage == "grasp" and d_grip < 0.10 \
                and not getattr(self, "arm_frozen", False) and float(self.joint_cmd[0]) < GRIPPER_CLOSE / 2:
            box_xyz = box.ravel().astype(np.float64)
            cr, sr = np.cos(rpy[0]), np.sin(rpy[0]); cp, sp = np.cos(rpy[1]), np.sin(rpy[1]); cy, sy = np.cos(yaw), np.sin(yaw)
            R_wb = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            obj_body = (R_wb.T @ (box_xyz - pos)).astype(np.float32)
            hs, ha = self._grasp_frames(obs, frames, hist_a, obj_body)
            z_above = 0.012 if self.grasp_stage == "grasp" else 0.06
            seq16, info = self.grasp_planner.plan_pulse(hs, ha, hs[-1], obj_body, yaw, rpy=att, omega=frames[-1, 3:6], z_above=z_above)
            chunk = self._finish_chunk(np.asarray(seq16, np.float32), yaw_err, att, frames)
            self.grasp_pulse_steps = None
            jaw = float(self.joint_cmd[0]); self.joint_cmd = ARMED_JOINTS.copy(); self.joint_cmd[0] = jaw
            self._n_gp = getattr(self, "_n_gp", 0) + 1
            ch = info["choice"]; p1 = info["pred1"]
            print(f"[wam-server] grasp-pulse#{self._n_gp} stage={self.grasp_stage} d={info['dist']:.3f} err=({info['ex']:+.3f},{info['ey']:+.3f},{info['ez']:+.3f}) "
                  f"yaw_err={yaw_err:+.3f} v=({frames[-1,0]:+.3f},{frames[-1,1]:+.3f},{frames[-1,2]:+.3f}) choice=({'hold' if ch[0]=='hold' else tuple(round(float(x),2) for x in ch[0])},{ch[1]},{ch[2]}) "
                  f"pred0.5s=({p1[0]:+.3f},{p1[1]:+.3f},{p1[2]:+.3f}) pred2s=({info['pred_ex']:+.3f},{info['pred_ey']:+.3f},{info['pred_ez']:+.3f}) "
                  f"v_end={info['pred_v_end']:.3f} J={info['J']:.1f} hold={info['hold_J']:.1f}", flush=True)
            return chunk, chunk[0].copy(), v_goal, yaw_err
        if mode == "arm" and self.grasp_stage == "grasp" and d_grip < 0.08 and settled and not arm_busy \
                and not getattr(self, "arm_frozen", False) and float(self.joint_cmd[0]) < GRIPPER_CLOSE / 2:
            box_xyz = box.ravel().astype(np.float64)
            cr, sr = np.cos(rpy[0]), np.sin(rpy[0]); cp, sp = np.cos(rpy[1]), np.sin(rpy[1]); cy, sy = np.cos(yaw), np.sin(yaw)
            R_wb = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            obj_body = (R_wb.T @ (box_xyz - pos)).astype(np.float32)
            hs, ha = self._grasp_frames(obs, frames, hist_a, obj_body)
            hold5 = self._level_chunk(att, frames[-1, 3:6])[:5] if att is not None else np.zeros((5, 8), np.float32)
            q_target, info = self.grasp_planner.plan_arm(hs, ha, hs[-1], obj_body, yaw, hold5)
            q_now = np.ravel(np.asarray(self._get(obs, "state.joint_pos"), np.float32))[:5]
            if info["J"] < info.get("hold_J", float("inf")) - 0.5 and np.abs(q_target[1:] - q_now[1:]).max() > 0.01:
                jaw = float(self.joint_cmd[0])
                self.joint_cmd = np.asarray(q_target, np.float32).copy(); self.joint_cmd[0] = jaw
                self.arm_move_ticks = 2
                if att is not None:
                    chunk = self._level_chunk(att, frames[-1, 3:6]).astype(np.float32)
            self._n_gp = getattr(self, "_n_gp", 0) + 1
            if self._n_gp % 3 == 1:
                print(f"[wam-server] grasp-arm#{self._n_gp} stage={self.grasp_stage} d={info['dist']:.3f} ex={info['ex']:+.3f} "
                      f"ey={info['ey']:+.3f} ez={info['ez']:+.3f} imagined ey hold->plan {info.get('hold_J', float('nan')):.1f}->{info['J']:.1f} "
                      f"pred=({info['pred_ex']:+.3f},{info['pred_ey']:+.3f},{info['pred_ez']:+.3f}) q={np.round(q_target[1:], 3).tolist()}", flush=True)
        elif getattr(self, "arm_frozen", False) or float(self.joint_cmd[0]) >= GRIPPER_CLOSE / 2:
            self.arm_frozen = True
        elif self.grasp_stage in ("search", "approach"):
            jaw = float(self.joint_cmd[0])
            self.joint_cmd = ARMED_JOINTS.copy(); self.joint_cmd[0] = jaw
        return chunk, u1, v_goal, yaw_err

    def _grasp_plan_act(self, frames, hist_a, obs, pos, rpy, att):
        box = self._get(obs, "state.box_pos")
        if box is None:
            return None
        box_xyz = box.ravel().astype(np.float64)
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0]); cp, sp = np.cos(rpy[1]), np.sin(rpy[1]); cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_wb = Rz @ Ry @ Rx
        obj_body = (R_wb.T @ (box_xyz - pos)).astype(np.float32)
        obj_priv = obj_body.copy()
        hs, ha = self._grasp_frames(obs, frames, hist_a, obj_body)
        self.gp_acts = getattr(self, "gp_acts", 0) + 1
        ee_now = self.grasp_planner.ee_body(np.asarray(hs[-1][19:24], np.float32))
        d_now = float(np.linalg.norm(ee_now - obj_body))
        if (self.gp_acts > self.GRASP_RETRY_ACTS and self.grasp_planner.aligned_ticks == 0) or d_now > 0.30:
            print(f"[wam-server] grasp-planner: retry (acts={self.gp_acts}, d={d_now:.3f}) -> approach", flush=True)
            self.grasp_stage, self.grasp_hold_ticks, self.gp_acts = "approach", 0, 0
            self.grasp_planner.reset()
            return None
        pwm_seq, q_target, info = self.grasp_planner.plan(hs, ha, hs[-1], obj_body, yaw=float(rpy[2]), rpy=att,
                                                          omega=frames[-1, 3:6])
        jaw = float(self.joint_cmd[0])
        self.joint_cmd = np.asarray(q_target, np.float32).copy(); self.joint_cmd[0] = jaw
        v_now = self._get(obs, "state.dvl_v")
        settled = float(np.linalg.norm(np.zeros(3) if v_now is None else np.ravel(v_now)[:3])) < 0.04
        if info["aligned"] and info["dist"] < 0.035 and settled and self.grasp_planner.aligned_ticks >= self.GRASP_CLOSE_TICKS:
            self.joint_cmd[0] = GRIPPER_CLOSE
            self.arm_frozen = True
            print(f"[wam-server] grasp-planner: aligned (d={info['dist']:.3f} ex={info['ex']:+.3f} ey={info['ey']:+.3f}) -> close",
                  flush=True)
            self.lift_anchor = (self._get(obs, "state.box_pos").ravel().astype(np.float64).copy(), float(rpy[2]))
            self.grasp_stage, self.grasp_hold_ticks = "lift", 0
        self._n_gp = getattr(self, "_n_gp", 0) + 1
        if os.environ.get("WAM_GRASP_DEBUG"):
            dv = frames[-1, 0:3]
            print(f"[gp-debug] act={self._n_gp} obj_body=({obj_body[0]:+.3f},{obj_body[1]:+.3f},{obj_body[2]:+.3f}) "
                  f"ee_body=({info['ee_body'][0]:+.3f},{info['ee_body'][1]:+.3f},{info['ee_body'][2]:+.3f}) "
                  f"err_body=({info['ee_body'][0]-obj_body[0]:+.3f},{info['ee_body'][1]-obj_body[1]:+.3f},{info['ee_body'][2]-obj_body[2]:+.3f}) "
                  f"v_goal_dvl=({info['v_goal'][0]:+.3f},{info['v_goal'][1]:+.3f},{info['v_goal'][2]:+.3f}) "
                  f"dvl=({dv[0]:+.3f},{dv[1]:+.3f},{dv[2]:+.3f}) yaw={rpy[2]:+.2f} pwm0={np.round(pwm_seq[0], 2).tolist()}", flush=True)
        if self._n_gp % 3 == 1:
            print(f"[wam-server] grasp-planner#{self._n_gp} d={info['dist']:.3f} ex={info['ex']:+.3f} ey={info['ey']:+.3f} "
                  f"ez={info['ez']:+.3f} z_above={info['z_above']:.3f} pred=({info['pred_ex']:+.3f},{info['pred_ey']:+.3f},"
                  f"{info['pred_ez']:+.3f}) J={info['J']:.1f} q={np.round(q_target[1:], 3).tolist()}", flush=True)
        box_rpy = self._get(obs, "state.box_rpy")
        yaw = float(rpy[2])
        box_yaw = float(box_rpy.ravel()[2]) if box_rpy is not None else yaw
        yaw_ref = min([box_yaw, _yaw_wrap(box_yaw + np.pi)], key=lambda a: abs(_yaw_wrap(a - yaw)))
        yaw_err = float(np.clip(_yaw_wrap(yaw_ref - yaw), -1.0, 1.0))
        chunk = self._finish_chunk(np.asarray(pwm_seq, np.float32), yaw_err, att, frames)
        return chunk, chunk[0].copy(), np.asarray(info["v_goal"], np.float32), yaw_err

    YAW_KP = 0.5
    YAW_CLIP = 0.8
    YAW_RAMP_STEPS = 10
    YAW_MIX = np.array([1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], np.float32)

    def _attitude(self, obs):
        r = self._get(obs, "state.imu_rpy")
        if r is None:
            r = self._get(obs, "state.odom_rpy")
        return None if r is None else np.ravel(r).astype(np.float32)[:3]

    def _plan_chunk(self, frames, hist_a, s_t, v_goal, yaw_err, rpy=None):
        t0 = time.perf_counter()
        if self.direct is not None and self.action_source == "direct":
            u1, info = self.direct.plan(frames, hist_a, s_t, v_goal)
        elif self.direct is not None and self.action_source == "direct_cem":
            seed = self.direct.plan(frames, hist_a, s_t, v_goal)[1]["seq"]
            self.mpc._prev_seq = np.vstack([seed[:1], seed[:-1]])
            n0, it0 = self.mpc.cfg.n_samples, self.mpc.cfg.cem_iters
            self.mpc.cfg.n_samples, self.mpc.cfg.cem_iters = 32, 1
            try:
                u1, info = self.mpc.plan(frames, hist_a, s_t, v_goal, task_index=0,
                                         sid_u=None, rpy=rpy, yaw_err=0.0)
            finally:
                self.mpc.cfg.n_samples, self.mpc.cfg.cem_iters = n0, it0
        else:
            u1, info = self.mpc.plan(frames, hist_a, s_t, v_goal, task_index=0,
                                     sid_u=None, rpy=rpy, yaw_err=0.0)
        self._plan_ms.append(1e3 * (time.perf_counter() - t0))
        if len(self._plan_ms) % 50 == 0:
            arr = np.asarray(self._plan_ms[-50:])
            print(f"[wam-server] plan latency ({self.action_source}, n={self.mpc.cfg.n_samples}, "
                  f"iters={self.mpc.cfg.cem_iters}): mean {arr.mean():.1f} ms  p50 {np.median(arr):.1f}  "
                  f"p95 {np.percentile(arr, 95):.1f}  over last 50 acts", flush=True)
        seq = np.asarray(info.get("seq"), np.float32)
        if seq is None or seq.ndim != 2:
            seq = np.tile(u1[None], (CHUNK, 1))
        return self._finish_chunk(seq, yaw_err, rpy, frames), u1

    def _finish_chunk(self, seq, yaw_err, rpy, frames):
        reps = int(np.ceil(CHUNK / len(seq)))
        chunk = np.tile(seq, (reps, 1))[:CHUNK].copy()
        yaw_comp = (chunk[:, 0] - chunk[:, 1] - chunk[:, 2] + chunk[:, 3]) / 4.0
        chunk -= yaw_comp[:, None] * self.YAW_MIX[None, :]
        yy0 = self.YAW_KP * float(np.clip(yaw_err, -self.YAW_CLIP, self.YAW_CLIP))
        ramp = np.clip(1.0 - np.arange(CHUNK) / self.YAW_RAMP_STEPS, 0.0, 1.0)
        chunk += (yy0 * ramp)[:, None] * self.YAW_MIX[None, :]
        rr = (chunk[:, 4] - chunk[:, 5] + chunk[:, 6] - chunk[:, 7]) / 4.0
        pp = (-chunk[:, 4] - chunk[:, 5] + chunk[:, 6] + chunk[:, 7]) / 4.0
        chunk[:, 4] -= rr - pp
        chunk[:, 5] -= -rr - pp
        chunk[:, 6] -= rr + pp
        chunk[:, 7] -= -rr + pp
        if rpy is not None:
            chunk += self._level_chunk(rpy, frames[-1, 3:6])
        return np.clip(chunk, -1.0, 1.0)

    LEVEL_CLIP = 0.25
    LEVEL_RAMP_STEPS = 8

    def _level_chunk(self, rpy, omega):
        lvl = attitude_pwm(rpy, omega, clip=self.LEVEL_CLIP)
        ramp = np.clip(1.0 - np.arange(CHUNK) / self.LEVEL_RAMP_STEPS, 0.0, 1.0)
        return ramp[:, None] * lvl[None, :]

    def observe_sighted(self, obs):
        task = obs.get("annotation.human.action.task_description") or [""]
        task_str = task[0] if isinstance(task, (list, tuple)) else str(task)
        tc = obs.get("meta.task_code")
        with self.lock:
            tc = (tc[0] if isinstance(tc, (list, tuple, np.ndarray)) else tc) if tc is not None else None
            self.task_code = str(tc) if tc else None
            self._maybe_new_episode(task_str)
            frames, hist_a, _ = self._frames_from_obs(obs)
            odom_pos = self._get(obs, "state.odom_pos")
            odom_rpy = self._get(obs, "state.odom_rpy")
            if frames is None:
                return
            if odom_pos is not None and odom_rpy is not None:
                self.last_pose = (odom_pos.ravel().astype(np.float64),
                                  odom_rpy.ravel().astype(np.float64))
                self._goal_from_waypoints(self.last_pose[0], float(self.last_pose[1][2]))
                self._update_sighted_calib(frames, self.last_pose[0], float(self.last_pose[1][2]))
            v_sight, _, _ = self._v_est(frames, hist_a)
            if v_sight is not None:
                self.est_sighted.append(v_sight)
                self.est_sighted = self.est_sighted[-5:]
            self.v_anchor = None
            self.est_hist = []
            self.alpha = 0.0
            self.cusum.reset()
            self.brake_done = False
            hpw = self._get(obs, "state.hist_pwm")
            if hpw is not None and len(hpw):
                self.u_trans_hist = list(hpw[-10:])
                if self.hold_anchor == "median":
                    self.u_trans_hold = np.clip(np.median(hpw[-10:], axis=0), -1, 1)
                else:
                    self.u_trans_hold = np.clip(np.mean(hpw[-10:], axis=0), -1, 1)
                self.last_u = np.asarray(hpw[-1], np.float32).copy()
            self.blind_prev = False

    def blind_act(self, obs):
        task = obs.get("annotation.human.action.task_description") or [""]
        task_str = task[0] if isinstance(task, (list, tuple)) else str(task)
        with self.lock:
            self._maybe_new_episode(task_str)
            frames, hist_a, dvl_zero = self._frames_from_obs(obs)
            if frames is None:
                return self._package(np.zeros((CHUNK, 8), np.float32))
            chunk = self._blind_chunk(frames, hist_a, imu_yaw=self._imu_yaw(obs), obs=obs,
                                      dvl_zero=dvl_zero)
            chunk = self._attitude_guard(chunk, frames, obs)
            self.last_u = chunk[-1].copy()
            return self._package(chunk)

    def act(self, obs):
        task = obs.get("annotation.human.action.task_description") or [""]
        task_str = task[0] if isinstance(task, (list, tuple)) else str(task)
        tc = obs.get("meta.task_code")
        with self.lock:
            tc = (tc[0] if isinstance(tc, (list, tuple, np.ndarray)) else tc) if tc is not None else None
            self.task_code = str(tc) if tc else None
            self._maybe_new_episode(task_str)
            frames, hist_a, dvl_zero = self._frames_from_obs(obs)
            odom_pos = self._get(obs, "state.odom_pos")
            odom_rpy = self._get(obs, "state.odom_rpy")
            dvl_valid = self._get(obs, "meta.dvl_valid")
            if dvl_valid is not None:
                blind = bool(float(np.ravel(dvl_valid)[0]) < 0.5)
            else:
                blind = bool(dvl_zero is not None and len(dvl_zero) >= 2
                             and dvl_zero[-1] and dvl_zero[-2])

            if frames is None:
                return self._package(np.zeros((CHUNK, 8), np.float32))

            pos = odom_pos.ravel().astype(np.float64) if odom_pos is not None else np.zeros(3)
            rpy = odom_rpy.ravel().astype(np.float64) if odom_rpy is not None else np.zeros(3)
            yaw = float(rpy[2])
            self._n_act_total = getattr(self, "_n_act_total", 0) + 1
            if getattr(self, "box_head", None) is not None and self.mode == "transfer" and not blind:
                self._update_dest_estimate(obs, pos, yaw)
            if getattr(self, "wrist_head", None) is not None and self.mode in ("grasp", "transfer"):
                rel = obs.get("state.box_rel")
                obj_priv = None if rel is None else np.ravel(np.asarray(rel, np.float32))[:3]
                self._wrist_object(obs, obj_priv)
                if self.obj_source == "wrist":
                    obs = {k: v for k, v in obs.items() if k not in ("state.box_pos", "state.box_rpy", "state.box_rel", "state.box_rel_yaw")}

            if not blind:
                obs = self._box_in_frame(obs, pos, yaw)
                self.last_pose = (pos.copy(), rpy.copy())
                self.dr_pos, self.dr_yaw = None, None
                self.imu_vw = None
                self.last_dvl_v = frames[-1, 0:3].astype(np.float64).copy()
                if odom_pos is not None:
                    self._update_sighted_calib(frames, pos, yaw)
                v_sight, _, _ = self._v_est(frames, hist_a)
                if v_sight is not None:
                    self.est_sighted.append(v_sight)
                    self.est_sighted = self.est_sighted[-5:]
                self.v_anchor = None
                self.est_hist = []
                self.alpha = 0.0
                self.cusum.reset()
                self.brake_done = False
                s_t = frames[-1].copy()
                att = self._attitude(obs)
                planned = None
                if getattr(self, "direct_manip", None) is not None and self.mode in ("grasp", "transfer") and self.percept is None:
                    box = self._get(obs, "state.box_pos")
                    if box is not None:
                        cr, sr = np.cos(rpy[0]), np.sin(rpy[0]); cp, sp = np.cos(rpy[1]), np.sin(rpy[1]); cy, sy = np.cos(yaw), np.sin(yaw)
                        R_wb = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                        obj_body = (R_wb.T @ (box.ravel().astype(np.float64) - pos)).astype(np.float32)
                        hs, ha = self._grasp_frames(obs, frames, hist_a, obj_body)
                        c13 = self.direct_manip.act(hs, ha)
                        self.joint_cmd = np.asarray(c13[0, 8:13], np.float32).copy()
                        chunk = np.asarray(c13[:, :8], np.float32)
                        planned = (chunk, chunk[0].copy(), np.zeros(3, np.float32), 0.0)
                        self._n_dm = getattr(self, "_n_dm", 0) + 1
                        if self._n_dm % 20 == 1:
                            print(f"[wam-server] direct-manip#{self._n_dm} |pwm|={np.abs(chunk).mean():.3f} jaw={self.joint_cmd[0]:.3f} "
                                  f"obj_body=({obj_body[0]:+.2f},{obj_body[1]:+.2f},{obj_body[2]:+.2f})", flush=True)
                if planned is None and self.grasp_planner is not None and self.mode in ("grasp", "transfer") and self.percept is None:
                    if os.environ.get("WAM_GRASP_MODE", "hybrid") == "full":
                        if self.grasp_stage == "grasp":
                            planned = self._grasp_plan_act(frames, hist_a, obs, pos, rpy, att)
                    else:
                        planned = self._grasp_arm_act(frames, hist_a, obs, pos, rpy, att)
                if planned is not None:
                    chunk, u1, v_goal, yaw_err = planned
                else:
                    self._v_now = frames[-1, 0:3]
                    v_goal, yaw_err = self._task_goal(pos, yaw, obs)
                    chunk, u1 = self._plan_chunk(frames, hist_a, s_t, v_goal, yaw_err, rpy=att)
                k = getattr(self, "grasp_pulse_steps", None)
                if planned is None and k is not None and att is not None:
                    hold = self._level_chunk(att, frames[-1, 3:6])
                    chunk = chunk.copy()
                    chunk[k:] = hold[k:] if len(hold) == CHUNK else 0.0
                self._record_dump(frames, hist_a)
                self._n_act = getattr(self, "_n_act", 0) + 1
                if self._n_act % 5 == 1:
                    print(f"[wam-server] act#{self._n_act} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                          f"yaw={yaw:.2f} wp={self.wp_idx}/{0 if self.waypoints is None else len(self.waypoints)} "
                          f"v_goal=({v_goal[0]:.2f},{v_goal[1]:.2f},{v_goal[2]:.2f}) yaw_err={yaw_err:.2f} "
                          f"dvl=({frames[-1,0]:.2f},{frames[-1,1]:.2f},{frames[-1,2]:.2f}) "
                          f"|u|={np.abs(chunk).mean():.3f}", flush=True)
                self.u_trans_hist.append(u1.copy())
                self.u_trans_hist = self.u_trans_hist[-10:]
                if self.hold_anchor == "mixer":
                    self.u_trans_hold = vel_track_pwm(
                        frames[-1, 0:3], v_goal, omega=frames[-1, 3:6], rpy=att, yaw_err=yaw_err)
                elif self.hold_anchor == "median":
                    self.u_trans_hold = np.clip(
                        np.median(np.stack(self.u_trans_hist, 0), axis=0), -1, 1)
                else:
                    self.u_trans_hold = np.clip(
                        np.mean(np.stack(self.u_trans_hist, 0), axis=0), -1, 1)
                self.blind_prev = False
            else:
                chunk = self._blind_chunk(frames, hist_a, imu_yaw=self._imu_yaw(obs), obs=obs,
                                          dvl_zero=dvl_zero)
            chunk = self._attitude_guard(chunk, frames, obs)
            self.last_u = chunk[-1].copy()
            return self._package(chunk)

    ATT_ENTER, ATT_EXIT, ATT_MAX = 0.35, 0.15, 3

    def _attitude_guard(self, chunk, frames, obs):
        att = self._attitude(obs)
        if att is None:
            return chunk
        tilt = float(max(abs(att[0]), abs(att[1])))
        if self.att_recover > 0:
            if tilt < self.ATT_EXIT or self.att_recover >= self.ATT_MAX:
                print(f"[wam-server] attitude guard: released (tilt={tilt:.2f} after "
                      f"{self.att_recover} chunks)", flush=True)
                self.att_recover = 0
                self.att_cool = 2
                return chunk
            self.att_recover += 1
        elif tilt > self.ATT_ENTER and self.att_cool == 0:
            self.att_recover = 1
            print(f"[wam-server] attitude guard: tilt={tilt:.2f} rad -> level only", flush=True)
        else:
            self.att_cool = max(0, self.att_cool - 1)
            return chunk
        return np.clip(self._level_chunk(att, frames[-1, 3:6]), -1.0, 1.0).astype(np.float32)

    COAST_TAU = 2.1
    BRAKE_SPEED = 0.30

    def _takeover_brake(self, frames, dvl_zero, imu_yaw):
        if getattr(self, "brake_done", False) or dvl_zero is None:
            return None
        valid = np.nonzero(~np.asarray(dvl_zero, bool))[0]
        if len(valid) == 0:
            return None
        k = int(valid[-1])
        v0 = frames[k, 0:3].astype(np.float64)
        self.brake_done = True
        if np.linalg.norm(v0[:2]) <= self.BRAKE_SPEED:
            return None
        n_since = len(frames) - 1 - k
        yaw = float(imu_yaw) if imu_yaw is not None else (
            self.dr_yaw if self.dr_yaw is not None else 0.0)
        cy, sy = np.cos(yaw), np.sin(yaw)

        def to_world(v):
            vx, vy = float(v[0]), AXIS_SIGN[1] * float(v[1])
            return np.array([cy * vx - sy * vy, sy * vx + cy * vy])

        prev = np.zeros(2)
        for i in range(len(frames)):
            prev += to_world(frames[i, 0:3] if not dvl_zero[i] else v0) * DT
        if self.dr_pos is not None:
            self.dr_pos[0] += prev[0]
            self.dr_pos[1] += prev[1]
        tau = self.COAST_TAU
        coast = tau * (1.0 - np.exp(-DT * CHUNK / tau))
        self.coast_disp = to_world(v0) * coast
        v_end = v0 * np.exp(-DT * CHUNK / tau)
        self.est_hist = [v_end.astype(np.float32)]
        print(f"[wam-server] takeover at {np.linalg.norm(v0[:2]):.2f} m/s -> coast chunk "
              f"(drop {n_since} ticks ago, prev-chunk travel {np.linalg.norm(prev):.2f} m, "
              f"coast travel {np.linalg.norm(self.coast_disp):.2f} m, "
              f"exit speed {np.linalg.norm(v_end[:2]):.2f} m/s)", flush=True)
        return np.zeros((CHUNK, 8), np.float32)

    def _blind_chunk(self, frames, hist_a, imu_yaw=None, obs=None, dvl_zero=None):
        if self.blind_est == "imu":
            self.imu_v_body = self._imu_integrate(frames, dvl_zero, imu_yaw, obs)
        if not self.blind_prev:
            if self.last_pose is not None:
                self.dr_pos = self.last_pose[0].copy()
                self.dr_yaw = float(self.last_pose[1][2])
            self.blind_prev = True
            brake = None if self._uses_live_target() else self._takeover_brake(frames, dvl_zero, imu_yaw)
            if brake is not None:
                att0 = self._attitude(obs) if obs is not None else None
                return self._level_chunk(att0, frames[-1, 3:6]).astype(np.float32)
        f = frames.copy()
        v_est, sig_alea, sig_epi = self._v_est(f, hist_a)
        if v_est is None:
            v_est = np.zeros(3, np.float32)
        f[-1, 0:3] = v_est
        if self.v_anchor is None and self.est_sighted:
            self.v_anchor = np.mean(np.stack(self.est_sighted, 0), axis=0)
            self.cusum.start_blind(self.v_anchor / self.vel_std)
        self.est_hist.append(v_est.copy())
        self.est_hist = self.est_hist[-5:]
        v_sm = np.mean(np.stack(self.est_hist, 0), axis=0)
        goal_active = self._uses_live_target() or self._percept_nav_active() or (
            self.waypoints is not None and self.wp_idx < len(self.waypoints))
        if sig_alea is not None and self.v_anchor is not None:
            sig_eff = np.sqrt(sig_alea ** 2 / max(1, len(self.est_hist)) + sig_epi ** 2)
            self.alpha = self.cusum.step(v_sm / self.vel_std, sig_eff, goal_changed=goal_active)
        elif goal_active:
            self.alpha = 1.0
        else:
            self.alpha = max(0.0, self.alpha * 0.90)
        if self.dr_pos is not None:
            yaw = self.dr_yaw if self.dr_yaw is not None else 0.0
            if imu_yaw is not None:
                yaw = float(imu_yaw)
                self.dr_yaw = yaw
            else:
                gyro_z = float(np.mean(frames[:, 5])) - self.gyro_bias
                self.dr_yaw = _yaw_wrap(yaw + gyro_z * DT * CHUNK)
            cy, sy = np.cos(yaw), np.sin(yaw)
            vx, vy, vz = float(v_est[0]), AXIS_SIGN[1] * float(v_est[1]), AXIS_SIGN[2] * float(v_est[2])
            step = np.array([cy * vx - sy * vy, sy * vx + cy * vy, vz]) * DT * CHUNK
            coast_disp = getattr(self, "coast_disp", None)
            if coast_disp is not None:
                step[0], step[1] = coast_disp[0], coast_disp[1]
                self.coast_disp = None
            self.dr_pos = self.dr_pos + step
            if self.pz_fit is not None:
                k, b = self.pz_fit
                self.dr_pos[2] = k * float(f[-1, 9]) / 1e4 + b
        att = self._attitude(obs) if obs is not None else None
        rel_ok = obs is not None and (obs.get("state.box_rel") is not None or (
            getattr(self, "wrist_head", None) is not None and getattr(self, "obj_source", "") == "wrist"
            and getattr(self, "obj_est", None) is not None))
        if (self.grasp_planner is not None and self.mode in ("grasp", "transfer") and self.percept is None
                and rel_ok and self.dr_pos is not None):
            yaw_dr = self.dr_yaw if self.dr_yaw is not None else 0.0
            rpy_dr = np.array([att[0] if att is not None else 0.0, att[1] if att is not None else 0.0, yaw_dr])
            obs_dr = self._box_in_frame(obs, self.dr_pos, yaw_dr)
            planned = self._grasp_arm_act(f, hist_a, obs_dr, self.dr_pos, rpy_dr, att)
            if planned is not None:
                self._n_blind = getattr(self, "_n_blind", 0) + 1
                if self._n_blind % 10 == 1:
                    print(f"[wam-server] blind-grasp#{self._n_blind} dr_pos=({self.dr_pos[0]:.2f},{self.dr_pos[1]:.2f},"
                          f"{self.dr_pos[2]:.2f}) dr_yaw={yaw_dr:.2f} stage={self.grasp_stage} "
                          f"v_est=({v_est[0]:.3f},{v_est[1]:.3f},{v_est[2]:.3f})", flush=True)
                return np.clip(planned[0], -1, 1).astype(np.float32)
        if self.hold_anchor == "mixer":
            v_g = self.v_anchor if self.v_anchor is not None else v_est
            u_hold = vel_track_pwm(v_est, v_g, omega=frames[-1, 3:6], rpy=None)
        else:
            u_hold = self.u_trans_hold if self.u_trans_hold is not None else np.zeros(8, np.float32)
        hold_chunk = np.tile(u_hold[None], (CHUNK, 1)) + (self._level_chunk(att, None) if att is not None else 0.0)
        a = float(self.alpha)
        if a <= 0.01 or self.dr_pos is None:
            return np.clip(hold_chunk, -1, 1)
        pos = self.dr_pos
        yaw = self.dr_yaw if self.dr_yaw is not None else 0.0
        self._v_now = v_est
        v_goal, yaw_err = self._task_goal(pos, yaw, obs if obs is not None else {})
        s_t = f[-1].copy()
        replan_chunk, _ = self._plan_chunk(f, hist_a, s_t, v_goal, yaw_err, rpy=att)
        self._n_blind = getattr(self, "_n_blind", 0) + 1
        if self._n_blind % 5 == 1:
            print(f"[wam-server] blind#{self._n_blind} dr_pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"dr_yaw={yaw:.2f} src={'imu' if imu_yaw is not None else 'gyro'} alpha={a:.2f} "
                  f"wp={self.wp_idx}/{len(self.waypoints) if self.waypoints is not None else 0} "
                  f"v_goal=({v_goal[0]:.2f},{v_goal[1]:.2f},{v_goal[2]:.2f}) yaw_err={yaw_err:.2f} "
                  f"v_est=({v_est[0]:.2f},{v_est[1]:.2f},{v_est[2]:.2f})", flush=True)
        return np.clip((1.0 - a) * hold_chunk + a * replan_chunk, -1, 1)

    def _package(self, chunk):
        joints = getattr(self, "joint_cmd", None)
        if joints is None:
            jp = np.zeros((CHUNK, N_JOINTS), np.float64)
        else:
            jp = np.tile(np.asarray(joints, np.float64).reshape(1, -1), (CHUNK, 1))
        return [{"action.pwm": chunk.astype(np.float64),
                 "action.joint_pos": jp}, {}]


def make_handler(policy):
    import json_numpy

    json_numpy.patch()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({"status": "healthy", "model": "WAM"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/act":
                self.send_response(404)
                self.end_headers()
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n).decode())
                if "encoded" in payload:
                    payload = json.loads(payload["encoded"])
                obs = payload["observation"]
                t0 = time.time()
                action = policy.act(obs)
                dt_ms = (time.time() - t0) * 1e3
                tgt = os.environ.get("WAM_TARGET_STEP_MS", "")
                if tgt:
                    tm = {k: float(v) for k, v in (kv.split(":") for kv in tgt.split(",") if ":" in kv)}
                    want = tm.get("grasp" if getattr(policy, "mode", "") in ("grasp", "transfer") else "nav",
                                  max(tm.values()) if tm else 0.0)
                    if dt_ms < want:
                        time.sleep((want - dt_ms) / 1e3)
                        dt_ms = (time.time() - t0) * 1e3
                dump = os.environ.get("WAM_DUMP_REQ", "")
                if dump:
                    Path(dump).mkdir(parents=True, exist_ok=True)
                    k = len(list(Path(dump).glob("req_*.json")))
                    (Path(dump) / f"req_{k:05d}.json").write_bytes(json.dumps(payload).encode())
                    with open(Path(dump) / "server_act_ms.csv", "a") as f:
                        f.write(f"{k},{dt_ms:.2f},{getattr(policy, 'mode', '')}\n")
                if dt_ms > 400:
                    print(f"[wam-server] slow act: {dt_ms:.0f} ms", flush=True)
                body = json.dumps(action).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                import traceback

                traceback.print_exc()
                body = json.dumps({"detail": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_scenes.pt")
    ap.add_argument("--vel-ens", default="/hy-tmp/models/uwam/vel_ens_scenes.pt",
                    help="dead-reckoning ensemble; the _deploy variant is trained on task-scene "
                         "recordings with canonicalized pressure/altitude")
    ap.add_argument("--gate-calib", default="/hy-tmp/models/uwam/gate_calib.json")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore",
                    help="comma-separated OU-schema dirs for the planner's action library")
    ap.add_argument("--eval-root", default="/hy-tmp/u0env/dataset/eval")
    ap.add_argument("--hold-anchor", choices=["mean", "median", "mixer"], default="mixer")
    ap.add_argument("--vmax-scale", type=float, default=1.0,
                    help="scale on the per-family cruise-speed caps (1.8 -> nav 0.45 m/s)")
    ap.add_argument("--dump-dir", default="/hy-tmp/data/planner_task",
                    help="OU-schema dumps of MPC-in-the-loop commands (b3 mix). Empty to disable.")
    ap.add_argument("--goal-source", choices=["privileged", "percept"], default="privileged",
                    help="grasp/transfer goal: privileged /box/odometry standoffs, or the "
                         "vision perception head (no privileged reads; the no-asterisk arm)")
    ap.add_argument("--percept-nav-head", default="/hy-tmp/models/uwam/percept_nav_head.pt",
                    help="navigation head for the percept arm (goto/scan/inspect/follow goals from "
                         "images); '' keeps locomotion on the waypoint file")
    ap.add_argument("--percept-e2e", default="",
                    help="end-to-end perception checkpoint (percept/train_e2e.py); when set, locomotion "
                         "goals are the model's 3 s intent instead of the frozen-feature node head")
    ap.add_argument("--n-samples", type=int, default=0,
                    help="override ControlCfg.n_samples (CEM population; 0 = config default 128)")
    ap.add_argument("--cem-iters", type=int, default=0,
                    help="override ControlCfg.cem_iters (0 = config default 2)")
    ap.add_argument("--action-source", choices=["mpc", "direct", "direct_cem"], default="mpc",
                    help="mpc: sampling MPC through the world model (default); direct: the amortized "
                         "action head (uwam/direct.py) emits the 0.5 s PWM sequence in one forward pass; "
                         "direct_cem: the head's sequence seeds a small (n=32, 1 iter) CEM polish")
    ap.add_argument("--direct-ckpt", default="/hy-tmp/models/uwam/direct_head.pt",
                    help="amortized action head checkpoint (scripts/train_direct.py)")
    ap.add_argument("--grasp-planner", choices=["primitive", "wam"], default="primitive",
                    help="primitive: hull-only staged standoffs + threshold close (the pilot arm); wam: the "
                         "manipulation world model plans thrusters + joints for the final alignment")
    ap.add_argument("--grasp-ckpt", default="/hy-tmp/models/uwam/grasp_core.pt",
                    help="35-d arm+object world model (train_stage1.py --use-object)")
    ap.add_argument("--arm-kin", default="/hy-tmp/models/uwam/arm_kin.pt", help="learned forward kinematics")
    ap.add_argument("--wrist-pose-ckpt", default="",
                    help="wrist-camera relative-pose head (percept/train_wrist_pose.py); when set, the grasp stage takes "
                         "the object from the images instead of /box/odometry")
    args = ap.parse_args()

    policy = WamPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"[wam-server] listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
