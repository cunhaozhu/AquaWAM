
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Paths:
    root: Path = Path("/hy-tmp")
    usim: Path = Path("/hy-tmp/data/usim")
    u0env: Path = Path("/hy-tmp/u0env")
    ckpt: Path = Path("/hy-tmp/models/uwam")
    logs: Path = Path("/hy-tmp/logs/uwam")
    ou_data: Path = Path("/hy-tmp/data/ou_explore")


@dataclass
class Schema:

    state_dim: int = 29
    action_dim: int = 13
    fps: int = 10
    dt: float = 0.1
    image_hw: Tuple[int, int] = (96, 128)
    ego_key: str = "observation.images.ego"
    wrist_key: str = "observation.images.wrist"
    state_key: str = "observation.state"
    action_key: str = "action"

    joint_pos: Tuple[int, int] = (0, 5)
    pwm: Tuple[int, int] = (5, 13)
    joint_v: Tuple[int, int] = (13, 18)
    dvl_v: Tuple[int, int] = (18, 21)
    imu_av: Tuple[int, int] = (21, 24)
    imu_la: Tuple[int, int] = (24, 27)
    pressure: Tuple[int, int] = (27, 28)
    dvl_h: Tuple[int, int] = (28, 29)

    action_joint: Tuple[int, int] = (0, 5)
    action_pwm: Tuple[int, int] = (5, 13)


@dataclass
class ModelCfg:
    history_len: int = 16
    horizon_dyn: int = 5
    horizon_vis: int = 15
    latent_dim: int = 128
    disturbance_dim: int = 96
    hidden: int = 384
    proprio_dim: int = 16
    dyn_state_dim: int = 19
    pwm_dim: int = 8
    dropout: float = 0.1
    use_language: bool = False
    use_dt: bool = False
    use_arm: bool = False
    use_object: bool = False
    n_joints: int = 5
    n_tasks: int = 9
    visual_channels: int = 6


@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1
    max_train_windows: int = 0
    amp: bool = True
    lambda_pix: float = 1.0
    lambda_lpips: float = 0.1
    lambda_rank: float = 0.5
    lambda_state: float = 1.0
    lambda_dvl: float = 2.0
    lambda_hist: float = 0.5
    rank_margin: float = 0.05
    modality_dropout: float = 0.2
    lambda_aux: float = 0.15
    lambda_vel: float = 1.0
    lambda_eta: float = 1.0
    lambda_dcons: float = 0.1
    vel_fault_weight: float = 3.0


@dataclass
class ControlCfg:
    horizon: int = 5
    horizon_chunks: int = 4
    cem_iters: int = 2
    cem_elites: int = 16
    n_warm_seeds: int = 8
    n_samples: int = 128
    action_std: float = 0.08
    mixer_std: float = 0.05
    goal_w: float = 5.0
    control_w: float = 0.02
    safety_w: float = 0.4
    att_w: float = 0.5
    n_sysid_seeds: int = 16
    n_mixer_seeds: int = 40
    n_hold_seeds: int = 8
    mixer_kp: float = 2.0
    mixer_kff: float = 2.0
    mixer_kd_w: float = 0.25
    blind_gate_lo: float = 0.05
    blind_gate_hi: float = 0.12
    blind_alpha_decay: float = 0.90
    blind_innov_smooth: int = 5
    ou_theta: float = 0.15
    ou_mu: float = 0.0
    ou_sigma: float = 0.35
    ou_dt: float = 0.1


@dataclass
class Cfg:
    paths: Paths = field(default_factory=Paths)
    schema: Schema = field(default_factory=Schema)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    control: ControlCfg = field(default_factory=ControlCfg)


def enable_arm(cfg: Cfg) -> Cfg:
    m = cfg.model
    if m.use_arm:
        return cfg
    m.use_arm = True
    m.dyn_state_dim = 19 + 2 * m.n_joints
    m.pwm_dim = 8 + m.n_joints
    return cfg


def enable_object(cfg: Cfg) -> Cfg:
    enable_arm(cfg)
    m = cfg.model
    if getattr(m, "use_object", False):
        return cfg
    m.use_object = True
    m.dyn_state_dim = 19 + 2 * m.n_joints + 6
    return cfg


ARM_DYN_SLICES = {
    "dvl_v": (0, 3), "imu_av": (3, 6), "imu_la": (6, 9),
    "pressure": (9, 10), "dvl_h": (10, 11), "pwm": (11, 19),
    "joint_pos": (19, 24), "joint_v": (24, 29),
    "obj": (29, 35),
}


def ensure_dirs(cfg: Cfg) -> None:
    for p in (cfg.paths.ckpt, cfg.paths.logs, cfg.paths.ou_data):
        p.mkdir(parents=True, exist_ok=True)
