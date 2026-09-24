"""HJI最適フィードバック制御による軌道シミュレーション(AD モデルとの直接比較用)。

AD モデルの「改良推定法」(ad_drg/ad_model.py)と同様に、相手の実軌道を固定して
自分側のみをシミュレーションする独立方式を採用する。守備者側は捕捉可能集合の
勾配から導かれる最適(min側)加速度を、攻撃者側は最適(max側)加速度を用いる。

AD モデルと異なり、HJI 側にはパラメータの最適化は不要(捕捉可能集合が既に
最適フィードバック方策を内包している)。research_plan.md 5節③の比較に対応する。
"""

from __future__ import annotations

import numpy as np

from ad_drg.ad_model import error_metric
from ad_drg.extraction import Trajectory
from ad_drg.reachability import BRSResult


def simulate_defender_hji(traj: Trajectory, brs: BRSResult) -> np.ndarray:
    p = traj.p_d[0].copy()
    v = traj.v_d[0].copy()
    out = [p.copy()]
    duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
    for k in range(len(traj.p_d) - 1):
        p_a, v_a = traj.p_a[k], traj.v_a[k]
        state = np.concatenate([p_a - p, v_a - v])
        remaining = np.clip(duration - traj.t[k], 0.0, brs.times[0] - brs.times[-1])
        u = brs.optimal_defender_accel(state, remaining)
        v = v + u * traj.dt
        p = p + v * traj.dt
        out.append(p.copy())
    return np.array(out)


def simulate_attacker_hji(traj: Trajectory, brs: BRSResult) -> np.ndarray:
    p = traj.p_a[0].copy()
    v = traj.v_a[0].copy()
    out = [p.copy()]
    duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
    for k in range(len(traj.p_a) - 1):
        p_d, v_d = traj.p_d[k], traj.v_d[k]
        state = np.concatenate([p - p_d, v - v_d])
        remaining = np.clip(duration - traj.t[k], 0.0, brs.times[0] - brs.times[-1])
        u = brs.optimal_attacker_accel(state, remaining)
        v = v + u * traj.dt
        p = p + v * traj.dt
        out.append(p.copy())
    return np.array(out)


def fit_event_hji(traj: Trajectory, brs: BRSResult) -> dict:
    sim_d = simulate_defender_hji(traj, brs)
    sim_a = simulate_attacker_hji(traj, brs)
    error_d = error_metric(sim_d, traj.p_d)
    error_a = error_metric(sim_a, traj.p_a)
    return dict(error_a=error_a, error_d=error_d)
