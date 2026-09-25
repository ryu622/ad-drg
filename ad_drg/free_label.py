"""「守備者を外した」ラベル(フェーズ13)。

定義: 攻撃者が次のプレー(シュート・パス・前進)を実行するのに T_act 秒かかるとき、
その間に守備者がボール(またはシュートコース)に届かない状態を「外した(free)」とする。
距離や前後関係ではなく、守備者の介入に必要な時間で判定する。

守備者の到達可能範囲: 減衰付きダブルインテグレータ v' = u - v/τ, ||u|| <= a_max では、
初期状態 (p, v) から t 秒後に到達できる位置の集合は厳密に円になる。
    中心 c(t) = p + τ v (1 - e^{-t/τ})              (何もしなければ惰性で進む位置)
    半径 R(t) = a_max τ (t - τ (1 - e^{-t/τ}))
(制御の効き方がスカラー核なので、||u|| <= a_max の像が円になる。)

攻撃者側: T_act の間は次のプレーを実行しているので、ボールは攻撃者の速度で等速に進むと仮定する
    b(t) = ball + v_a t
(最悪ケースで逃げ回るゲームにすると、実行中の攻撃者に不利な守備者を仮定することになり、判定が甘くなりすぎる)。

    ボールに届く   : ある t ∈ [0, T_act] で ||b(t) - c(t)|| <= R(t) + r_reach
    コースに届く   : ||dist(c(T_act), 線分[b(T_act), ゴール中心])|| <= R(T_act) + r_reach
                     (ゴールまで lane_max_dist 以内のときだけ適用)
    外した(free)   : どちらにも届かない
    1対1の開始    : 守備者が初めてボールに届く状態になったフレーム(以後 horizon 秒で判定)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ad_drg.extraction import Trajectory


@dataclass(frozen=True)
class FreeParams:
    a_max_def: float
    tau: float
    t_act: float = 0.5
    r_reach: float = 1.0
    use_lane: bool = True
    lane_max_dist: float = 30.0
    n_t: int = 11


def reach_center_radius(p: np.ndarray, v: np.ndarray, t: np.ndarray, a_max: float, tau: float):
    """p, v: (N, 2), t: (K,) -> 中心 (N, K, 2), 半径 (K,)。"""
    decay = 1.0 - np.exp(-t / tau)
    center = p[:, None, :] + tau * v[:, None, :] * decay[None, :, None]
    radius = a_max * tau * (t - tau * decay)
    return center, radius


def _point_segment_dist(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    w = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(q - (a + w[:, None] * ab), axis=1)


def defender_access(traj: Trajectory, prm: FreeParams) -> dict:
    """各フレームで守備者がボール/シュートコースに T_act 以内に届くかを返す。"""
    t = np.linspace(0.0, prm.t_act, prm.n_t)
    center, radius = reach_center_radius(traj.p_d, traj.v_d, t, prm.a_max_def, prm.tau)
    ball_path = traj.ball[:, None, :] + traj.v_a[:, None, :] * t[None, :, None]
    ball_reach = np.any(np.linalg.norm(ball_path - center, axis=-1) <= radius[None, :] + prm.r_reach, axis=1)

    lane_reach = np.zeros(len(traj.ball), dtype=bool)
    if prm.use_lane:
        release = ball_path[:, -1, :]
        goal = np.broadcast_to(traj.goal, release.shape)
        near = np.linalg.norm(goal - release, axis=1) <= prm.lane_max_dist
        d_lane = _point_segment_dist(center[:, -1, :], release, goal)
        lane_reach = near & (d_lane <= radius[-1] + prm.r_reach)
    return dict(ball_reach=ball_reach, lane_reach=lane_reach, free=~(ball_reach | lane_reach))


def free_label(traj: Trajectory, prm: FreeParams, horizon: float = 3.0, min_free: float = 0.2) -> dict:
    """1対1の開始(守備者が初めてボールに介入可能になったフレーム k_eng)を求め、
    そこから horizon 秒以内に min_free 秒以上続けて「外した」状態になったかを返す。

    守備者が一度も介入可能にならないイベントは 1対1が成立していない(engaged=False)。
    """
    acc = defender_access(traj, prm)
    reach = np.nonzero(acc["ball_reach"])[0]
    if not len(reach):
        return dict(engaged=False, k_eng=None, freed=False, t_free=None)
    k_eng = int(reach[0])
    n_end = min(len(traj.ball), k_eng + int(round(horizon / traj.dt)) + 1)
    k_min = max(1, int(round(min_free / traj.dt)))
    free = acc["free"]
    # horizon 秒以内に始まり、セグメント内で k_min フレーム以上続く「外した」区間を探す
    t_free = None
    for k in range(k_eng, n_end):
        if k + k_min <= len(free) and free[k: k + k_min].all():
            t_free = (k - k_eng) * traj.dt
            break
    return dict(engaged=True, k_eng=k_eng, freed=t_free is not None, t_free=t_free)
