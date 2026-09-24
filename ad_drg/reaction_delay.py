"""守備者の反応遅れ δ を入れた reach-avoid 価値(フェーズ8)。

完全情報・遅れなしの差分ゲームでは、守備者は攻撃者の動きを見た瞬間に最適反応できるため
フェイント(騙し)が成立しない。本モジュールでは最も単純な形で遅れを入れる:

    最初の δ 秒間、守備者は攻撃者の新しい動きに反応できず、加速度 0(減衰のみで惰性走行)。
    その間、攻撃者は一定加速度 u_a (||u_a|| = a_max_a の全方向 + 静止)を自由に選べる。
    δ 秒後からは遅れなしの reach-avoid ゲーム(残り時間 T - δ)を最適にプレーする。

    V_δ(x) = min_{u_a} max( max_{t∈[0,δ]} g(x(t)),
                            min( min_{t∈[0,δ]} l(x(t)), V_0(x(δ), T - δ) ) )

l, g は ad_drg/reach_avoid.py と同じ(l<=0: 抜いた、g>0: 間合いに入った)。
V_0(·) - V_δ(·) >= 0 が「δ 秒の反応遅れを攻撃者が突いたときに得られる利得」になる。

これは「フェイントの瞬間に一度だけ生じる遅れ」のモデルであり、情報遅れが常に
続く差分ゲーム(非予見的戦略の厳密な扱い)の近似である点に注意。
"""

from __future__ import annotations

import numpy as np

from ad_drg.reach_avoid import ReachAvoidResult

N_DIRECTIONS = 24
N_WINDOW_SAMPLES = 6


def _propagate_relative(x: np.ndarray, u: np.ndarray, t: np.ndarray, tau: float) -> np.ndarray:
    """減衰付きダブルインテグレータの解析解。x: (N,4), u: (M,2), t: (K,) -> (N,M,K,4)。

    相対加速度 u(= u_a - u_d、ここでは u_d = 0)が一定のとき
        dv(t) = dv0 e^{-t/τ} + τ u (1 - e^{-t/τ})
        dp(t) = dp0 + τ dv0 (1 - e^{-t/τ}) + τ u (t - τ (1 - e^{-t/τ}))
    """
    dp0 = x[:, None, None, :2]
    dv0 = x[:, None, None, 2:]
    uu = u[None, :, None, :]
    tt = t[None, None, :, None]
    decay = 1.0 - np.exp(-tt / tau)
    dv = dv0 * (1.0 - decay) + tau * uu * decay
    dp = dp0 + tau * dv0 * decay + tau * uu * (tt - tau * decay)
    return np.concatenate([dp, dv], axis=-1)


def attacker_controls(a_max: float, n_directions: int = N_DIRECTIONS) -> np.ndarray:
    ang = np.linspace(0.0, 2 * np.pi, n_directions, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1) * a_max
    return np.vstack([np.zeros((1, 2)), dirs])


def delayed_values(
    ra: ReachAvoidResult,
    states: np.ndarray,
    delay: float,
    horizon: float,
    return_best_control: bool = False,
):
    """反応遅れ delay [s] のもとでの価値 V_δ を、各状態 (N,4) について返す。"""
    states = np.asarray(states, dtype=float)
    if delay <= 0:
        v0 = ra.values_at(states, horizon)
        return (v0, np.zeros((len(states), 2))) if return_best_control else v0

    tau = ra.dynamics.tau
    controls = attacker_controls(ra.dynamics.a_max_attacker)
    t_win = np.linspace(0.0, delay, N_WINDOW_SAMPLES)
    traj = _propagate_relative(states, controls, t_win, tau)  # (N, M, K, 4)

    l_win = (ra.s_pass - traj[..., 0]).min(axis=-1)  # 窓内で一度でも抜けたか
    g_win = (ra.r_capture - np.linalg.norm(traj[..., :2], axis=-1)).max(axis=-1)  # 窓内で間合いに入ったか
    end_states = traj[:, :, -1, :].reshape(-1, 4)
    v_end = ra.values_at(end_states, horizon - delay).reshape(len(states), len(controls))
    # 遅れ期間終了時の状態がグリッド外(NaN)なら、その攻撃者の選択肢は評価できないので除外する
    v_end = np.where(np.isnan(v_end), np.inf, v_end)
    v_per_u = np.maximum(g_win, np.minimum(l_win, v_end))
    best = v_per_u.argmin(axis=1)
    v_delta = v_per_u[np.arange(len(states)), best]
    v_delta = np.where(np.isinf(v_delta), np.nan, v_delta)
    if return_best_control:
        return v_delta, controls[best]
    return v_delta
