"""Attacker-Defender (AD) モデル: Narizuka & Yamazaki (2026, arXiv:2512.22596) の
「改良推定法」(相手の実軌道を固定した独立最適化)の再実装。

運動方程式(論文 式(1)(2)):
    dv_a/dt = -v_a/tau_a + f_a * (beta_a*e_ag - e_ad) / ||beta_a*e_ag - e_ad||
    dv_d/dt = -v_d/tau_d + f_d * (beta_d*e_dg + e_da) / ||beta_d*e_dg + e_da||

パラメータ最適化(論文 2.3節):
    - COBYLA (scipy.optimize.minimize)
    - 初期値: ラテン超方格サンプリングで N 個生成
        f in [-11.3, 11.3], beta in [-5, 5], tau in [0.9, 2.5]
    - 最適化中の制約: tau >= 0.9 (上限なし), beta in [-10, 10], |f*tau| <= 10.2
    - 誤差指標 (論文 式(3)):
        eps_p = (1/T) * sum_t ||r_p(t)-r_p'(t)|| / sum_t ||r_p(t)-r_p(t-1)||
    - 採択基準: eps_a < 0.1 かつ eps_d < 0.1 (原著は31,028件中27,457件=88.5%が該当)

論文はN=100(31,028件全件、フル解像度)で実行しているが、ローカル開発環境では
計算コストの都合上デフォルトN_INIT_DEFAULTを小さくしている。全件・N=100での
本番実行はColab側で行う想定(research_plan.md 4.5節)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.stats import qmc

from ad_drg.extraction import Trajectory

F_LHS_RANGE = (-11.3, 11.3)
BETA_LHS_RANGE = (-5.0, 5.0)
TAU_LHS_RANGE = (0.9, 2.5)

TAU_MIN = 0.9
BETA_BOUND = 10.0
F_TAU_MAX = 10.2

ACCURATE_ERROR_THRESHOLD = 0.1  # 論文: eps_p < 0.1 で「正確に再現」と判定
N_INIT_DEFAULT = 20  # ローカル開発用(論文は N=100)
BOUNDARY_TOL = 0.03  # 境界からこの割合以内なら「張り付いた」とみなす


def unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def simulate_attacker(traj: Trajectory, f: float, tau: float, beta: float) -> np.ndarray:
    p = traj.p_a[0].copy()
    v = traj.v_a[0].copy()
    out = [p.copy()]
    for k in range(len(traj.p_a) - 1):
        p_d = traj.p_d[k]
        e_ag = unit(traj.goal - p)
        e_ad = unit(p_d - p)
        drive = unit(beta * e_ag - e_ad)
        v = v + (-v / tau + f * drive) * traj.dt
        p = p + v * traj.dt
        out.append(p.copy())
    return np.array(out)


def simulate_defender(traj: Trajectory, f: float, tau: float, beta: float) -> np.ndarray:
    p = traj.p_d[0].copy()
    v = traj.v_d[0].copy()
    out = [p.copy()]
    for k in range(len(traj.p_d) - 1):
        p_a = traj.p_a[k]
        e_dg = unit(traj.goal - p)  # 攻撃者の狙うゴール = 守備者の自陣ゴール
        e_da = unit(p_a - p)
        drive = unit(beta * e_dg + e_da)
        v = v + (-v / tau + f * drive) * traj.dt
        p = p + v * traj.dt
        out.append(p.copy())
    return np.array(out)


def error_metric(sim: np.ndarray, obs: np.ndarray) -> float:
    denom = np.linalg.norm(np.diff(obs, axis=0), axis=1).sum()
    if denom < 1e-6:
        return np.inf
    num = np.linalg.norm(obs - sim, axis=1).mean()
    return num / denom


def _lhs_initial_points(n_init: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n_init)
    lo = np.array([F_LHS_RANGE[0], TAU_LHS_RANGE[0], BETA_LHS_RANGE[0]])
    hi = np.array([F_LHS_RANGE[1], TAU_LHS_RANGE[1], BETA_LHS_RANGE[1]])
    return qmc.scale(unit_samples, lo, hi)


def fit_side(traj: Trajectory, side: str, n_init: int = N_INIT_DEFAULT, seed: int | None = None) -> dict:
    """論文2.3節の独立最適化を1イベント・1サイド分実行する。

    x = [f, tau, beta]
    """
    simulate = simulate_attacker if side == "attacker" else simulate_defender
    obs = traj.p_a if side == "attacker" else traj.p_d

    def objective(x):
        f, tau, beta = x
        sim = simulate(traj, f, tau, beta)
        return error_metric(sim, obs)

    bounds = Bounds([-np.inf, TAU_MIN, -BETA_BOUND], [np.inf, np.inf, BETA_BOUND])
    constraints = [{"type": "ineq", "fun": lambda x: F_TAU_MAX - abs(x[0] * x[1])}]

    if seed is None:
        seed = hash((traj.match_id, traj.attacker, traj.defender, side)) % (2**32)
    x0_list = _lhs_initial_points(n_init, seed)

    best = None
    for x0 in x0_list:
        # ラテン超方格の初期値が制約を破っていたら最寄りの実行可能点に射影する
        x0 = x0.copy()
        x0[1] = max(x0[1], TAU_MIN)
        max_f = F_TAU_MAX / x0[1]
        x0[0] = np.clip(x0[0], -max_f, max_f)
        res = minimize(objective, x0, method="COBYLA", bounds=bounds, constraints=constraints)
        if best is None or res.fun < best.fun:
            best = res

    f, tau, beta = best.x
    at_bound = {
        "f": bool(abs(abs(f * tau) - F_TAU_MAX) <= BOUNDARY_TOL * F_TAU_MAX),
        "tau": bool(tau <= TAU_MIN * (1 + BOUNDARY_TOL)),
        "beta": bool(abs(beta) >= BETA_BOUND * (1 - BOUNDARY_TOL)),
    }

    return dict(
        f=float(f),
        tau=float(tau),
        beta=float(beta),
        error=float(best.fun),
        accurate=bool(best.fun < ACCURATE_ERROR_THRESHOLD),
        at_bound=at_bound,
        any_at_bound=any(at_bound.values()),
    )


@dataclass
class EventFit:
    match_id: str
    attacker: str
    defender: str
    n_frames: int
    attacker_fit: dict
    defender_fit: dict

    @property
    def both_accurate(self) -> bool:
        return self.attacker_fit["accurate"] and self.defender_fit["accurate"]


def fit_event(traj: Trajectory, n_init: int = N_INIT_DEFAULT) -> EventFit:
    atk_fit = fit_side(traj, "attacker", n_init=n_init)
    def_fit = fit_side(traj, "defender", n_init=n_init)
    return EventFit(
        match_id=traj.match_id,
        attacker=traj.attacker,
        defender=traj.defender,
        n_frames=len(traj.p_a),
        attacker_fit=atk_fit,
        defender_fit=def_fit,
    )
