"""「守備者を抜く」reach-avoid 差分ゲーム(フェーズ7以降)。

フェーズ0〜5の追跡回避ゲーム(攻撃者は捕まらないことだけを目的とする)は、
能力が拮抗すると攻撃者が「真後ろへ逃げる」だけで捕捉を回避できるため、捕捉可能
集合が捕捉半径の近傍に退化していた(results_summary.md 5節、phase4bc)。
実際の1対1で攻撃者が目指すのは逃げることではなく守備者を抜いて前進することなので、
攻撃者の目的をそれに合わせて定式化し直す。

座標系: 初期時刻の攻撃者位置からゴール中心へ向かう方向を s 軸、その左手側を l 軸とする
(ad_drg.evaluation.rotate_to_goal_frame)。相対状態は
    x = (s, l, vs, vl) = R (p_a - p_d, v_a - v_d)
で、s < 0 は守備者が攻撃者よりゴール側にいることを表す。

運動: 藤村・杉原型の減衰付きダブルインテグレータ(AD モデルの -v/τ 項と同じ構造)
    v_i' = u_i - v_i / τ,  ||u_i|| <= a_max_i   (i = a, d)
τ を攻守共通にすると相対状態だけで閉じる: dv' = u_a - u_d - dv / τ。
各選手の速度は自動的に a_max_i τ 以下に抑えられる(フェーズ0〜5のモデルには
速度上限が入っていなかった)。

ゲーム(攻撃者視点の reach-avoid):
    目標集合 T = { s >= s_pass }            (守備者より s_pass [m] 以上ゴール側に出る = 抜いた)
    回避集合 C = { ||(s, l)|| <= r_capture } (守備者の間合いに入る = 止められた)
攻撃者(min側)は時間 t 以内に C を踏まずに T へ到達したい。守備者(max側)はそれを阻止したい。
価値関数 V(x, t) <= 0 なら「守備者がどう対応しても攻撃者は t 秒以内に抜ける」、
V > 0 なら「守備者は最適に対応すれば t 秒間は抜かれない」ことを意味する。
"""

from __future__ import annotations

from dataclasses import dataclass

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

DEFAULT_R_CAPTURE = 1.5  # m
DEFAULT_S_PASS = 1.0  # m


class DamperPassDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """相対座標系・減衰付きダブルインテグレータ。攻撃者が control(min)、守備者が disturbance(max)。"""

    def __init__(self, a_max_attacker: float, a_max_defender: float, tau: float):
        self.a_max_attacker = a_max_attacker
        self.a_max_defender = a_max_defender
        self.tau = tau
        super().__init__(
            control_mode="min",
            disturbance_mode="max",
            control_space=hj.sets.Ball(jnp.zeros(2), a_max_attacker),
            disturbance_space=hj.sets.Ball(jnp.zeros(2), a_max_defender),
        )

    def open_loop_dynamics(self, state, time):
        _, _, vs, vl = state
        return jnp.array([vs, vl, -vs / self.tau, -vl / self.tau])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])


@dataclass(frozen=True)
class GridSpec:
    s_lo: float = -20.0
    s_hi: float = 8.0
    l_bound: float = 15.0
    v_bound: float = 16.0
    shape: tuple[int, int, int, int] = (41, 41, 25, 25)


def target_margin(states, s_pass: float):
    """l(x): 目標集合 T の内側で <= 0。"""
    return s_pass - states[..., 0]


def avoid_margin(states, r_capture: float):
    """g(x): 回避集合 C の内側で > 0。"""
    return r_capture - jnp.linalg.norm(states[..., :2], axis=-1)


@dataclass
class ReachAvoidResult:
    grid: hj.Grid
    times: np.ndarray  # 0, -dt, ..., -horizon
    values: np.ndarray  # (len(times), *grid.shape)
    dynamics: DamperPassDynamics
    s_pass: float
    r_capture: float

    def _time_index(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - (-abs(t)))))

    def value_at(self, state: np.ndarray, t: float) -> float:
        t_idx = self._time_index(t)
        return float(self.grid.interpolate(jnp.asarray(self.values[t_idx]), jnp.asarray(state)))

    def values_at(self, states: np.ndarray, t: float) -> np.ndarray:
        """複数状態 (N, 4) をまとめて補間する。"""
        t_idx = self._time_index(t)
        vals = jnp.asarray(self.values[t_idx])
        interp = lambda x: self.grid.interpolate(vals, x)
        import jax

        return np.asarray(jax.vmap(interp)(jnp.asarray(states)))


def solve_reach_avoid(
    a_max_attacker: float,
    a_max_defender: float,
    tau: float,
    grid_spec: GridSpec = GridSpec(),
    horizon: float = 3.0,
    n_time_steps: int = 31,
    s_pass: float = DEFAULT_S_PASS,
    r_capture: float = DEFAULT_R_CAPTURE,
    accuracy: str = "medium",
) -> ReachAvoidResult:
    domain = hj.sets.Box(
        lo=jnp.array([grid_spec.s_lo, -grid_spec.l_bound, -grid_spec.v_bound, -grid_spec.v_bound]),
        hi=jnp.array([grid_spec.s_hi, grid_spec.l_bound, grid_spec.v_bound, grid_spec.v_bound]),
    )
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(domain, grid_spec.shape)
    dynamics = DamperPassDynamics(a_max_attacker, a_max_defender, tau)

    l_x = target_margin(grid.states, s_pass)
    g_x = avoid_margin(grid.states, r_capture)
    # reach-avoid の標準的な後処理(Fisac et al. 2015): 各ステップで
    #   V <- max(min(V, l), g)
    # min(·, l) で「途中で T に入れば到達済み」(tube 化)、max(·, g) で「C に入ったら失敗」を課す。
    settings = hj.SolverSettings.with_accuracy(
        accuracy,
        value_postprocessor=lambda t, v: jnp.maximum(jnp.minimum(v, l_x), g_x),
    )
    times = jnp.linspace(0.0, -horizon, n_time_steps)
    initial_values = jnp.maximum(l_x, g_x)
    values = hj.solve(settings, dynamics, grid, times, initial_values, progress_bar=False)
    return ReachAvoidResult(
        grid=grid,
        times=np.asarray(times),
        values=np.asarray(values),
        dynamics=dynamics,
        s_pass=s_pass,
        r_capture=r_capture,
    )
