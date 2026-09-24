"""Hamilton-Jacobi-Isaacs (HJI) による捕捉可能集合(Backward Reachable Set)の計算。

research_plan.md 3.1節の定式化をそのまま実装する。

状態: x = (dpx, dpy, dvx, dvy), dp = p_a - p_d, dv = v_a - v_d
運動: dp' = dv,  dv' = u_a - u_d,  ||u_a|| <= a_max_a,  ||u_d|| <= a_max_d

HJI方程式:
    dV/dt + min_{u_d} max_{u_a} [grad(V) . f(x,u_a,u_d)] = 0,  V(x,0) = ||dp|| - r_capture

守備者を「control」(min側)、攻撃者を「disturbance」(max側)に割り当てる
(hj_reachability の ControlAndDisturbanceAffineDynamics の命名規則上の対応で、
実際の意味は research_plan.md の min_{u_d} max_{u_a} と一致する)。

V(x,t) <= 0 となる状態集合が、時間|t|以内に守備者が確実に捕捉できる捕捉可能集合。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hj_reachability as hj
import jax.numpy as jnp
import numpy as np

DEFAULT_R_CAPTURE = 1.5  # m


class PursuitEvasionDynamics(hj.ControlAndDisturbanceAffineDynamics):
    """相対座標系での1対1追跡回避ダブルインテグレータ動力学。"""

    def __init__(self, a_max_attacker: float, a_max_defender: float):
        self.a_max_attacker = a_max_attacker
        self.a_max_defender = a_max_defender
        control_space = hj.sets.Ball(jnp.zeros(2), a_max_defender)  # 守備者の加速度(min側)
        disturbance_space = hj.sets.Ball(jnp.zeros(2), a_max_attacker)  # 攻撃者の加速度(max側)
        super().__init__(
            control_mode="min",
            disturbance_mode="max",
            control_space=control_space,
            disturbance_space=disturbance_space,
        )

    def open_loop_dynamics(self, state, time):
        _, _, dvx, dvy = state
        return jnp.array([dvx, dvy, 0.0, 0.0])

    def control_jacobian(self, state, time):
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )

    def disturbance_jacobian(self, state, time):
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )


def terminal_cost(states: jnp.ndarray, r_capture: float = DEFAULT_R_CAPTURE) -> jnp.ndarray:
    dp = states[..., :2]
    return jnp.linalg.norm(dp, axis=-1) - r_capture


def build_grid(
    p_bound: float,
    v_bound: float,
    shape: tuple[int, int, int, int] = (21, 21, 15, 15),
) -> hj.Grid:
    domain = hj.sets.Box(
        lo=jnp.array([-p_bound, -p_bound, -v_bound, -v_bound]),
        hi=jnp.array([p_bound, p_bound, v_bound, v_bound]),
    )
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(domain, shape)


@dataclass
class BRSResult:
    grid: hj.Grid
    times: np.ndarray
    values: np.ndarray  # (len(times), *grid.shape)
    dynamics: PursuitEvasionDynamics
    r_capture: float
    _grad_cache: dict = field(default_factory=dict, repr=False)

    def _time_index(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - (-abs(t)))))

    def value_at(self, state: np.ndarray, t: float) -> float:
        """状態stateにおける、時刻tでの価値関数V(x,t)を補間して返す。

        V(x,t) <= 0 なら、時間|t|以内に守備者が確実に捕捉できることを意味する。
        """
        t_idx = self._time_index(t)
        return float(self.grid.interpolate(jnp.asarray(self.values[t_idx]), jnp.asarray(state)))

    def capturable(self, state: np.ndarray, t: float) -> bool:
        v = self.value_at(state, t)
        return bool(v <= 0) if not np.isnan(v) else False

    def _grad_field(self, t_idx: int) -> np.ndarray:
        if t_idx not in self._grad_cache:
            self._grad_cache[t_idx] = np.asarray(self.grid.grad_values(jnp.asarray(self.values[t_idx])))
        return self._grad_cache[t_idx]

    def grad_at(self, state: np.ndarray, t: float) -> np.ndarray:
        """状態stateにおける、残り時間tでの grad(V) を補間して返す(4次元ベクトル)。"""
        t_idx = self._time_index(t)
        grad_field = self._grad_field(t_idx)
        return np.asarray(self.grid.interpolate(jnp.asarray(grad_field), jnp.asarray(state)))

    def optimal_defender_accel(self, state: np.ndarray, t: float) -> np.ndarray:
        """状態state・残り時間tにおける守備者の最適(min側)加速度ベクトルを返す。"""
        grad_value = self.grad_at(state, t)
        u_star, _ = self.dynamics.optimal_control_and_disturbance(jnp.asarray(state), 0.0, jnp.asarray(grad_value))
        return np.asarray(u_star)

    def optimal_attacker_accel(self, state: np.ndarray, t: float) -> np.ndarray:
        """状態state・残り時間tにおける攻撃者の最適(max側)加速度ベクトルを返す。"""
        grad_value = self.grad_at(state, t)
        _, d_star = self.dynamics.optimal_control_and_disturbance(jnp.asarray(state), 0.0, jnp.asarray(grad_value))
        return np.asarray(d_star)


def solve_brs(
    a_max_attacker: float,
    a_max_defender: float,
    p_bound: float,
    v_bound: float,
    grid_shape: tuple[int, int, int, int] = (21, 21, 15, 15),
    time_horizon: float = 3.0,
    n_time_steps: int = 21,
    accuracy: str = "low",
    r_capture: float = DEFAULT_R_CAPTURE,
    progress_bar: bool = True,
) -> BRSResult:
    grid = build_grid(p_bound, v_bound, grid_shape)
    dynamics = PursuitEvasionDynamics(a_max_attacker=a_max_attacker, a_max_defender=a_max_defender)
    times = jnp.linspace(0.0, -time_horizon, n_time_steps)
    initial_values = terminal_cost(grid.states, r_capture)

    values = hj.solve(
        hj.SolverSettings.with_accuracy(accuracy),
        dynamics=dynamics,
        grid=grid,
        times=times,
        initial_values=initial_values,
        progress_bar=progress_bar,
    )
    values = np.asarray(values)
    # hj.solve は「時刻tちょうどに目標へ到達する」値を各時刻で独立に解くため、
    # 生の出力は時間方向に単調ではない(検証済み: 対称な加速度上限では時間が
    # 経つほど値が増加し続け、「時刻t以内に到達可能」という捕捉可能集合の
    # 意味論と合わない)。時間方向の累積最小値を取ることで、
    # 「0〜|t|のいずれかの時刻で到達可能」という捕捉可能集合(BRT)に変換する。
    values = np.minimum.accumulate(values, axis=0)

    return BRSResult(
        grid=grid,
        times=np.asarray(times),
        values=np.asarray(values),
        dynamics=dynamics,
        r_capture=r_capture,
    )
