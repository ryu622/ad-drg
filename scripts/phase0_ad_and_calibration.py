"""フェーズ0(b)(c): AD モデルの境界張り付き現象の再現確認 と 運動制約のキャリブレーション。

(b) Narizuka & Yamazaki (2026) が報告した「最適化パラメータが探索範囲の境界に
    張り付く現象」が idsse-data でも観測されるかを、AD モデルの独立最適化
    (3.2節: 相手の実軌道を固定して自分側のみ最適化)を再実装して確認する。
(c) a_max, v_max をidsse-dataの実測値から分布推定できるか確認する
    (5節(c))。こちらは全1対1イベントに対して行う(最適化不要で軽量)。

注意: AD モデルのパラメータ探索範囲は Narizuka & Yamazaki (2026) 本文に
明記された数値を参照していない(未入手)。ここでは物理的に妥当な範囲を
仮置きしている。フェーズ1本実装時に原著の探索範囲へ差し替える必要がある。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from kloppy import sportec
from kloppy.domain import AttackingDirection, BallState, Ground
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from scripts.phase0_extract_1v1 import (
    MAX_CARRIER_DIST,
    MIN_DISPLACEMENT,
    MIN_DURATION,
)

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
GOAL_Y_FRAC = 0.5  # ゴール中心のy座標(ピッチ幅に対する比率)
N_SAMPLE_PER_MATCH = 6  # AD モデル最適化(b)に使うサンプル数/試合(計算コストのため間引く)
BOUNDARY_TOL = 0.03  # 探索範囲の何%以内なら「張り付いた」とみなすか

# AD モデルパラメータの探索範囲(暫定値。原著未参照のため物理的に妥当な範囲で仮置き)
BOUNDS = dict(
    f=(0.0, 10.0),      # 駆動力の大きさ [m/s^2]
    tau=(0.1, 5.0),      # 速度緩和時定数 [s]
    beta=(-3.0, 3.0),    # 目標志向バランス [無次元]
)
PARAM_NAMES = ["f", "tau", "beta"]


@dataclass
class Trajectory:
    match_id: str
    attacker: str
    defender: str
    dt: float
    t: np.ndarray
    p_a: np.ndarray  # (N, 2) m
    p_d: np.ndarray
    v_a: np.ndarray  # savgolによる速度推定 (N, 2) m/s
    v_d: np.ndarray
    goal: np.ndarray  # (2,) attacker が狙うゴール = defender の自陣ゴール


def frame_positions_m(frame, pitch_length: float, pitch_width: float):
    pos = {}
    for player, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue
        pos[player.player_id] = np.array(
            [pdata.coordinates.x * pitch_length, pdata.coordinates.y * pitch_width]
        )
    return pos


def goal_position(team_ground: Ground, attacking_direction: AttackingDirection, pitch_length: float, pitch_width: float):
    home_attacks_ltr = attacking_direction == AttackingDirection.LTR
    team_attacks_ltr = home_attacks_ltr if team_ground == Ground.HOME else not home_attacks_ltr
    goal_x = pitch_length if team_attacks_ltr else 0.0
    return np.array([goal_x, pitch_width * GOAL_Y_FRAC])


def smooth_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    n = len(pos)
    window = min(9, n if n % 2 == 1 else n - 1)
    window = max(window, 5) if n >= 5 else n
    if window < 3:
        # too short to filter: fall back to simple finite differences
        v = np.gradient(pos, axis=0) * fps
        return v
    if window % 2 == 0:
        window -= 1
    polyorder = min(3, window - 1)
    return savgol_filter(pos, window_length=window, polyorder=polyorder, deriv=1, delta=1.0 / fps, axis=0)


def build_trajectories(match_id: str) -> list[Trajectory]:
    tracking = sportec.load_open_tracking_data(match_id=match_id, only_alive=True)
    fps = tracking.metadata.frame_rate
    dt = 1.0 / fps
    pitch_length = tracking.metadata.pitch_dimensions.pitch_length
    pitch_width = tracking.metadata.pitch_dimensions.pitch_width
    team_by_player = {p.player_id: team.team_id for team in tracking.metadata.teams for p in team.players}
    ground_by_team = {team.team_id: team.ground for team in tracking.metadata.teams}

    segments: list[tuple[str, list, AttackingDirection]] = []
    cur_carrier, cur_frames, cur_dir = None, [], None

    def flush():
        if cur_carrier is not None and len(cur_frames) >= 2:
            segments.append((cur_carrier, cur_frames, cur_dir))

    for frame in tracking.records:
        if frame.ball_coordinates is None or frame.ball_state != BallState.ALIVE:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            continue
        owning_team = frame.ball_owning_team
        pos = frame_positions_m(frame, pitch_length, pitch_width)
        if owning_team is None or not pos:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            continue
        ball_xy = np.array([frame.ball_coordinates.x * pitch_length, frame.ball_coordinates.y * pitch_width])
        candidates = [(pid, xy) for pid, xy in pos.items() if team_by_player.get(pid) == owning_team.team_id]
        if not candidates:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            continue
        carrier_id, carrier_xy = min(candidates, key=lambda kv: np.linalg.norm(kv[1] - ball_xy))
        if np.linalg.norm(carrier_xy - ball_xy) > MAX_CARRIER_DIST:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            continue
        if carrier_id != cur_carrier:
            flush()
            cur_carrier = carrier_id
            cur_frames = []
            cur_dir = frame.attacking_direction
        cur_frames.append(pos)
    flush()

    trajectories = []
    for carrier_id, frames, atk_dir in segments:
        duration = len(frames) / fps
        if duration <= MIN_DURATION:
            continue
        attacker_team = team_by_player[carrier_id]
        start_pos = frames[0]
        if carrier_id not in start_pos:
            continue

        def nearest_opponent(pos):
            opp = [(pid, xy) for pid, xy in pos.items() if team_by_player.get(pid) != attacker_team]
            if not opp or carrier_id not in pos:
                return None
            return min(opp, key=lambda kv: np.linalg.norm(kv[1] - pos[carrier_id]))[0]

        defender_id = nearest_opponent(start_pos)
        if defender_id is None:
            continue
        stable = all(nearest_opponent(pos) == defender_id for pos in frames)
        if not stable:
            continue
        if not all(carrier_id in pos and defender_id in pos for pos in frames):
            continue

        p_a = np.array([pos[carrier_id] for pos in frames])
        p_d = np.array([pos[defender_id] for pos in frames])
        atk_dist = float(np.linalg.norm(np.diff(p_a, axis=0), axis=1).sum())
        def_dist = float(np.linalg.norm(np.diff(p_d, axis=0), axis=1).sum())
        if atk_dist <= MIN_DISPLACEMENT or def_dist <= MIN_DISPLACEMENT:
            continue

        v_a = smooth_velocity(p_a, fps)
        v_d = smooth_velocity(p_d, fps)
        goal = goal_position(ground_by_team[attacker_team], atk_dir, pitch_length, pitch_width)
        t = np.arange(len(p_a)) * dt

        trajectories.append(
            Trajectory(match_id=match_id, attacker=carrier_id, defender=defender_id, dt=dt, t=t, p_a=p_a, p_d=p_d, v_a=v_a, v_d=v_d, goal=goal)
        )
    return trajectories


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


def fit_side(traj: Trajectory, side: str) -> dict:
    simulate = simulate_attacker if side == "attacker" else simulate_defender
    bounds = [BOUNDS["f"], BOUNDS["tau"], BOUNDS["beta"]]

    def objective(x):
        f, tau, beta = x
        sim = simulate(traj, f, tau, beta)
        obs = traj.p_a if side == "attacker" else traj.p_d
        return error_metric(sim, obs)

    rng = np.random.default_rng(hash((traj.match_id, traj.attacker, traj.defender, side)) % (2**32))
    best = None
    for _ in range(4):
        x0 = [rng.uniform(*bounds[0]), rng.uniform(*bounds[1]), rng.uniform(*bounds[2])]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res

    f, tau, beta = best.x
    at_bound = {}
    for name, val, (lo, hi) in zip(PARAM_NAMES, [f, tau, beta], bounds):
        span = hi - lo
        at_bound[name] = bool(val <= lo + BOUNDARY_TOL * span or val >= hi - BOUNDARY_TOL * span)

    return dict(f=f, tau=tau, beta=beta, error=best.fun, at_bound=at_bound, any_at_bound=any(at_bound.values()))


def calibration_stats(trajectories: list[Trajectory]) -> dict:
    max_speeds, max_accels = [], []
    for traj in trajectories:
        for v in (traj.v_a, traj.v_d):
            speeds = np.linalg.norm(v, axis=1)
            max_speeds.append(speeds.max())
            acc = np.gradient(v, traj.dt, axis=0)
            accels = np.linalg.norm(acc, axis=1)
            max_accels.append(accels.max())
    max_speeds = np.array(max_speeds)
    max_accels = np.array(max_accels)
    return dict(
        v_max_p5=float(np.percentile(max_speeds, 5)),
        v_max_p50=float(np.percentile(max_speeds, 50)),
        v_max_p95=float(np.percentile(max_speeds, 95)),
        v_max_max=float(max_speeds.max()),
        a_max_p5=float(np.percentile(max_accels, 5)),
        a_max_p50=float(np.percentile(max_accels, 50)),
        a_max_p95=float(np.percentile(max_accels, 95)),
        a_max_max=float(max_accels.max()),
        n_player_segments=len(max_speeds),
    )


def main():
    rng = np.random.default_rng(0)
    all_trajectories: list[Trajectory] = []
    fit_results = []

    t0 = time.time()
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
        print(f"[{match_id}] trajectories={len(trajs)} elapsed={time.time() - t0:.1f}s", flush=True)

        sample_idx = rng.choice(len(trajs), size=min(N_SAMPLE_PER_MATCH, len(trajs)), replace=False)
        for i in sample_idx:
            traj = trajs[i]
            atk_fit = fit_side(traj, "attacker")
            def_fit = fit_side(traj, "defender")
            fit_results.append(
                dict(match_id=match_id, attacker=traj.attacker, defender=traj.defender, n_frames=len(traj.p_a), attacker_fit=atk_fit, defender_fit=def_fit)
            )
        print(f"[{match_id}] AD fit done for {len(sample_idx)} sampled events elapsed={time.time() - t0:.1f}s", flush=True)

    # (b) 境界張り付き現象の集計
    n_total_fits = len(fit_results) * 2
    n_at_bound = sum(r["attacker_fit"]["any_at_bound"] for r in fit_results) + sum(r["defender_fit"]["any_at_bound"] for r in fit_results)
    per_param_bound_rate = {}
    for name in PARAM_NAMES:
        cnt = sum(r["attacker_fit"]["at_bound"][name] for r in fit_results) + sum(r["defender_fit"]["at_bound"][name] for r in fit_results)
        per_param_bound_rate[name] = cnt / n_total_fits

    errors = [r["attacker_fit"]["error"] for r in fit_results] + [r["defender_fit"]["error"] for r in fit_results]

    print("\n=== (b) AD モデル境界張り付き現象 ===")
    print(f"サンプル数: {len(fit_results)}イベント x 2(攻撃者/守備者) = {n_total_fits}回の独立最適化")
    print(f"いずれかのパラメータが境界に張り付いた割合: {n_at_bound / n_total_fits:.1%}")
    print(f"パラメータ別の張り付き率: {per_param_bound_rate}")
    print(f"誤差 ε_p: mean={np.mean(errors):.3f}, median={np.median(errors):.3f}")

    # (c) 運動制約キャリブレーション
    calib = calibration_stats(all_trajectories)
    print("\n=== (c) 運動制約キャリブレーション ===")
    print(f"v_max [m/s]: 5%ile={calib['v_max_p5']:.2f}, 中央値={calib['v_max_p50']:.2f}, 95%ile={calib['v_max_p95']:.2f}, 最大={calib['v_max_max']:.2f}")
    print(f"a_max [m/s^2]: 5%ile={calib['a_max_p5']:.2f}, 中央値={calib['a_max_p50']:.2f}, 95%ile={calib['a_max_p95']:.2f}, 最大={calib['a_max_max']:.2f}")
    print(f"(選手xセグメント数: {calib['n_player_segments']})")

    out = dict(
        bounds_used=BOUNDS,
        boundary_tolerance=BOUNDARY_TOL,
        n_fit_events=len(fit_results),
        boundary_sticking_rate_overall=n_at_bound / n_total_fits,
        boundary_sticking_rate_per_param=per_param_bound_rate,
        error_mean=float(np.mean(errors)),
        error_median=float(np.median(errors)),
        fit_results=fit_results,
        calibration=calib,
    )
    out_path = "documents/phase0_bc_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")


if __name__ == "__main__":
    main()
