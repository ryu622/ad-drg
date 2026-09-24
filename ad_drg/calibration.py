"""運動制約(a_max, v_max)のキャリブレーション(research_plan.md 5節(c))。

フェーズ0(c)で、生データに非物理的な外れ値(v_max最大244.84 m/s等)が
含まれることが判明した(トラッキング欠損区間での位置ジャンプ、または短い
セグメントでのSavitzky-Golay微分のノイズ増幅が原因と推定)。本モジュールでは
人間の走行として明らかに非物理的な値を足切りしたうえで分布を推定する。
"""

from __future__ import annotations

import numpy as np

from ad_drg.extraction import Trajectory

# 足切り閾値: エリート短距離走者のトップスピード(~12.4 m/s, ウサイン・ボルト)、
# 初速加速度(~10 m/s^2)を参考に、明らかに非物理的な値のみを除外する目的で
# 少し余裕を持たせた値を設定。
PHYSICAL_MAX_SPEED = 12.0  # m/s
PHYSICAL_MAX_ACCEL = 12.0  # m/s^2


def _role_stats(
    trajectories: list[Trajectory],
    velocities: list[np.ndarray],
    speed_cutoff: float,
    accel_cutoff: float,
) -> dict:
    max_speeds, max_accels = [], []
    n_outlier_speed = 0
    n_outlier_accel = 0

    for traj, v in zip(trajectories, velocities):
        speeds = np.linalg.norm(v, axis=1)
        acc = np.gradient(v, traj.dt, axis=0)
        accels = np.linalg.norm(acc, axis=1)
        ms, ma = speeds.max(), accels.max()
        if ms > speed_cutoff:
            n_outlier_speed += 1
            continue
        if ma > accel_cutoff:
            n_outlier_accel += 1
            continue
        max_speeds.append(ms)
        max_accels.append(ma)

    max_speeds = np.array(max_speeds)
    max_accels = np.array(max_accels)
    return dict(
        v_max_p50=float(np.percentile(max_speeds, 50)),
        v_max_p95=float(np.percentile(max_speeds, 95)),
        v_max_p99=float(np.percentile(max_speeds, 99)),
        v_max_max=float(max_speeds.max()),
        a_max_p50=float(np.percentile(max_accels, 50)),
        a_max_p95=float(np.percentile(max_accels, 95)),
        a_max_p99=float(np.percentile(max_accels, 99)),
        a_max_max=float(max_accels.max()),
        n_player_segments=int(len(max_speeds)),
        n_outlier_speed_removed=n_outlier_speed,
        n_outlier_accel_removed=n_outlier_accel,
        speed_cutoff=speed_cutoff,
        accel_cutoff=accel_cutoff,
    )


def calibration_stats_by_role(
    trajectories: list[Trajectory],
    speed_cutoff: float = PHYSICAL_MAX_SPEED,
    accel_cutoff: float = PHYSICAL_MAX_ACCEL,
) -> dict:
    """攻撃者役・守備者役を分離してキャリブレーションする(非対称a_max用)。

    calibration_stats は攻守のv/aをプールして1つの分布にするが、こちらは
    role(attacker=ボール保持者, defender=最近傍守備者)ごとに独立した分布を返す。
    """
    return dict(
        attacker=_role_stats(trajectories, [t.v_a for t in trajectories], speed_cutoff, accel_cutoff),
        defender=_role_stats(trajectories, [t.v_d for t in trajectories], speed_cutoff, accel_cutoff),
    )


def calibration_stats(
    trajectories: list[Trajectory],
    speed_cutoff: float = PHYSICAL_MAX_SPEED,
    accel_cutoff: float = PHYSICAL_MAX_ACCEL,
) -> dict:
    max_speeds, max_accels = [], []
    n_outlier_speed = 0
    n_outlier_accel = 0

    for traj in trajectories:
        for v in (traj.v_a, traj.v_d):
            speeds = np.linalg.norm(v, axis=1)
            acc = np.gradient(v, traj.dt, axis=0)
            accels = np.linalg.norm(acc, axis=1)
            ms, ma = speeds.max(), accels.max()
            if ms > speed_cutoff:
                n_outlier_speed += 1
                continue
            if ma > accel_cutoff:
                n_outlier_accel += 1
                continue
            max_speeds.append(ms)
            max_accels.append(ma)

    max_speeds = np.array(max_speeds)
    max_accels = np.array(max_accels)
    return dict(
        v_max_p50=float(np.percentile(max_speeds, 50)),
        v_max_p95=float(np.percentile(max_speeds, 95)),
        v_max_p99=float(np.percentile(max_speeds, 99)),
        v_max_max=float(max_speeds.max()),
        a_max_p50=float(np.percentile(max_accels, 50)),
        a_max_p95=float(np.percentile(max_accels, 95)),
        a_max_p99=float(np.percentile(max_accels, 99)),
        a_max_max=float(max_accels.max()),
        n_player_segments=int(len(max_speeds)),
        n_outlier_speed_removed=n_outlier_speed,
        n_outlier_accel_removed=n_outlier_accel,
        speed_cutoff=speed_cutoff,
        accel_cutoff=accel_cutoff,
    )
