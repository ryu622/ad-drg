"""フェーズ4c: 守備者優位比a_max_defender/a_max_attackerを掃引し、
捕捉可能集合の退化がどの程度の能力差で解消するかを調べる感度分析。

フェーズ4b(役割別キャリブレーション、差約4%)では退化が解消しなかった
(V*<=0割合9.3%、守備側の非最適性0件、いずれも対称ケースとほぼ同じ)。
実データの自然な非対称性が小さすぎたのか、それとも根本的に捕捉半径付近でしか
捕捉可能集合が広がらない構造なのかを切り分けるため、a_max_defenderを
人為的に攻撃者比1.0倍〜4.0倍まで掃引し、frac_v_star_capturable・
守備側非最適性件数がどう変化するかを確認する。

出力: documents/phase4c_asymmetric_sweep_results.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.calibration import calibration_stats_by_role
from ad_drg.extraction import MATCH_IDS, build_trajectories
from ad_drg.reachability import DEFAULT_R_CAPTURE, solve_brs

P_BOUND = 25.0
GRID_SHAPE = (31, 31, 21, 21)
TIME_HORIZON = 3.0
N_TIME_STEPS = 21
ACCURACY = "medium"

DEFENDER_ADVANTAGE_RATIOS = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def main():
    t0 = time.time()
    all_trajectories = []
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
    print(f"全軌道数={len(all_trajectories)} elapsed={time.time() - t0:.1f}s", flush=True)

    calib_role = calibration_stats_by_role(all_trajectories)
    a_max_attacker = calib_role["attacker"]["a_max_p95"]
    v_max_attacker = calib_role["attacker"]["v_max_p95"]
    v_max_defender = calib_role["defender"]["v_max_p95"]
    v_bound = v_max_attacker + v_max_defender
    r_capture = DEFAULT_R_CAPTURE

    print(f"a_max_attacker(固定)={a_max_attacker:.2f} m/s^2, v_bound={v_bound:.2f} m/s")

    x0_list, dist0_list = [], []
    for traj in all_trajectories:
        dp0 = traj.p_a[0] - traj.p_d[0]
        dv0 = traj.v_a[0] - traj.v_d[0]
        x0_list.append(np.concatenate([dp0, dv0]))
        dist0_list.append(float(np.linalg.norm(dp0)))

    results = []
    for ratio in DEFENDER_ADVANTAGE_RATIOS:
        a_max_defender = a_max_attacker * ratio
        t1 = time.time()
        brs = solve_brs(
            a_max_attacker=a_max_attacker,
            a_max_defender=a_max_defender,
            p_bound=P_BOUND,
            v_bound=v_bound,
            grid_shape=GRID_SHAPE,
            time_horizon=TIME_HORIZON,
            n_time_steps=N_TIME_STEPS,
            accuracy=ACCURACY,
            progress_bar=False,
        )
        solve_time = time.time() - t1

        v_star_list = []
        for x0, traj in zip(x0_list, all_trajectories):
            duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
            v_star_list.append(brs.value_at(x0, t=min(duration, TIME_HORIZON)))
        v_star_arr = np.array(v_star_list)
        valid = ~np.isnan(v_star_arr)

        v_obs_list = []
        for traj in all_trajectories:
            dist = np.linalg.norm(traj.p_a - traj.p_d, axis=1)
            v_obs_list.append(float(dist.min() - r_capture))
        v_obs_arr = np.array(v_obs_list)

        n_valid = int(valid.sum())
        frac_capturable = float((v_star_arr[valid] <= 0).mean())
        n_defender_subopt = int(np.sum(valid & (v_star_arr <= 0) & (v_obs_arr > 0)))
        n_attacker_subopt = int(np.sum(valid & (v_star_arr > 0) & (v_obs_arr <= 0)))

        # 相対速度ゼロ状態での捕捉可能半径(4次元グリッドの中心スライス)を目安として記録
        zero_v_idx = np.array([0.0, 0.0, 0.0, 0.0])
        radii = []
        for r in np.linspace(r_capture, P_BOUND, 60):
            state = np.array([r, 0.0, 0.0, 0.0])
            v = brs.value_at(state, t=TIME_HORIZON)
            if not np.isnan(v) and v <= 0:
                radii.append(r)
        capture_radius_zero_v = max(radii) if radii else r_capture

        row = dict(
            ratio=ratio,
            a_max_defender=a_max_defender,
            n_valid=n_valid,
            frac_v_star_capturable=frac_capturable,
            n_defender_suboptimal=n_defender_subopt,
            n_attacker_suboptimal=n_attacker_subopt,
            capture_radius_at_zero_relvel=float(capture_radius_zero_v),
            solve_time_sec=solve_time,
        )
        results.append(row)
        print(
            f"ratio={ratio:.2f} a_max_d={a_max_defender:.2f} "
            f"frac_capturable={frac_capturable:.1%} defender_subopt={n_defender_subopt} "
            f"attacker_subopt={n_attacker_subopt} capture_radius@v=0={capture_radius_zero_v:.2f}m "
            f"solve={solve_time:.1f}s",
            flush=True,
        )

    out = dict(
        a_max_attacker=a_max_attacker,
        v_max_attacker=v_max_attacker,
        v_max_defender=v_max_defender,
        v_bound=v_bound,
        r_capture=r_capture,
        n_events=len(all_trajectories),
        sweep=results,
        elapsed_sec=time.time() - t0,
    )
    out_path = "documents/phase4c_asymmetric_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")


if __name__ == "__main__":
    main()
