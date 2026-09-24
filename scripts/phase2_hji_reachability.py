"""フェーズ2: HJI数値解法の実装・検証(research_plan.md 4.3節)。

1. idsse-dataから運動制約(a_max, v_max)をキャリブレーション(外れ値除去込み、5節(c))
2. `ad_drg.reachability` で4次元グリッド・ダブルインテグレータ動力学を定義し、
   捕捉可能集合(Backward Reachable Set)を計算
3. 実イベントの初期状態を使い、捕捉可能集合による予測と実際の結果を突き合わせる
   簡易デモ(本格的な検証は5節②でフェーズ3として実施)
4. 可視化(dpx-dpy平面のスライス、dvx=dvy=0)を保存

開発時の重要な知見(実装中に発見):
    `hj.solve` の生の出力は「時刻tちょうどに目標に到達するための価値」を
    各時刻で独立に解くため、時間方向に単調ではない(対称な加速度上限では
    時間が経つほど値が発散的に増加する現象を確認)。「時刻t以内に到達可能」
    という捕捉可能集合の意味論に合わせるには、時間方向の累積最小値を取る
    後処理が必須(`ad_drg/reachability.py` の `solve_brs` 内で実施済み)。
    また、低解像度・低精度グリッド(15x15x11x11, accuracy=low)では数値拡散に
    より捕捉可能集合が実際より過小評価されることを確認したため、
    本スクリプトでは中解像度(31x31x21x21, accuracy=medium)を用いる。
    全1,454イベント・本番解像度でのグリッド計算はColab側で実施する。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.calibration import calibration_stats
from ad_drg.extraction import MATCH_IDS, build_trajectories
from ad_drg.reachability import DEFAULT_R_CAPTURE, solve_brs

P_BOUND = 25.0  # m
GRID_SHAPE = (31, 31, 21, 21)
TIME_HORIZON = 3.0  # s (research_plan.md 4.4節の疑似コードに合わせる)
N_TIME_STEPS = 21
ACCURACY = "medium"


def main():
    t0 = time.time()
    all_trajectories = []
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
        print(f"[{match_id}] trajectories={len(trajs)} elapsed={time.time() - t0:.1f}s", flush=True)

    calib = calibration_stats(all_trajectories)
    print("\n=== 運動制約キャリブレーション(外れ値除去後) ===")
    print(
        f"v_max: p50={calib['v_max_p50']:.2f} p95={calib['v_max_p95']:.2f} p99={calib['v_max_p99']:.2f} "
        f"max={calib['v_max_max']:.2f} [m/s]"
    )
    print(
        f"a_max: p50={calib['a_max_p50']:.2f} p95={calib['a_max_p95']:.2f} p99={calib['a_max_p99']:.2f} "
        f"max={calib['a_max_max']:.2f} [m/s^2]"
    )
    print(
        f"除外件数: 速度={calib['n_outlier_speed_removed']}, 加速度={calib['n_outlier_accel_removed']} "
        f"(母数={calib['n_player_segments'] + calib['n_outlier_speed_removed'] + calib['n_outlier_accel_removed']})"
    )

    a_max = calib["a_max_p95"]
    v_max = calib["v_max_p95"]
    # 状態空間は相対速度 dv = v_a - v_d であり、各選手の速度上限が v_max のとき
    # dv は理論上 2*v_max まで到達しうる(正面から逆方向に走る場合)。
    # v_bound を v_max のままにすると、シミュレーション中に相対速度がグリッド範囲外に
    # 出て grid.interpolate が NaN を返す(実装時に発見・修正)。
    v_bound = 2.0 * v_max
    print(f"\n採用値: a_max={a_max:.2f} m/s^2, v_max(個人)={v_max:.2f} m/s, v_bound(相対速度)={v_bound:.2f} m/s")

    print("\n=== HJI 捕捉可能集合を計算 ===")
    t1 = time.time()
    res = solve_brs(
        a_max_attacker=a_max,
        a_max_defender=a_max,
        p_bound=P_BOUND,
        v_bound=v_bound,
        grid_shape=GRID_SHAPE,
        time_horizon=TIME_HORIZON,
        n_time_steps=N_TIME_STEPS,
        accuracy=ACCURACY,
        progress_bar=False,
    )
    print(f"グリッド形状={GRID_SHAPE}, T={TIME_HORIZON}s, 計算時間={time.time() - t1:.1f}s")

    # 実イベントとの簡易照合デモ(本格検証はフェーズ3)
    print("\n=== 実イベント初期状態との簡易照合(デモ) ===")
    rng = np.random.default_rng(0)
    sample = rng.choice(len(all_trajectories), size=min(30, len(all_trajectories)), replace=False)
    n_capturable = 0
    demo_rows = []
    for i in sample:
        traj = all_trajectories[i]
        dp0 = traj.p_a[0] - traj.p_d[0]
        dv0 = traj.v_a[0] - traj.v_d[0]
        x0 = np.concatenate([dp0, dv0])
        duration = traj.t[-1]
        v = res.value_at(x0, t=min(duration, TIME_HORIZON))
        capturable = bool(v <= 0) if not np.isnan(v) else None
        if capturable:
            n_capturable += 1
        demo_rows.append(
            dict(
                match_id=traj.match_id,
                dist0=float(np.linalg.norm(dp0)),
                duration=float(duration),
                value=None if v is None or np.isnan(v) else float(v),
                predicted_capturable=capturable,
            )
        )
    print(f"サンプル{len(sample)}件中、初期状態が捕捉可能集合に入っていたもの: {n_capturable}件")
    print("(注: これは予測精度の検証ではなく、パイプラインの疎通確認。フェーズ3で本格的に実施)")

    out = dict(
        calibration=calib,
        a_max_used=a_max,
        v_max_used=v_max,
        v_bound_used=v_bound,
        p_bound=P_BOUND,
        grid_shape=list(GRID_SHAPE),
        time_horizon=TIME_HORIZON,
        r_capture=DEFAULT_R_CAPTURE,
        accuracy=ACCURACY,
        demo_matching=demo_rows,
        demo_n_capturable=n_capturable,
        demo_n_sample=len(sample),
    )
    out_path = "documents/phase2_hji_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")

    # 可視化: dpx-dpy平面のスライス(dvx=dvy=0)、複数の時刻で捕捉可能集合の境界を描画
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = res.grid
    xs = np.linspace(-P_BOUND, P_BOUND, GRID_SHAPE[0])
    ys = np.linspace(-P_BOUND, P_BOUND, GRID_SHAPE[1])
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    fig, ax = plt.subplots(figsize=(6, 6))
    time_indices = [0, N_TIME_STEPS // 4, N_TIME_STEPS // 2, N_TIME_STEPS - 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    for idx, color in zip(time_indices, colors):
        t_val = float(res.times[idx])
        ZZ = np.array(
            [
                [grid.interpolate(res.values[idx], np.array([x, y, 0.0, 0.0])) for y in ys]
                for x in xs
            ]
        )
        ax.contour(XX, YY, ZZ, levels=[0.0], colors=[color])
        ax.plot([], [], color=color, label=f"t={t_val:.1f}s")
    ax.scatter([0], [0], marker="x", color="red", label="defender (origin)")
    ax.set_xlabel("dpx [m] (attacker - defender)")
    ax.set_ylabel("dpy [m]")
    ax.set_title(f"Backward Reachable Set (dvx=dvy=0)\na_max={a_max:.1f} m/s^2, v_max={v_max:.1f} m/s")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig_path = "documents/phase2_brs_slice.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"可視化を {fig_path} に保存しました")


if __name__ == "__main__":
    main()
