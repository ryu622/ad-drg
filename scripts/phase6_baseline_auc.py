"""フェーズ6: 対称ゲームHJI(フェーズ3)の勝敗予測力を単純なベースラインと比較する。

フェーズ3の AUC 0.745 はランダム(0.5)としか比較していなかった。本スクリプトでは
初期状態だけから計算できる単純な特徴量(距離・接近速度・等速仮定の最小距離など)と
比較し、HJI の価値関数 V に上乗せの予測力があるかを検証する。

【リーク対策】フェーズ3は V を「そのイベントの実際の継続時間」で評価していたが、
継続時間は事後にしかわからない量である。本スクリプトでは固定ホライズン(3秒)で
評価した V を主とし、継続時間を使う版(フェーズ3と同一)は参考値として併記する。

評価対象:
  - 全イベント(outcome ラベルあり)
  - 「デュエル」サブセット: 初期距離 8m 以下かつ守備者がゴール側にいる(goal_side_cos>0)
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.cache import CACHE_DIR, load_all_trajectories
from ad_drg.calibration import calibration_stats
from ad_drg.evaluation import (
    HORIZON,
    auc_with_ci,
    baseline_features,
    initial_state,
    lomo_cv_scores,
    paired_auc_diff,
)
from ad_drg.reachability import solve_brs

P_BOUND = 25.0
GRID_SHAPE = (31, 31, 21, 21)
N_TIME_STEPS = 21
ACCURACY = "medium"
DUEL_MAX_DIST = 8.0

# 単一特徴量AUCで「大きいほど captured 寄り」になるよう符号をそろえる(事前に決めた向き)
SINGLE_FEATURES = {
    "dist0": -1,
    "closing_speed": +1,
    "min_dist_cv": -1,
    "rel_speed": -1,
    "goal_side_cos": +1,
    "hji_V_fixed3s": -1,
    "hji_V_duration(leaky)": -1,
    "duration(leaky)": +1,
}
BASELINE_SET = ["dist0", "closing_speed", "min_dist_cv", "rel_speed", "atk_speed", "def_speed", "goal_side_cos", "dist_to_goal"]


def evaluate(rows: list[dict], label: str) -> dict:
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["match_id"] for r in rows])
    out = dict(label=label, n=len(rows), n_positive=int(y.sum()), base_rate=float(y.mean()), single={}, cv={})

    for feat, sign in SINGLE_FEATURES.items():
        s = sign * np.array([r[feat] for r in rows])
        out["single"][feat] = auc_with_ci(y, s)

    X_base = np.array([[r[f] for f in BASELINE_SET] for r in rows])
    X_hji = np.array([[r["hji_V_fixed3s"]] for r in rows])
    X_both = np.hstack([X_base, X_hji])
    oof_base = lomo_cv_scores(X_base, y, groups)
    oof_hji = lomo_cv_scores(X_hji, y, groups)
    oof_both = lomo_cv_scores(X_both, y, groups)
    out["cv"]["baseline"] = auc_with_ci(y, oof_base)
    out["cv"]["hji_only"] = auc_with_ci(y, oof_hji)
    out["cv"]["baseline+hji"] = auc_with_ci(y, oof_both)
    out["cv"]["diff(baseline+hji - baseline)"] = paired_auc_diff(y, oof_base, oof_both)
    out["cv"]["diff(hji_only - dist0)"] = paired_auc_diff(y, -X_base[:, 0], -X_hji[:, 0])
    return out


def print_eval(res: dict):
    print(f"\n=== {res['label']}: n={res['n']} 陽性={res['n_positive']} (base rate {res['base_rate']:.3f}) ===")
    for k, v in res["single"].items():
        print(f"  single {k:24s} AUC={v['auc']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    for k, v in res["cv"].items():
        if "diff" in k:
            print(f"  CV {k:34s} ΔAUC={v['diff']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}] p={v['p_boot']:.3f}")
        else:
            print(f"  CV {k:34s} AUC={v['auc']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")


def main():
    t0 = time.time()
    trajs = load_all_trajectories()
    calib = calibration_stats(trajs)
    a_max = calib["a_max_p95"]
    v_bound = 2.0 * calib["v_max_p95"]
    brs = solve_brs(a_max, a_max, P_BOUND, v_bound, GRID_SHAPE, HORIZON, N_TIME_STEPS, ACCURACY, progress_bar=False)
    np.save(CACHE_DIR / "brs_symmetric_values.npy", brs.values)
    print(f"BRS計算完了 elapsed={time.time() - t0:.1f}s")

    rows = []
    for traj in trajs:
        if traj.outcome not in ("captured", "evaded"):
            continue
        x0 = initial_state(traj)
        duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
        v_fixed = brs.value_at(x0, HORIZON)
        v_dur = brs.value_at(x0, min(duration, HORIZON))
        if np.isnan(v_fixed) or np.isnan(v_dur):
            continue
        rows.append(
            dict(
                match_id=traj.match_id,
                y=1 if traj.outcome == "captured" else 0,
                hji_V_fixed3s=v_fixed,
                **{"hji_V_duration(leaky)": v_dur, "duration(leaky)": float(duration)},
                **baseline_features(traj),
            )
        )

    duel = [r for r in rows if r["dist0"] <= DUEL_MAX_DIST and r["goal_side_cos"] > 0]
    res_all = evaluate(rows, "全イベント")
    res_duel = evaluate(duel, f"デュエル(距離≤{DUEL_MAX_DIST}m & 守備者ゴール側)")
    print_eval(res_all)
    print_eval(res_duel)

    out = dict(
        a_max=a_max,
        v_bound=v_bound,
        horizon=HORIZON,
        grid_shape=GRID_SHAPE,
        baseline_features=BASELINE_SET,
        all_events=res_all,
        duel_subset=res_duel,
    )
    path = "documents/phase6_baseline_auc_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {path} に保存しました (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
