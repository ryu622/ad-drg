"""フェーズ7: 「守備者を抜く」reach-avoid ゲームの価値関数による勝敗予測。

ad_drg/reach_avoid.py の価値関数 V(x0, 3s) を予測スコアとし、2種類の勝敗ラベルで
単純なベースライン(フェーズ6と同じ特徴量 + ゴール座標系での相対状態)と比較する。

ラベル:
  y_captured : 次にボールを持ったのが守備側チームなら1(フェーズ3以来の近似ラベル)
  y_stopped  : 最初の3秒間(セグメントがそれより短ければその間)に攻撃者が
               ゴール方向 s で守備者より s_pass [m] 以上前に出なければ1(抜かれなかった)。
               トラッキングデータから直接判定する新ラベル。セグメントが短いまま終わった
               (パス・ロストなど)場合も「抜かれていない」に数える点に注意。

対象: 初期時刻で守備者がゴール側(s0 < 0)にいるイベント。デュエル(初期距離 8m 以下)は別集計。
感度分析として (s_pass, r_capture) を変えた複数設定で解く(ラベルの s_pass も合わせる)。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.cache import load_all_trajectories
from ad_drg.calibration import calibration_stats
from ad_drg.evaluation import (
    HORIZON,
    auc_with_ci,
    baseline_features,
    goal_axis,
    lomo_cv_scores,
    paired_auc_diff,
    rotate_to_goal_frame,
)
from ad_drg.reach_avoid import GridSpec, solve_reach_avoid
from ad_drg.reachability import solve_brs

DUEL_MAX_DIST = 8.0
SETTINGS = [
    dict(s_pass=1.0, r_capture=1.5),  # 主設定
    dict(s_pass=0.0, r_capture=1.5),
    dict(s_pass=1.0, r_capture=1.0),
]
BASELINE_SET = [
    "dist0", "closing_speed", "min_dist_cv", "rel_speed", "atk_speed", "def_speed",
    "goal_side_cos", "dist_to_goal", "s0", "l0_abs", "vs0", "vl0_abs",
]


def goal_frame_state(traj, k: int = 0) -> np.ndarray:
    e_g = goal_axis(traj)
    dp = rotate_to_goal_frame(traj.p_a[k] - traj.p_d[k], e_g)
    dv = rotate_to_goal_frame(traj.v_a[k] - traj.v_d[k], e_g)
    return np.concatenate([dp, dv])


def stopped_label(traj, s_pass: float) -> int:
    e_g = goal_axis(traj)
    n = min(len(traj.p_a), int(round(HORIZON / traj.dt)) + 1)
    s = (traj.p_a[:n] - traj.p_d[:n]) @ e_g
    return int(not np.any(s >= s_pass))


def evaluate(rows, label_key, score_key, name):
    y = np.array([r[label_key] for r in rows])
    groups = np.array([r["match_id"] for r in rows])
    X_base = np.array([[r[f] for f in BASELINE_SET] for r in rows])
    v = np.array([[r[score_key]] for r in rows])
    v_sym = np.array([[r["V_sym"]] for r in rows])
    oof_base = lomo_cv_scores(X_base, y, groups)
    oof_both = lomo_cv_scores(np.hstack([X_base, v]), y, groups)
    res = dict(
        name=name,
        label=label_key,
        n=len(rows),
        n_positive=int(y.sum()),
        base_rate=float(y.mean()),
        single=dict(
            V_reach_avoid=auc_with_ci(y, v[:, 0]),
            V_symmetric_pe=auc_with_ci(y, -v_sym[:, 0]),
            dist0=auc_with_ci(y, -X_base[:, 0]),
        ),
        cv=dict(
            baseline=auc_with_ci(y, oof_base),
            baseline_plus_V=auc_with_ci(y, oof_both),
            diff_V_minus_dist0=paired_auc_diff(y, -X_base[:, 0], v[:, 0]),
            diff_baseline_plus_V_minus_baseline=paired_auc_diff(y, oof_base, oof_both),
        ),
    )
    s = res["single"]
    c = res["cv"]
    print(
        f"  [{name} / {label_key}] n={res['n']} pos={res['n_positive']} "
        f"| V_RA {s['V_reach_avoid']['auc']:.3f} [{s['V_reach_avoid']['ci_lo']:.3f},{s['V_reach_avoid']['ci_hi']:.3f}] "
        f"| V_PE {s['V_symmetric_pe']['auc']:.3f} | dist0 {s['dist0']['auc']:.3f} "
        f"| ΔV-dist0 {c['diff_V_minus_dist0']['diff']:+.3f} (p={c['diff_V_minus_dist0']['p_boot']:.3f}) "
        f"| CV base {c['baseline']['auc']:.3f} base+V {c['baseline_plus_V']['auc']:.3f} "
        f"Δ {c['diff_baseline_plus_V_minus_baseline']['diff']:+.3f} "
        f"[{c['diff_baseline_plus_V_minus_baseline']['ci_lo']:+.3f},{c['diff_baseline_plus_V_minus_baseline']['ci_hi']:+.3f}]",
        flush=True,
    )
    return res


def main():
    t0 = time.time()
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    a_max = calib["a_max_p95"]
    v_max = calib["v_max_p95"]
    tau = v_max / a_max
    grid_spec = GridSpec(v_bound=2.0 * v_max)
    print(f"a_max={a_max:.2f}, v_max={v_max:.2f}, tau={tau:.3f}s")

    brs_sym = solve_brs(a_max, a_max, 25.0, 2.0 * v_max, (31, 31, 21, 21), HORIZON, 21, "medium", progress_bar=False)

    base_rows = []
    for traj in trajs:
        if traj.outcome not in ("captured", "evaded"):
            continue
        x0 = goal_frame_state(traj)
        if x0[0] >= 0:  # 守備者がゴール側にいないイベントは対象外
            continue
        feats = baseline_features(traj)
        pitch_state = np.concatenate([traj.p_a[0] - traj.p_d[0], traj.v_a[0] - traj.v_d[0]])
        base_rows.append(
            dict(
                traj=traj,
                x0=x0,
                match_id=traj.match_id,
                y_captured=1 if traj.outcome == "captured" else 0,
                V_sym=brs_sym.value_at(pitch_state, HORIZON),
                s0=float(x0[0]),
                l0_abs=float(abs(x0[1])),
                vs0=float(x0[2]),
                vl0_abs=float(abs(x0[3])),
                **feats,
            )
        )
    print(f"守備者ゴール側のイベント: {len(base_rows)}")

    results = []
    for setting in SETTINGS:
        t1 = time.time()
        ra = solve_reach_avoid(a_max, a_max, tau, grid_spec, HORIZON, **setting)
        V = ra.values_at(np.array([r["x0"] for r in base_rows]), HORIZON)
        rows = []
        for r, v in zip(base_rows, V):
            if np.isnan(v) or np.isnan(r["V_sym"]):
                continue
            rows.append({**{k: val for k, val in r.items() if k not in ("traj", "x0")},
                         "V_RA": float(v), "y_stopped": stopped_label(r["traj"], setting["s_pass"])})
        print(f"\n設定 {setting} (solve {time.time() - t1:.1f}s, 有効 {len(rows)}件, "
              f"V<=0 の割合 {np.mean([r['V_RA'] <= 0 for r in rows]):.3f})")
        duel = [r for r in rows if r["dist0"] <= DUEL_MAX_DIST]
        setting_res = dict(setting=setting, n_valid=len(rows),
                           frac_V_nonpositive=float(np.mean([r["V_RA"] <= 0 for r in rows])), evals=[])
        for subset_name, subset in (("goal-side all", rows), ("duel<=8m", duel)):
            for label in ("y_stopped", "y_captured"):
                setting_res["evals"].append(evaluate(subset, label, "V_RA", subset_name))
        # ラベル同士の関係
        ys = np.array([r["y_stopped"] for r in rows])
        yc = np.array([r["y_captured"] for r in rows])
        setting_res["label_crosstab"] = dict(
            stopped_and_captured=int(np.sum((ys == 1) & (yc == 1))),
            stopped_not_captured=int(np.sum((ys == 1) & (yc == 0))),
            beaten_and_captured=int(np.sum((ys == 0) & (yc == 1))),
            beaten_not_captured=int(np.sum((ys == 0) & (yc == 0))),
        )
        print("  ラベルのクロス集計:", setting_res["label_crosstab"])
        results.append(setting_res)

    out = dict(a_max=a_max, v_max=v_max, tau=tau, horizon=HORIZON, grid_spec=grid_spec.__dict__,
               baseline_features=BASELINE_SET, results=results)
    path = "documents/phase7_reach_avoid_auc_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {path} に保存しました (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
