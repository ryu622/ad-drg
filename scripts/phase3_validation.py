"""フェーズ3: 実データでの検証(research_plan.md 5節②③)。

② 捕捉可能集合の予測精度(主結果):
    HJIの捕捉可能集合による予測(初期状態が捕捉可能集合に入っているか)と、
    実際のドリブル結果(守備成功/失敗)を混同行列で照合し、適合率・再現率・AUCで評価する。

③ AD モデルとの直接比較:
    AD モデルの再現誤差が大きい(または境界に張り付いた)事例において、HJIの
    最適軌道(相手の実軌道を固定した独立シミュレーション、ad_drg/hji_trajectory.py)が
    AD モデルより実軌道を良く説明できるかを、誤差分布のWilcoxon符号順位検定で比較する。

【outcome ラベルの限界】守備成功/失敗のラベルは、このセグメントの直後にボールを
持つチームで近似したプロキシであり(ad_drg/extraction.py参照)、正式なタックル/
インターセプトのイベントラベルではない。ファウルやアウトオブプレーを挟むケースなどで
誤ラベルが混入し得る点に注意。
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from ad_drg.ad_model import fit_side
from ad_drg.calibration import calibration_stats
from ad_drg.extraction import MATCH_IDS, build_trajectories
from ad_drg.hji_trajectory import fit_event_hji
from ad_drg.reachability import DEFAULT_R_CAPTURE, solve_brs

P_BOUND = 25.0
GRID_SHAPE = (31, 31, 21, 21)
TIME_HORIZON = 3.0
N_TIME_STEPS = 21
ACCURACY = "medium"

N_INIT_AD = 20
N_SAMPLE_PER_MATCH = 10


def section2_prediction_accuracy(all_trajectories, brs):
    """5節② 捕捉可能集合の予測精度。"""
    y_true, y_pred, scores, dists = [], [], [], []
    for traj in all_trajectories:
        if traj.outcome not in ("captured", "evaded"):
            continue
        dp0 = traj.p_a[0] - traj.p_d[0]
        dv0 = traj.v_a[0] - traj.v_d[0]
        x0 = np.concatenate([dp0, dv0])
        duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
        v = brs.value_at(x0, t=min(duration, TIME_HORIZON))
        if np.isnan(v):
            continue
        y_true.append(1 if traj.outcome == "captured" else 0)
        y_pred.append(1 if v <= 0 else 0)
        scores.append(-v)  # 大きいほど「捕捉可能」寄り
        dists.append(float(np.linalg.norm(dp0)))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, scores) if len(set(y_true.tolist())) == 2 else float("nan")

    return dict(
        n_events=len(y_true),
        n_positive=int(y_true.sum()),
        confusion_matrix=dict(tp=tp, fp=fp, fn=fn, tn=tn),
        precision=float(precision),
        recall=float(recall),
        auc=float(auc),
        base_rate=float(y_true.mean()),
    )


def section3_ad_vs_hji(all_trajectories, brs, rng):
    """5節③ AD モデルとの直接比較。"""
    by_match = {}
    for traj in all_trajectories:
        by_match.setdefault(traj.match_id, []).append(traj)

    rows = []
    t0 = time.time()
    for match_id, trajs in by_match.items():
        idx = rng.choice(len(trajs), size=min(N_SAMPLE_PER_MATCH, len(trajs)), replace=False)
        for i in idx:
            traj = trajs[i]
            ad_a = fit_side(traj, "attacker", n_init=N_INIT_AD)
            ad_d = fit_side(traj, "defender", n_init=N_INIT_AD)
            hji = fit_event_hji(traj, brs)
            rows.append(
                dict(
                    match_id=match_id,
                    attacker=traj.attacker,
                    defender=traj.defender,
                    n_frames=len(traj.p_a),
                    ad_error_a=ad_a["error"],
                    ad_error_d=ad_d["error"],
                    ad_any_at_bound=ad_a["any_at_bound"] or ad_d["any_at_bound"],
                    hji_error_a=hji["error_a"],
                    hji_error_d=hji["error_d"],
                )
            )
        print(f"[{match_id}] sampled={len(idx)} elapsed={time.time() - t0:.1f}s", flush=True)

    valid = [
        r
        for r in rows
        if not (np.isnan(r["hji_error_a"]) or np.isnan(r["hji_error_d"]))
    ]

    def pooled(rs):
        ad = [r["ad_error_a"] for r in rs] + [r["ad_error_d"] for r in rs]
        hji = [r["hji_error_a"] for r in rs] + [r["hji_error_d"] for r in rs]
        return np.array(ad), np.array(hji)

    ad_all, hji_all = pooled(valid)
    overall_test = wilcoxon(ad_all, hji_all) if len(ad_all) > 0 else None

    boundary_subset = [r for r in valid if r["ad_any_at_bound"]]
    ad_b, hji_b = pooled(boundary_subset)
    boundary_test = wilcoxon(ad_b, hji_b) if len(ad_b) >= 1 else None

    return dict(
        n_sampled=len(rows),
        n_valid_hji=len(valid),
        n_nan_hji_excluded=len(rows) - len(valid),
        n_boundary_subset_events=len(boundary_subset),
        overall_ad_error_mean=float(np.mean(ad_all)) if len(ad_all) else None,
        overall_hji_error_mean=float(np.mean(hji_all)) if len(hji_all) else None,
        overall_wilcoxon_statistic=float(overall_test.statistic) if overall_test else None,
        overall_wilcoxon_pvalue=float(overall_test.pvalue) if overall_test else None,
        boundary_ad_error_mean=float(np.mean(ad_b)) if len(ad_b) else None,
        boundary_hji_error_mean=float(np.mean(hji_b)) if len(hji_b) else None,
        boundary_wilcoxon_statistic=float(boundary_test.statistic) if boundary_test else None,
        boundary_wilcoxon_pvalue=float(boundary_test.pvalue) if boundary_test else None,
        rows=rows,
    )


def main():
    t0 = time.time()
    all_trajectories = []
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
        print(f"[{match_id}] trajectories={len(trajs)} elapsed={time.time() - t0:.1f}s", flush=True)

    n_captured = sum(1 for t in all_trajectories if t.outcome == "captured")
    n_evaded = sum(1 for t in all_trajectories if t.outcome == "evaded")
    print(f"\noutcome内訳: captured={n_captured}, evaded={n_evaded}, unknown={len(all_trajectories) - n_captured - n_evaded}")

    calib = calibration_stats(all_trajectories)
    a_max = calib["a_max_p95"]
    v_max = calib["v_max_p95"]
    v_bound = 2.0 * v_max
    print(f"a_max={a_max:.2f} m/s^2, v_max={v_max:.2f} m/s, v_bound={v_bound:.2f} m/s")

    t1 = time.time()
    brs = solve_brs(
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
    print(f"BRS計算 elapsed={time.time() - t1:.1f}s")

    print("\n=== 5節② 捕捉可能集合の予測精度 ===")
    sec2 = section2_prediction_accuracy(all_trajectories, brs)
    print(json.dumps(sec2, indent=2, ensure_ascii=False))

    print("\n=== 5節③ AD モデルとの直接比較 ===")
    rng = np.random.default_rng(123)
    sec3 = section3_ad_vs_hji(all_trajectories, brs, rng)
    print(f"サンプル数: {sec3['n_sampled']} (有効: {sec3['n_valid_hji']}, HJI発散除外: {sec3['n_nan_hji_excluded']})")
    print(f"全体: AD誤差平均={sec3['overall_ad_error_mean']:.3f}  HJI誤差平均={sec3['overall_hji_error_mean']:.3f}")
    print(f"  Wilcoxon統計量={sec3['overall_wilcoxon_statistic']:.1f}  p値={sec3['overall_wilcoxon_pvalue']:.4f}")
    print(f"境界張り付きサブセット(n={sec3['n_boundary_subset_events']}事例):")
    print(f"  AD誤差平均={sec3['boundary_ad_error_mean']}  HJI誤差平均={sec3['boundary_hji_error_mean']}")
    print(f"  Wilcoxon統計量={sec3['boundary_wilcoxon_statistic']}  p値={sec3['boundary_wilcoxon_pvalue']}")

    out = dict(
        n_trajectories=len(all_trajectories),
        n_captured=n_captured,
        n_evaded=n_evaded,
        calibration=calib,
        a_max=a_max,
        v_max=v_max,
        v_bound=v_bound,
        section2=sec2,
        section3=sec3,
    )
    out_path = "documents/phase3_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")


if __name__ == "__main__":
    main()
