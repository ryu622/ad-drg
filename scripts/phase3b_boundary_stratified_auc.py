"""境界張り付きサブセットでのAUC層別比較(拡張サンプル版)。

前回70件(N_SAMPLE_PER_MATCH=10)でのAD境界フラグ×HJI捕捉可能性のAUC層別比較は、
非張り付き群がn=16(陽性5件)しかなく、統計的に判断できないという結論だった。
本スクリプトはAD再フィット(CPU律速、Colab不要)のサンプル数のみを約300件規模
(N_SAMPLE_PER_MATCH=45)に拡張し、同じ診断を再実行する。BRSはphase3_validation.py
と同一パラメータ(medium精度、31x31x21x21)で1回だけ計算し、全イベントで再利用する。

出力: documents/phase3b_boundary_stratified_auc_results.json
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from ad_drg.ad_model import fit_side
from ad_drg.calibration import calibration_stats
from ad_drg.extraction import MATCH_IDS, build_trajectories
from ad_drg.reachability import solve_brs

P_BOUND = 25.0
GRID_SHAPE = (31, 31, 21, 21)
TIME_HORIZON = 3.0
N_TIME_STEPS = 21
ACCURACY = "medium"

N_INIT_AD = 20
N_SAMPLE_PER_MATCH = 45
N_BOOTSTRAP = 2000
SEED = 123


def bootstrap_auc_ci(y_true, scores, n_boot=N_BOOTSTRAP, seed=SEED):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    if n == 0 or len(set(y_true.tolist())) < 2:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, sc = y_true[idx], scores[idx]
        if len(set(yt.tolist())) < 2:
            continue
        aucs.append(roc_auc_score(yt, sc))
    if len(aucs) < n_boot // 2:
        return None
    aucs = np.array(aucs)
    return dict(
        n_boot_valid=len(aucs),
        ci_low=float(np.percentile(aucs, 2.5)),
        ci_high=float(np.percentile(aucs, 97.5)),
        std=float(np.std(aucs)),
    )


def stratum_stats(rows):
    y_true = np.array([r["captured"] for r in rows])
    y_pred = np.array([r["predicted_capturable"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    n = len(rows)
    n_pos = int(y_true.sum())
    out = dict(n=n, n_positive=n_pos)
    if n == 0:
        return out
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    if len(set(y_true.tolist())) == 2:
        out["auc"] = float(roc_auc_score(y_true, scores))
        out["auc_bootstrap"] = bootstrap_auc_ci(y_true, scores)
    else:
        out["auc"] = None
        out["auc_bootstrap"] = None
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    all_trajectories = []
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
        print(f"[{match_id}] trajectories={len(trajs)} elapsed={time.time() - t0:.1f}s", flush=True)

    calib = calibration_stats(all_trajectories)
    a_max = calib["a_max_p95"]
    v_max = calib["v_max_p95"]
    v_bound = 2.0 * v_max
    print(f"a_max={a_max:.2f} m/s^2, v_max={v_max:.2f} m/s, v_bound={v_bound:.2f} m/s", flush=True)

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
    print(f"BRS計算 elapsed={time.time() - t1:.1f}s", flush=True)

    by_match = {}
    for traj in all_trajectories:
        if traj.outcome not in ("captured", "evaded"):
            continue
        by_match.setdefault(traj.match_id, []).append(traj)

    rows = []
    t2 = time.time()
    for match_id, trajs in by_match.items():
        idx = rng.choice(len(trajs), size=min(N_SAMPLE_PER_MATCH, len(trajs)), replace=False)
        for i in idx:
            traj = trajs[i]
            ad_a = fit_side(traj, "attacker", n_init=N_INIT_AD)
            ad_d = fit_side(traj, "defender", n_init=N_INIT_AD)
            any_at_bound = ad_a["any_at_bound"] or ad_d["any_at_bound"]

            dp0 = traj.p_a[0] - traj.p_d[0]
            dv0 = traj.v_a[0] - traj.v_d[0]
            x0 = np.concatenate([dp0, dv0])
            duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
            v = brs.value_at(x0, t=min(duration, TIME_HORIZON))
            if np.isnan(v):
                continue

            rows.append(
                dict(
                    match_id=match_id,
                    attacker=traj.attacker,
                    defender=traj.defender,
                    ad_any_at_bound=bool(any_at_bound),
                    ad_error_a=ad_a["error"],
                    ad_error_d=ad_d["error"],
                    captured=1 if traj.outcome == "captured" else 0,
                    predicted_capturable=1 if v <= 0 else 0,
                    score=float(-v),
                    init_dist=float(np.linalg.norm(dp0)),
                )
            )
        print(
            f"[{match_id}] sampled={len(idx)} total_rows={len(rows)} elapsed={time.time() - t2:.1f}s",
            flush=True,
        )

    boundary_rows = [r for r in rows if r["ad_any_at_bound"]]
    non_boundary_rows = [r for r in rows if not r["ad_any_at_bound"]]

    result = dict(
        n_init_ad=N_INIT_AD,
        n_sample_per_match=N_SAMPLE_PER_MATCH,
        n_total_valid=len(rows),
        overall=stratum_stats(rows),
        boundary=stratum_stats(boundary_rows),
        non_boundary=stratum_stats(non_boundary_rows),
        elapsed_sec=time.time() - t0,
        rows=rows,
    )

    print("\n=== 境界張り付きサブセットでのAUC層別比較(拡張サンプル) ===")
    print(f"全体 n={result['overall']['n']} AUC={result['overall'].get('auc')}")
    print(f"境界張り付き群 n={result['boundary']['n']} (陽性{result['boundary']['n_positive']}) "
          f"AUC={result['boundary'].get('auc')} CI={result['boundary'].get('auc_bootstrap')}")
    print(f"非張り付き群 n={result['non_boundary']['n']} (陽性{result['non_boundary']['n_positive']}) "
          f"AUC={result['non_boundary'].get('auc')} CI={result['non_boundary'].get('auc_bootstrap')}")
    print(f"総経過時間={result['elapsed_sec']:.1f}s")

    out_path = "documents/phase3b_boundary_stratified_auc_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")


if __name__ == "__main__":
    main()
