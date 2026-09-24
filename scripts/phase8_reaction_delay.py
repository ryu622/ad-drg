"""フェーズ8: 守備者の反応遅れ δ を入れた reach-avoid 価値による勝敗予測。

ad_drg/reaction_delay.py の V_δ を δ ∈ {0, 0.1, ..., 1.0} s で計算し、
  (a) 「理論上確実に抜ける」(V_δ <= 0) と判定される割合が、実際に抜かれた割合と
      一致する δ(= データが示唆する実効的な反応遅れ)
  (b) δ ごとの予測力(単体AUC、ベースラインへの上乗せ、V_δ<=0 の適合率・再現率)
を調べる。遅れなし(δ=0)では確実に抜ける状態がほとんどない(フェーズ7: 1〜3%)のに
対し、実際には 3〜10% が抜かれている。この差を反応遅れで説明できるかを見る。

ラベル y_beaten = 1 - y_stopped(フェーズ7と同じ判定、3秒以内に s >= s_pass)。
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import precision_score, recall_score

from ad_drg.cache import load_all_trajectories
from ad_drg.calibration import calibration_stats
from ad_drg.evaluation import HORIZON, auc_with_ci, baseline_features, lomo_cv_scores, paired_auc_diff
from ad_drg.reach_avoid import GridSpec, solve_reach_avoid
from ad_drg.reaction_delay import delayed_values
from scripts.phase7_reach_avoid_auc import BASELINE_SET, DUEL_MAX_DIST, goal_frame_state, stopped_label

DELAYS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
S_PASS_LIST = [0.0, 1.0]
R_CAPTURE = 1.5


def build_rows(trajs):
    rows = []
    for traj in trajs:
        if traj.outcome not in ("captured", "evaded"):
            continue
        x0 = goal_frame_state(traj)
        if x0[0] >= 0:
            continue
        rows.append(
            dict(
                traj=traj,
                x0=x0,
                match_id=traj.match_id,
                attacker=traj.attacker,
                defender=traj.defender,
                s0=float(x0[0]),
                l0_abs=float(abs(x0[1])),
                vs0=float(x0[2]),
                vl0_abs=float(abs(x0[3])),
                **baseline_features(traj),
            )
        )
    return rows


def analyse(rows, V_by_delay, s_pass, subset_name):
    y = np.array([1 - stopped_label(r["traj"], s_pass) for r in rows])  # 1 = 抜かれた
    groups = np.array([r["match_id"] for r in rows])
    X_base = np.array([[r[f] for f in BASELINE_SET] for r in rows])
    oof_base = lomo_cv_scores(X_base, y, groups)
    observed_rate = float(y.mean())
    print(f"\n--- s_pass={s_pass} / {subset_name}: n={len(y)}, 実際に抜かれた割合 {observed_rate:.3f} "
          f"(CV baseline AUC {auc_with_ci(y, oof_base)['auc']:.3f}) ---")
    per_delay = []
    for d, V in V_by_delay.items():
        pred = (V <= 0).astype(int)
        oof_both = lomo_cv_scores(np.hstack([X_base, V[:, None]]), y, groups)
        single = auc_with_ci(y, -V)  # V が小さいほど攻撃者有利 = 抜かれやすい
        diff = paired_auc_diff(y, oof_base, oof_both)
        res = dict(
            delay=d,
            frac_guaranteed_pass=float(pred.mean()),
            precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)),
            single_auc=single,
            cv_base_plus_V=auc_with_ci(y, oof_both),
            cv_diff=diff,
        )
        per_delay.append(res)
        print(f"  δ={d:.1f}s  確実に抜ける判定 {res['frac_guaranteed_pass']:.3f}  "
              f"適合率 {res['precision']:.3f} 再現率 {res['recall']:.3f}  "
              f"単体AUC {single['auc']:.3f} [{single['ci_lo']:.3f},{single['ci_hi']:.3f}]  "
              f"base+V {res['cv_base_plus_V']['auc']:.3f} (Δ {diff['diff']:+.3f} [{diff['ci_lo']:+.3f},{diff['ci_hi']:+.3f}])",
              flush=True)

    # (a) 判定割合が実際の割合と一致する δ を線形補間で求める
    ds = np.array([p["delay"] for p in per_delay])
    fr = np.array([p["frac_guaranteed_pass"] for p in per_delay])
    matched = None
    for i in range(len(ds) - 1):
        if (fr[i] - observed_rate) * (fr[i + 1] - observed_rate) <= 0 and fr[i + 1] != fr[i]:
            matched = float(ds[i] + (observed_rate - fr[i]) * (ds[i + 1] - ds[i]) / (fr[i + 1] - fr[i]))
            break
    best = max(per_delay, key=lambda p: p["single_auc"]["auc"])
    print(f"  → 判定割合が実測と一致する δ ≈ {matched}  / 単体AUC最大の δ = {best['delay']}")
    return dict(
        s_pass=s_pass,
        subset=subset_name,
        n=len(y),
        observed_beaten_rate=observed_rate,
        cv_baseline=auc_with_ci(y, oof_base),
        per_delay=per_delay,
        delay_matching_rate=matched,
        delay_best_single_auc=best["delay"],
    )


def main():
    t0 = time.time()
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    a_max, v_max = calib["a_max_p95"], calib["v_max_p95"]
    tau = v_max / a_max
    grid_spec = GridSpec(v_bound=2.0 * v_max)
    rows_all = build_rows(trajs)

    results = []
    per_event = {}
    for s_pass in S_PASS_LIST:
        ra = solve_reach_avoid(a_max, a_max, tau, grid_spec, HORIZON, s_pass=s_pass, r_capture=R_CAPTURE)
        X0 = np.array([r["x0"] for r in rows_all])
        V_all = {d: delayed_values(ra, X0, d, HORIZON) for d in DELAYS}
        valid = ~np.any(np.isnan(np.stack(list(V_all.values()))), axis=0)
        rows = [r for r, ok in zip(rows_all, valid) if ok]
        V_valid = {d: V[valid] for d, V in V_all.items()}
        # V_δ <= V_0 になっているかの確認(一度きりの遅れモデルなので厳密には保証されない)
        viol = float(np.mean(V_valid[0.3] > V_valid[0.0] + 1e-3))
        print(f"\n[s_pass={s_pass}] 有効 {len(rows)}件, V_0.3 > V_0 となる割合 {viol:.3f}, elapsed {time.time() - t0:.1f}s")
        results.append(analyse(rows, V_valid, s_pass, "goal-side all"))
        duel_mask = np.array([r["dist0"] <= DUEL_MAX_DIST for r in rows])
        duel_rows = [r for r, m in zip(rows, duel_mask) if m]
        results.append(analyse(duel_rows, {d: V[duel_mask] for d, V in V_valid.items()}, s_pass, "duel<=8m"))
        per_event[s_pass] = dict(
            match_id=[r["match_id"] for r in rows],
            attacker=[r["attacker"] for r in rows],
            defender=[r["defender"] for r in rows],
            y_beaten=[1 - stopped_label(r["traj"], s_pass) for r in rows],
            V={str(d): V.tolist() for d, V in V_valid.items()},
        )

    out = dict(a_max=a_max, v_max=v_max, tau=tau, horizon=HORIZON, r_capture=R_CAPTURE,
               delays=DELAYS, results=results)
    with open("documents/phase8_reaction_delay_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open("documents/phase8_reaction_delay_per_event.json", "w") as f:
        json.dump(per_event, f, ensure_ascii=False)
    print(f"\n保存しました (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
