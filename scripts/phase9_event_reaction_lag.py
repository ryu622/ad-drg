"""フェーズ9: イベントごとの守備者の反応遅れをトラッキングから推定し、勝敗との関係を調べる。

フェーズ8で、一律の反応遅れ δ は「抜かれる割合」の水準は説明できる(δ* ≈ 0.2 s)が、
どのイベントで抜かれるかの順位付け(AUC)は改善しなかった。騙し合いを捉えるには
イベントごとに「守備者がどれだけ遅れて反応したか」を測る必要がある。

推定法: ゴール座標系(s: ゴール方向、l: 横方向)で、攻撃者の横速度 va_l(t) に
守備者の横速度 vd_l(t + lag) がどれだけ追随しているかを、lag = 0〜0.8 s の相互相関の
最大値で求める(1対1で守備者が相手の横移動に合わせる「ミラーリング」の遅れ)。
    推定区間: 0 〜 min(最初に抜かれた時刻, セグメント終了, 3 s)
    対象: 推定区間 >= 1.0 s、攻撃者の横速度の標準偏差 >= 0.5 m/s、最大相関 >= 0.3

注意: 推定区間はイベントの途中まで(抜かれる直前まで)の軌道を使うため、この分析は
事前予測ではなく「抜かれた局面を事後的に説明できるか」の検証である。

分析:
  (a) 推定遅れの分布と、フェーズ8の δ* ≈ 0.2 s との整合
  (b) 推定遅れ・追随相関が、初期状態ベースラインを超えて「抜かれた」を説明するか
  (c) イベント固有の遅れ δ_i を入れた V_{δ_i} は V_0 より「抜かれた」を説明するか
  (d) 守備者ごとの平均遅れの信頼性(奇数番目/偶数番目イベントでの折半相関)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from ad_drg.cache import load_all_trajectories
from ad_drg.calibration import calibration_stats
from ad_drg.evaluation import HORIZON, auc_with_ci, goal_axis, lomo_cv_scores, paired_auc_diff, rotate_to_goal_frame
from ad_drg.reach_avoid import GridSpec, solve_reach_avoid
from ad_drg.reaction_delay import delayed_values
from scripts.phase7_reach_avoid_auc import BASELINE_SET
from scripts.phase8_reaction_delay import build_rows

S_PASS = 0.0
MAX_LAG = 0.8  # s
MIN_WINDOW = 1.0  # s
MIN_LATERAL_STD = 0.5  # m/s
MIN_CORR = 0.3
MIN_EVENTS_PER_DEFENDER = 6


def estimate_lag(traj) -> dict | None:
    e_g = goal_axis(traj)
    dp = rotate_to_goal_frame(traj.p_a - traj.p_d, e_g)
    va = rotate_to_goal_frame(traj.v_a, e_g)
    vd = rotate_to_goal_frame(traj.v_d, e_g)
    n_max = min(len(traj.p_a), int(round(HORIZON / traj.dt)) + 1)
    beaten_idx = np.nonzero(dp[:n_max, 0] >= S_PASS)[0]
    n = int(beaten_idx[0]) if len(beaten_idx) else n_max
    if n * traj.dt < MIN_WINDOW:
        return None
    a_l, d_l = va[:n, 1], vd[:n, 1]
    if a_l.std() < MIN_LATERAL_STD:
        return None
    max_k = int(round(MAX_LAG / traj.dt))
    corrs = []
    for k in range(max_k + 1):
        x, z = a_l[: n - k], d_l[k:n]
        if len(x) < 10 or x.std() < 1e-6 or z.std() < 1e-6:
            corrs.append(np.nan)
            continue
        corrs.append(float(np.corrcoef(x, z)[0, 1]))
    corrs = np.array(corrs)
    if np.all(np.isnan(corrs)):
        return None
    k_best = int(np.nanargmax(corrs))
    if corrs[k_best] < MIN_CORR:
        return None
    return dict(lag=k_best * traj.dt, corr=float(corrs[k_best]), corr_at_0=float(corrs[0]), window=n * traj.dt,
                lag_at_edge=bool(k_best == max_k))


def main():
    t0 = time.time()
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    a_max, v_max = calib["a_max_p95"], calib["v_max_p95"]
    rows_all = build_rows(trajs)

    n_screened = defaultdict(int)
    rows = []
    for r in rows_all:
        est = estimate_lag(r["traj"])
        if est is None:
            n_screened["excluded"] += 1
            continue
        e_g = goal_axis(r["traj"])
        s = (r["traj"].p_a - r["traj"].p_d)[: int(round(HORIZON / r["traj"].dt)) + 1] @ e_g
        rows.append({**r, **est, "y_beaten": int(np.any(s >= S_PASS))})
    print(f"推定対象 {len(rows)} / {len(rows_all)} 件 (除外 {n_screened['excluded']})")

    lags = np.array([r["lag"] for r in rows])
    y = np.array([r["y_beaten"] for r in rows])
    edge = np.mean([r["lag_at_edge"] for r in rows])
    summary_a = dict(
        n=len(rows),
        lag_median=float(np.median(lags)),
        lag_q25=float(np.percentile(lags, 25)),
        lag_q75=float(np.percentile(lags, 75)),
        lag_mean=float(lags.mean()),
        frac_lag_at_upper_edge=float(edge),
        corr_median=float(np.median([r["corr"] for r in rows])),
        lag_median_beaten=float(np.median(lags[y == 1])) if y.sum() else None,
        lag_median_not_beaten=float(np.median(lags[y == 0])),
        mannwhitney_p=float(mannwhitneyu(lags[y == 1], lags[y == 0]).pvalue) if y.sum() else None,
        n_beaten=int(y.sum()),
    )
    print("(a) 推定遅れ:", json.dumps(summary_a, ensure_ascii=False))

    # (b) ベースライン + 遅れ/相関
    groups = np.array([r["match_id"] for r in rows])
    X_base = np.array([[r[f] for f in BASELINE_SET] for r in rows])
    X_lag = np.array([[r["lag"], r["corr"]] for r in rows])
    oof_base = lomo_cv_scores(X_base, y, groups)
    oof_both = lomo_cv_scores(np.hstack([X_base, X_lag]), y, groups)
    summary_b = dict(
        single_lag=auc_with_ci(y, lags),
        single_corr_neg=auc_with_ci(y, -X_lag[:, 1]),
        cv_baseline=auc_with_ci(y, oof_base),
        cv_baseline_plus_lag=auc_with_ci(y, oof_both),
        cv_diff=paired_auc_diff(y, oof_base, oof_both),
    )
    print("(b)", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in summary_b.items()}, ensure_ascii=False))

    # (c) イベント固有の遅れを入れた V
    ra = solve_reach_avoid(a_max, a_max, v_max / a_max, GridSpec(v_bound=2.0 * v_max), HORIZON, s_pass=S_PASS)
    X0 = np.array([r["x0"] for r in rows])
    V0 = delayed_values(ra, X0, 0.0, HORIZON)
    lag_grid = np.round(np.arange(0.0, MAX_LAG + 1e-9, 0.04), 2)
    V_by_lag = {d: delayed_values(ra, X0, float(d), HORIZON) for d in lag_grid}
    V_event = np.array([V_by_lag[np.round(l, 2)][i] if np.round(l, 2) in V_by_lag else np.nan for i, l in enumerate(lags)])
    ok = ~np.isnan(V0) & ~np.isnan(V_event)
    V_const = V_by_lag[np.round(0.2, 2)]
    ok &= ~np.isnan(V_const)
    summary_c = dict(
        n=int(ok.sum()),
        auc_V0=auc_with_ci(y[ok], -V0[ok]),
        auc_V_const_0p2=auc_with_ci(y[ok], -V_const[ok]),
        auc_V_event_lag=auc_with_ci(y[ok], -V_event[ok]),
        diff_event_minus_V0=paired_auc_diff(y[ok], -V0[ok], -V_event[ok]),
        frac_guaranteed_V0=float(np.mean(V0[ok] <= 0)),
        frac_guaranteed_event=float(np.mean(V_event[ok] <= 0)),
        observed_beaten=float(y[ok].mean()),
    )
    print("(c)", json.dumps(summary_c, ensure_ascii=False))

    # (d) 守備者ごとの平均遅れの折半信頼性
    by_def = defaultdict(list)
    for r in rows:
        by_def[r["defender"]].append(r["lag"])
    odd, even = [], []
    for d, ls in by_def.items():
        if len(ls) >= MIN_EVENTS_PER_DEFENDER:
            odd.append(np.mean(ls[0::2]))
            even.append(np.mean(ls[1::2]))
    rho = spearmanr(odd, even) if len(odd) >= 5 else None
    summary_d = dict(
        n_defenders=len(odd),
        split_half_spearman=float(rho.statistic) if rho else None,
        split_half_p=float(rho.pvalue) if rho else None,
    )
    print("(d)", json.dumps(summary_d, ensure_ascii=False))

    out = dict(
        s_pass=S_PASS, max_lag=MAX_LAG, min_window=MIN_WINDOW, min_lateral_std=MIN_LATERAL_STD, min_corr=MIN_CORR,
        a_lag_distribution=summary_a, b_explanatory_auc=summary_b, c_event_delay_value=summary_c,
        d_defender_reliability=summary_d,
        lag_histogram=dict(zip(*[x.tolist() for x in np.unique(np.round(lags, 2), return_counts=True)])),
    )
    out["lag_histogram"] = {str(k): v for k, v in out["lag_histogram"].items()}
    with open("documents/phase9_event_reaction_lag_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"保存しました (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
