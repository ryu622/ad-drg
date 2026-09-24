"""フェーズ10: reach-avoid 価値関数に基づく瞬間ごとの準最適性分解とフェイント検出。

HJ 理論では、両者が最適に動くとき価値 V は軌道に沿って変化しない。実際の軌道で V が
変化した分は、各選手が最適からどれだけずれたかに分解できる。

    dv' = u_a - u_d - dv/τ,  u_i = v_i' + v_i/τ(観測加速度から逆算、||u_i|| <= a_max に射影)
    p = ∇_v V(x)(価値関数の速度成分の勾配)

    攻撃者(min側)の損失率  L_a = p·u_a + a_max ||p||  ∈ [0, 2 a_max ||p||]  (最適なら 0)
    守備者(max側)の損失率  L_d = a_max ||p|| + p·u_d  ∈ [0, 2 a_max ||p||]  (最適なら 0)

L_d が大きい = 守備者が攻撃者に V を「譲った」(抜かれる方向に状況を悪化させた)。
L は ||p|| に比例するので、状態によってスケールが変わる(V の勾配が大きい局面ほど損失も
大きく出る)。選手の振る舞いそのものを見るため、最大損失で割った正規化損失
    ℓ_i = L_i / (2 a_max ||p||) ∈ [0, 1]
を主指標とする(0 = 最適方向に全力、0.5 = 無関係な方向、1 = 最悪方向に全力)。
V は残り3秒の価値を各フレームで使う(ローリングホライズン近似)。

【初回版からの修正】初回版は「抜かれた時刻までの区間」で損失を積算していたため、
区間長(抜かれたら打ち切り)と勾配スケールの交絡で「抜かれた局面ほど損失が小さい」
という見かけの関係が出た。本版では打ち切りの影響を受けない予測設計に変更した:

  (A) 観測区間 [0, W=1.0 s] の振る舞い(正規化損失・フェイント候補)で、
      その後 (W, 3 s] に抜かれるかを予測する。対象は W 時点でまだ抜かれておらず、
      セグメントが W 以上続いたイベント。ベースラインは W 時点の状態から作る
      (状態を条件づけた上で、振る舞いに追加の情報があるかの検定)。
  (B) 選手ごとの平均正規化損失の折半信頼性(選手評価指標として安定か)。
  (C) フェイント候補(攻撃者の ℓ_a が高い直後 0.5 s 以内に守備者の ℓ_d が高くなる)。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

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
from scripts.phase7_reach_avoid_auc import BASELINE_SET
from scripts.phase8_reaction_delay import build_rows

S_PASS = 0.0
OBS_WINDOW = 1.0  # s
FEINT_WINDOW = 0.5  # s
HIGH_LOSS = 0.75  # 正規化損失がこれを超えたら「最適から大きく外れた」とみなす
MIN_GRAD = 1e-3
MIN_EVENTS_PER_PLAYER = 6


def project_ball(u: np.ndarray, r: float) -> np.ndarray:
    n = np.linalg.norm(u, axis=-1, keepdims=True)
    return np.where(n > r, u * (r / np.maximum(n, 1e-9)), u)


def feint_flag(ell_a: np.ndarray, ell_d: np.ndarray, dt: float) -> int:
    k = int(round(FEINT_WINDOW / dt))
    hi_a = np.nonzero(ell_a > HIGH_LOSS)[0]
    hi_d = ell_d > HIGH_LOSS
    return int(any(hi_d[i + 1: i + 1 + k].any() for i in hi_a))


def split_half(rows, who, metric):
    by = defaultdict(list)
    for r in rows:
        if not np.isnan(r[metric]):
            by[r[who]].append(r[metric])
    odd, even = [], []
    for ls in by.values():
        if len(ls) >= MIN_EVENTS_PER_PLAYER:
            odd.append(np.mean(ls[0::2]))
            even.append(np.mean(ls[1::2]))
    rho = spearmanr(odd, even)
    return dict(n_players=len(odd), split_half_spearman=float(rho.statistic), p=float(rho.pvalue))


def main():
    t0 = time.time()
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    a_max, v_max = calib["a_max_p95"], calib["v_max_p95"]
    tau = v_max / a_max
    ra = solve_reach_avoid(a_max, a_max, tau, GridSpec(v_bound=2.0 * v_max), HORIZON, s_pass=S_PASS)
    V_grid = jnp.asarray(ra.values[ra._time_index(HORIZON)])
    grad_grid = ra.grid.grad_values(V_grid)
    interp_V = jax.jit(jax.vmap(lambda x: ra.grid.interpolate(V_grid, x)))
    interp_g = jax.jit(jax.vmap(lambda x: ra.grid.interpolate(grad_grid, x)))
    print(f"価値関数と勾配の準備 {time.time() - t0:.1f}s")

    events = []
    for r in build_rows(trajs):
        traj = r["traj"]
        e_g = goal_axis(traj)
        n_max = min(len(traj.p_a), int(round(HORIZON / traj.dt)) + 1)
        dp = rotate_to_goal_frame(traj.p_a - traj.p_d, e_g)[:n_max]
        va = rotate_to_goal_frame(traj.v_a, e_g)
        vd = rotate_to_goal_frame(traj.v_d, e_g)
        u_a = project_ball(np.gradient(va, traj.dt, axis=0) + va / tau, a_max)[:n_max]
        u_d = project_ball(np.gradient(vd, traj.dt, axis=0) + vd / tau, a_max)[:n_max]
        x = np.concatenate([dp, (va - vd)[:n_max]], axis=1)
        V = np.asarray(interp_V(jnp.asarray(x)))
        g = np.asarray(interp_g(jnp.asarray(x)))
        if np.any(np.isnan(V)) or np.any(np.isnan(g)):
            continue
        p = g[:, 2:]
        pn = np.linalg.norm(p, axis=1)
        L_a = np.einsum("ij,ij->i", p, u_a) + a_max * pn
        L_d = a_max * pn + np.einsum("ij,ij->i", p, u_d)
        valid = pn > MIN_GRAD
        ell_a = np.where(valid, L_a / np.maximum(2 * a_max * pn, 1e-12), np.nan)
        ell_d = np.where(valid, L_d / np.maximum(2 * a_max * pn, 1e-12), np.nan)
        beaten_idx = np.nonzero(dp[:, 0] >= S_PASS)[0]
        events.append(dict(row=r, traj=traj, n_max=n_max, V=V, L_a=L_a, L_d=L_d, ell_a=ell_a, ell_d=ell_d,
                           beaten_at=int(beaten_idx[0]) if len(beaten_idx) else None))
    print(f"損失を計算したイベント {len(events)} 件")

    # (A) 予測設計
    rows_A = []
    for ev in events:
        traj = ev["traj"]
        k_w = int(round(OBS_WINDOW / traj.dt))
        if ev["n_max"] <= k_w + 1:
            continue  # セグメントが W 秒より短い
        if ev["beaten_at"] is not None and ev["beaten_at"] <= k_w:
            continue  # W 時点で既に抜かれている
        ell_a, ell_d = ev["ell_a"][:k_w + 1], ev["ell_d"][:k_w + 1]
        if np.all(np.isnan(ell_a)) or np.all(np.isnan(ell_d)):
            continue
        e_g = goal_axis(traj)
        xw = np.concatenate([rotate_to_goal_frame(traj.p_a[k_w] - traj.p_d[k_w], e_g),
                             rotate_to_goal_frame(traj.v_a[k_w] - traj.v_d[k_w], e_g)])
        rows_A.append(dict(
            match_id=ev["row"]["match_id"],
            y=int(ev["beaten_at"] is not None),
            **baseline_features(traj, k_w),
            s0=float(xw[0]), l0_abs=float(abs(xw[1])), vs0=float(xw[2]), vl0_abs=float(abs(xw[3])),
            V_w=float(ev["V"][k_w]),
            ell_a=float(np.nanmean(ell_a)),
            ell_d=float(np.nanmean(ell_d)),
            ell_d_max=float(np.nanmax(ell_d)),
            L_d_int=float(np.nansum(ev["L_d"][:k_w + 1]) * traj.dt),
            dV_window=float(ev["V"][k_w] - ev["V"][0]),
            feint=feint_flag(ell_a, ell_d, traj.dt),
        ))
    y = np.array([r["y"] for r in rows_A])
    groups = np.array([r["match_id"] for r in rows_A])
    X_base = np.array([[r[f] for f in BASELINE_SET] for r in rows_A])
    X_V = np.array([[r["V_w"]] for r in rows_A])

    def cv(cols):
        X = np.hstack([X_base] + [np.array([[r[c] for c in cols] for r in rows_A])]) if cols else X_base
        return lomo_cv_scores(X, y, groups)

    oof_base = cv([])
    variants = {
        "+V_w": ["V_w"],
        "+ell_d": ["ell_d"],
        "+ell_a": ["ell_a"],
        "+ell_a+ell_d": ["ell_a", "ell_d"],
        "+ell_d_max": ["ell_d_max"],
        "+feint": ["feint"],
        "+dV_window": ["dV_window"],
        "+V_w+ell_a+ell_d+feint": ["V_w", "ell_a", "ell_d", "feint"],
    }
    res_A = dict(n=len(rows_A), n_beaten_after_W=int(y.sum()), cv_baseline=auc_with_ci(y, oof_base), variants={},
                 single={f: auc_with_ci(y, np.array([r[f] for r in rows_A])) for f in ("ell_a", "ell_d", "ell_d_max", "feint", "dV_window")})
    res_A["single"]["-V_w"] = auc_with_ci(y, -X_V[:, 0])
    print(f"\n(A) 観測 {OBS_WINDOW}s → その後に抜かれるか: n={res_A['n']} 抜かれた={res_A['n_beaten_after_W']}  "
          f"CV baseline(W時点の状態) AUC {res_A['cv_baseline']['auc']:.3f}")
    for f, v in res_A["single"].items():
        print(f"    single {f:10s} AUC {v['auc']:.3f} [{v['ci_lo']:.3f},{v['ci_hi']:.3f}]")
    for name, cols in variants.items():
        oof = cv(cols)
        d = paired_auc_diff(y, oof_base, oof)
        res_A["variants"][name] = dict(auc=auc_with_ci(y, oof), diff=d)
        print(f"    baseline{name:24s} AUC {res_A['variants'][name]['auc']['auc']:.3f}  "
              f"Δ {d['diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] p={d['p_boot']:.3f}")

    # (B) 選手ごとの信頼性(抜かれるまで or 3秒までの全区間で平均)
    rows_B = []
    for ev in events:
        end = ev["beaten_at"] if ev["beaten_at"] is not None else ev["n_max"]
        if end < 5:
            continue
        rows_B.append(dict(
            attacker=ev["row"]["attacker"], defender=ev["row"]["defender"],
            ell_a=float(np.nanmean(ev["ell_a"][:end])) if not np.all(np.isnan(ev["ell_a"][:end])) else np.nan,
            ell_d=float(np.nanmean(ev["ell_d"][:end])) if not np.all(np.isnan(ev["ell_d"][:end])) else np.nan,
            L_d_rate=float(np.mean(ev["L_d"][:end])),
            L_a_rate=float(np.mean(ev["L_a"][:end])),
        ))
    res_B = dict(
        defender_ell_d=split_half(rows_B, "defender", "ell_d"),
        attacker_ell_a=split_half(rows_B, "attacker", "ell_a"),
        defender_L_d_rate_unnormalized=split_half(rows_B, "defender", "L_d_rate"),
        attacker_L_a_rate_unnormalized=split_half(rows_B, "attacker", "L_a_rate"),
        mean_ell_a=float(np.nanmean([r["ell_a"] for r in rows_B])),
        mean_ell_d=float(np.nanmean([r["ell_d"] for r in rows_B])),
    )
    print("\n(B) 選手ごとの折半信頼性:", json.dumps(res_B, ensure_ascii=False))

    out = dict(s_pass=S_PASS, a_max=a_max, tau=tau, obs_window=OBS_WINDOW, feint_window=FEINT_WINDOW,
               high_loss=HIGH_LOSS, A_prediction=res_A, B_reliability=res_B)
    with open("documents/phase10_instantaneous_suboptimality_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"保存しました (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
