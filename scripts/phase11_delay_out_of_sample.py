"""フェーズ11: 反応遅れ δ* の試合外検証(層別の抜かれ率を1パラメータで予測できるか)。

フェーズ8の δ* は「全体の抜かれ率に一致させた」値で、単調な曲線なので必ずどこかで一致する。
それ自体は証拠にならない。本スクリプトでは δ を1つに固定したまま、別のもの
(局面タイプごとの抜かれ率の違い)を試合外で予測できるかを検証する。

手順(leave-one-match-out):
  1. 6試合で δ* を決める(「V_δ <= 0 の割合 = 実際の抜かれ率」となる δ)
  2. 残り1試合の各イベントについて V_{δ*} <= 0(確実に抜ける)を予測する
  3. 7試合分の予測を集め、層(下記)ごとの予測率と実測率を比べる

層: 初期状態の4変数(初期距離、ゴール方向の接近速度 vs0、横ずれ |l0|、攻撃者の速さ)を
それぞれ三分位で3層に分けた計12層(ラベルは使わずに境界を決める)。

比較する予測(いずれも 6 試合で当てはめ → 残り1試合で予測):
  M_delay : V_{δ*} <= 0 の指示関数(自由パラメータは δ の1つ)
  N0_const: 全層に同じ率(6試合の全体の抜かれ率)。層による違いを何も説明しない
  N1_scaled: 遅れなし V_0 <= 0 の指示関数を、全体の率が合うよう定数倍したもの
            (遅れが「層ごとの違いの形」を変えて説明しているのか、単に水準を上げているだけかを分ける)
  N2_logit: 初期状態12特徴量のロジスティック回帰の予測確率(柔軟な上限の参考)

【事前に決めた判定基準】(実行前に記述)
  (i)  層ごとの予測率と実測率の誤差(層の件数で重みづけした平均絶対誤差, MAE)で
       M_delay が N0_const と N1_scaled の両方を下回る
  (ii) 層ごとに個別に当てはめた δ* の95%CIが、全体の δ* とおおむね重なる
       (12層中、全体 δ* を含まない層が 2 層以下)
  両方を満たせば「1つの遅れで局面ごとの違いまで説明できる」と主張してよい。
  (i) を満たさなければ、δ* は全体の率を合わせるだけの1パラメータと結論づける。

運動能力はフェーズ8bの役割別実測値(攻撃者 a_max p95 / 守備者 a_max p95)を主設定とし、
攻守同等の設定を副として併記する。s_pass = 0(同じ高さまで出たら抜いた)。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.cache import load_all_trajectories
from ad_drg.calibration import calibration_stats, calibration_stats_by_role
from ad_drg.evaluation import HORIZON, lomo_cv_scores
from ad_drg.reach_avoid import GridSpec, solve_reach_avoid
from ad_drg.reaction_delay import delayed_values
from scripts.phase7_reach_avoid_auc import BASELINE_SET, stopped_label
from scripts.phase8_reaction_delay import build_rows
from scripts.phase8b_delay_sensitivity import matched_delay

S_PASS = 0.0
DELAYS = np.round(np.arange(0.0, 0.6001, 0.025), 3).tolist()
STRATA_VARS = {"dist0": "初期距離", "vs0": "接近速度(ゴール方向)", "l0_abs": "横ずれ|l0|", "atk_speed": "攻撃者の速さ"}
N_BOOT = 1000


def fit_delay(guaranteed: np.ndarray, y: np.ndarray) -> float | None:
    """guaranteed: (n_delay, n) の V_δ<=0 指示関数。全体の率が一致する δ を返す。"""
    return matched_delay(DELAYS, guaranteed.mean(axis=1).tolist(), float(y.mean()))


def indicator_at(guaranteed: np.ndarray, delay: float) -> np.ndarray:
    """δ グリッド間は線形補間(確率的な指示関数として扱う)。"""
    d = np.array(DELAYS)
    i = int(np.clip(np.searchsorted(d, delay) - 1, 0, len(d) - 2))
    w = (delay - d[i]) / (d[i + 1] - d[i])
    return (1 - w) * guaranteed[i] + w * guaranteed[i + 1]


def strata_labels(rows) -> dict:
    out = {}
    for var in STRATA_VARS:
        x = np.array([r[var] for r in rows])
        q = np.percentile(x, [100 / 3, 200 / 3])
        out[var] = (np.digitize(x, q), q.tolist())
    return out


def run(variant_name, a_att, a_def, v_max, rows, y, groups, X_base, strata):
    t0 = time.time()
    tau = v_max / max(a_att, a_def)
    ra = solve_reach_avoid(a_att, a_def, tau, GridSpec(v_bound=2.0 * v_max), HORIZON, s_pass=S_PASS)
    X0 = np.array([r["x0"] for r in rows])
    Vs = np.stack([delayed_values(ra, X0, d, HORIZON) for d in DELAYS])
    guaranteed = (Vs <= 0).astype(float)
    ok = ~np.any(np.isnan(Vs), axis=0)
    assert ok.all(), "グリッド外のイベントが含まれている"

    pred = {k: np.full(len(y), np.nan) for k in ("M_delay", "N0_const", "N1_scaled")}
    fold_delays = {}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        d_star = fit_delay(guaranteed[:, tr], y[tr])
        fold_delays[g] = d_star
        pred["M_delay"][te] = indicator_at(guaranteed[:, te], d_star)
        pred["N0_const"][te] = y[tr].mean()
        g0_tr = guaranteed[0, tr].mean()
        pred["N1_scaled"][te] = guaranteed[0, te] * (y[tr].mean() / g0_tr)
    pred["N2_logit"] = lomo_cv_scores(X_base, y, groups)

    # 層ごとの比較
    table = []
    for var, (lab, cuts) in strata.items():
        for s in range(3):
            m = lab == s
            row = dict(var=var, stratum=s, n=int(m.sum()), observed=float(y[m].mean()))
            for k, p in pred.items():
                row[k] = float(p[m].mean())
            table.append(row)
    n_tot = sum(r["n"] for r in table)
    mae = {k: float(sum(r["n"] * abs(r[k] - r["observed"]) for r in table) / n_tot) for k in pred}
    corr = {k: float(np.corrcoef([r[k] for r in table], [r["observed"] for r in table])[0, 1]) for k in pred}

    # 層ごとに個別に当てはめた δ*(全データ)とブートストラップCI
    d_all = fit_delay(guaranteed, y)
    rng = np.random.default_rng(0)
    per_stratum_delay = []
    for var, (lab, _) in strata.items():
        for s in range(3):
            idx_s = np.nonzero(lab == s)[0]
            d_s = fit_delay(guaranteed[:, idx_s], y[idx_s])
            boots = []
            for _ in range(N_BOOT):
                b = rng.choice(idx_s, len(idx_s))
                v = fit_delay(guaranteed[:, b], y[b])
                if v is not None:
                    boots.append(v)
            ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(boots) > N_BOOT * 0.5 else [None, None]
            per_stratum_delay.append(dict(var=var, stratum=s, n=len(idx_s), delay=d_s, ci95=ci,
                                          n_boot_valid=len(boots)))
    n_excl = sum(1 for r in per_stratum_delay
                 if r["ci95"][0] is None or not (r["ci95"][0] <= d_all <= r["ci95"][1]))

    crit_i = mae["M_delay"] < mae["N0_const"] and mae["M_delay"] < mae["N1_scaled"]
    crit_ii = n_excl <= 2

    print(f"\n===== {variant_name} (a_att={a_att:.2f}, a_def={a_def:.2f}, τ={tau:.2f}s, {time.time() - t0:.0f}s) =====")
    print(f"試合ごとの δ*(他6試合で当てはめ): " + ", ".join(f"{g}:{d:.3f}" for g, d in fold_delays.items()))
    print(f"{'層':28s} {'n':>4s} {'実測':>6s} {'M_delay':>8s} {'N0':>6s} {'N1':>6s} {'N2':>6s}")
    for r in table:
        name = f"{STRATA_VARS[r['var']]} T{r['stratum'] + 1}"
        print(f"{name:26s} {r['n']:4d} {r['observed']:6.3f} {r['M_delay']:8.3f} {r['N0_const']:6.3f} "
              f"{r['N1_scaled']:6.3f} {r['N2_logit']:6.3f}")
    print("層ごとの MAE:", {k: round(v, 4) for k, v in mae.items()})
    print("層間の相関(予測 vs 実測):", {k: round(v, 3) for k, v in corr.items()})
    print(f"全体 δ* = {d_all:.3f}")
    for r in per_stratum_delay:
        ci = r["ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "[CI不定]"
        d_s = f"{r['delay']:.3f}" if r["delay"] is not None else "なし"
        print(f"  {STRATA_VARS[r['var']]} T{r['stratum'] + 1}: δ*={d_s} {ci_s}")
    print(f"判定 (i) MAE で N0・N1 を下回る: {crit_i} / (ii) 全体 δ* を含まない層 {n_excl} 個 (<=2 で合格): {crit_ii}")

    return dict(variant=variant_name, a_att=a_att, a_def=a_def, tau=tau, fold_delays=fold_delays,
                strata_table=table, mae=mae, corr_across_strata=corr, delay_all=d_all,
                per_stratum_delay=per_stratum_delay, n_strata_excluding_overall_delay=n_excl,
                criterion_i=crit_i, criterion_ii=crit_ii)


def main():
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    by_role = calibration_stats_by_role(trajs)
    rows = build_rows(trajs)
    # 価値関数のグリッド範囲内の初期状態だけを対象にする(フェーズ7・8で NaN として除外していたものと同じ)
    gs, vb = GridSpec(), 2.0 * calib["v_max_p95"]
    rows = [r for r in rows if gs.s_lo <= r["x0"][0] <= gs.s_hi and abs(r["x0"][1]) <= gs.l_bound
            and np.all(np.abs(r["x0"][2:]) <= vb)]
    y = np.array([1 - stopped_label(r["traj"], S_PASS) for r in rows])
    groups = np.array([r["match_id"] for r in rows])
    X_base = np.array([[r[f] for f in BASELINE_SET] for r in rows])
    strata = strata_labels(rows)
    print(f"対象 {len(rows)} 件、抜かれた {int(y.sum())} 件 ({y.mean():.3f})")
    print("三分位の境界:", {k: [round(c, 2) for c in v[1]] for k, v in strata.items()})

    results = [
        run("役割別実測", by_role["attacker"]["a_max_p95"], by_role["defender"]["a_max_p95"], calib["v_max_p95"],
            rows, y, groups, X_base, strata),
        run("攻守同等", calib["a_max_p95"], calib["a_max_p95"], calib["v_max_p95"], rows, y, groups, X_base, strata),
    ]
    out = dict(s_pass=S_PASS, delays=DELAYS, strata_cuts={k: v[1] for k, v in strata.items()},
               n=len(rows), n_beaten=int(y.sum()), results=results)
    with open("documents/phase11_delay_out_of_sample_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
