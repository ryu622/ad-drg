"""フェーズ8b: 実効反応遅れ δ* の感度分析。

フェーズ8では「V_δ <= 0 と判定される割合 = 実際に抜かれた割合」となる δ* が 0.14〜0.20 s
だった。δ* はモデルの誤指定(運動能力の見積もりなど)を吸収しうる1スカラーなので、
a_max(分位点・攻守比)や τ を変えても δ* が大きく動かないかを確認する。
対象は s_pass=0、守備者ゴール側の全イベント。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.cache import load_all_trajectories
from ad_drg.calibration import calibration_stats, calibration_stats_by_role
from ad_drg.evaluation import HORIZON
from ad_drg.reach_avoid import GridSpec, solve_reach_avoid
from ad_drg.reaction_delay import delayed_values
from scripts.phase7_reach_avoid_auc import stopped_label
from scripts.phase8_reaction_delay import build_rows

S_PASS = 0.0
N_BOOT = 1000
DELAYS = np.round(np.arange(0.0, 0.61, 0.05), 2).tolist()


def matched_delay(ds, fr, target):
    for i in range(len(ds) - 1):
        if (fr[i] - target) * (fr[i + 1] - target) <= 0 and fr[i + 1] != fr[i]:
            return float(ds[i] + (target - fr[i]) * (ds[i + 1] - ds[i]) / (fr[i + 1] - fr[i]))
    return None


def main():
    t0 = time.time()
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    rows = build_rows(trajs)
    X0 = np.array([r["x0"] for r in rows])
    y = np.array([1 - stopped_label(r["traj"], S_PASS) for r in rows])

    base_a = calib["a_max_p95"]
    base_v = calib["v_max_p95"]
    variants = [
        dict(name="主設定 (p95, 攻守同等)", a_att=base_a, a_def=base_a, v_max=base_v),
        dict(name="a_max p50", a_att=calib["a_max_p50"], a_def=calib["a_max_p50"], v_max=base_v),
        dict(name="a_max p99", a_att=calib["a_max_p99"], a_def=calib["a_max_p99"], v_max=base_v),
        dict(name="v_max p50 (τ小)", a_att=base_a, a_def=base_a, v_max=calib["v_max_p50"]),
        dict(name="v_max p99 (τ大)", a_att=base_a, a_def=base_a, v_max=calib["v_max_p99"]),
        dict(name="守備者1.1倍", a_att=base_a, a_def=1.1 * base_a, v_max=base_v),
        dict(name="攻撃者1.1倍", a_att=1.1 * base_a, a_def=base_a, v_max=base_v),
        dict(name="守備者1.2倍", a_att=base_a, a_def=1.2 * base_a, v_max=base_v),
    ]
    by_role = calibration_stats_by_role(trajs)
    variants.insert(1, dict(name="役割別実測 (フェーズ4b)", a_att=by_role["attacker"]["a_max_p95"],
                            a_def=by_role["defender"]["a_max_p95"], v_max=base_v))
    out = []
    for var in variants:
        tau = var["v_max"] / max(var["a_att"], var["a_def"])
        ra = solve_reach_avoid(var["a_att"], var["a_def"], tau, GridSpec(v_bound=2.0 * base_v), HORIZON, s_pass=S_PASS)
        Vs = np.stack([delayed_values(ra, X0, d, HORIZON) for d in DELAYS])  # (n_delay, n_event)
        ok = ~np.any(np.isnan(Vs), axis=0)
        guaranteed = (Vs[:, ok] <= 0)
        yy = y[ok]
        fr = guaranteed.mean(axis=1).tolist()
        observed = float(yy.mean())
        dstar = matched_delay(DELAYS, fr, observed)
        # イベントのブートストラップで δ* の 95% CI
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(yy), len(yy))
            b = matched_delay(DELAYS, guaranteed[:, idx].mean(axis=1).tolist(), float(yy[idx].mean()))
            if b is not None:
                boots.append(b)
        ci = np.percentile(boots, [2.5, 97.5]).tolist() if boots else [None, None]
        out.append(dict(**var, tau=tau, frac_by_delay=dict(zip(map(str, DELAYS), fr)), observed=observed,
                        delay_star=dstar, delay_star_ci95=ci))
        print(f"{var['name']:22s} a_att={var['a_att']:.2f} a_def={var['a_def']:.2f} τ={tau:.2f}s  δ=0で{fr[0]:.3f}  "
              f"実測{observed:.3f}  → δ*={dstar:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  (elapsed {time.time() - t0:.0f}s)", flush=True)

    with open("documents/phase8b_delay_sensitivity_results.json", "w") as f:
        json.dump(dict(s_pass=S_PASS, delays=DELAYS, variants=out), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
