"""フェーズ1: AD モデル Baseline の再実装(idsse-data 版)。

Narizuka & Yamazaki (2026) の改良推定法(相手の実軌道を固定した独立最適化、
COBYLA、原著の探索範囲・制約)を idsse-data に対して再実装し、
    - 採択率(eps_a<0.1 かつ eps_d<0.1)が原著(31,028件中27,457件=88.5%)と
      同程度になるか
    - パラメータが探索範囲の境界に張り付く現象が再現されるか
を確認する(research_plan.md 4.3節 フェーズ1)。

計算コストの都合上、ローカルでは各試合からサンプリングしたイベントのみを対象とする
(N_INIT, N_SAMPLE_PER_MATCH は本スクリプト冒頭で調整可能)。全1,454件・原著同様の
N_INIT=100 での本番実行は Colab 側で行う想定(research_plan.md 4.5節)。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.ad_model import (
    BETA_BOUND,
    F_TAU_MAX,
    N_INIT_DEFAULT,
    TAU_MIN,
    fit_event,
)
from ad_drg.extraction import MATCH_IDS, build_trajectories

N_INIT = N_INIT_DEFAULT  # 論文はN=100。ローカル確認用に縮小。
N_SAMPLE_PER_MATCH = 10


def main():
    rng = np.random.default_rng(42)
    all_fits = []

    t0 = time.time()
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        sample_idx = rng.choice(len(trajs), size=min(N_SAMPLE_PER_MATCH, len(trajs)), replace=False)
        for i in sample_idx:
            fit = fit_event(trajs[i], n_init=N_INIT)
            all_fits.append(fit)
        print(
            f"[{match_id}] trajectories={len(trajs)} sampled={len(sample_idx)} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    n_events = len(all_fits)
    n_accurate = sum(f.both_accurate for f in all_fits)
    n_total_sides = n_events * 2
    n_at_bound = sum(f.attacker_fit["any_at_bound"] for f in all_fits) + sum(
        f.defender_fit["any_at_bound"] for f in all_fits
    )
    per_param_bound = {}
    for name in ("f", "tau", "beta"):
        cnt = sum(f.attacker_fit["at_bound"][name] for f in all_fits) + sum(
            f.defender_fit["at_bound"][name] for f in all_fits
        )
        per_param_bound[name] = cnt / n_total_sides

    errors_a = [f.attacker_fit["error"] for f in all_fits]
    errors_d = [f.defender_fit["error"] for f in all_fits]

    print("\n=== フェーズ1 AD モデル Baseline 再実装 結果 ===")
    print(f"サンプルイベント数: {n_events} (N_INIT={N_INIT}, N_SAMPLE_PER_MATCH={N_SAMPLE_PER_MATCH})")
    print(f"採択率 (eps_a<0.1 かつ eps_d<0.1): {n_accurate}/{n_events} = {n_accurate / n_events:.1%}")
    print(f"  (原著: 31,028件中27,457件 = 88.5%)")
    print(f"境界張り付き率(いずれかのパラメータ、原著の実制約 tau>=0.9, |beta|<=10, |f*tau|<=10.2 を使用): {n_at_bound / n_total_sides:.1%}")
    print(f"パラメータ別: {per_param_bound}")
    print(f"eps_a: mean={np.mean(errors_a):.3f} median={np.median(errors_a):.3f}")
    print(f"eps_d: mean={np.mean(errors_d):.3f} median={np.median(errors_d):.3f}")

    out = dict(
        n_init=N_INIT,
        n_sample_per_match=N_SAMPLE_PER_MATCH,
        constraints=dict(tau_min=TAU_MIN, beta_bound=BETA_BOUND, f_tau_max=F_TAU_MAX),
        n_events=n_events,
        n_accurate=n_accurate,
        accurate_rate=n_accurate / n_events,
        paper_accurate_rate=27457 / 31028,
        boundary_sticking_rate_overall=n_at_bound / n_total_sides,
        boundary_sticking_rate_per_param=per_param_bound,
        error_a_mean=float(np.mean(errors_a)),
        error_a_median=float(np.median(errors_a)),
        error_d_mean=float(np.mean(errors_d)),
        error_d_median=float(np.median(errors_d)),
        events=[
            dict(
                match_id=f.match_id,
                attacker=f.attacker,
                defender=f.defender,
                n_frames=f.n_frames,
                attacker_fit=f.attacker_fit,
                defender_fit=f.defender_fit,
            )
            for f in all_fits
        ],
    )
    out_path = "documents/phase1_ad_baseline_results.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")


if __name__ == "__main__":
    main()
