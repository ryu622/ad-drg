"""フェーズ4: 準最適性指標の実装・分析(research_plan.md 5節④)。

    Suboptimality(x0) = V_observed(x0) - V*(x0)

  V*(x0):        HJIの価値関数(理論的に最適な攻守が互いに最善を尽くした場合の
                  保証値。V*<=0 なら「守備者は時間T以内の捕捉を保証できる」)。
  V_observed(x0): 実際に観測された軌道上で守備者が最も接近できた瞬間のコスト
                  min_t ||p_a(t)-p_d(t)|| - r_capture。理論的な最適解ではなく、
                  実際に起きたことをV*と同じ土俵(距離-r_captureのコスト単位)で
                  表したもの。

Suboptimality > 0: V*<=0(理論上捕捉可能)だったのに実際には capture できなかった
                    → 守備側の非最適性(理論的に保証された機会を活かせなかった)
Suboptimality < 0: V*>0(理論上は捕捉を保証できない)のに実際には capture できた
                    → 攻撃側の非最適性(最悪ケースで回避しなかった)、または
                      運動制約(a_max等)のキャリブレーション誤りの可能性

フェーズ3②の混同行列(TP/FP/FN/TN)は outcome ラベル(次セグメントの保持チームに
よるプロキシ)に基づく二値判定だったが、本フェーズでは outcome ラベルを介さず、
軌道そのものから連続的な準最適性の大きさを計算する。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.calibration import calibration_stats
from ad_drg.extraction import MATCH_IDS, build_trajectories
from ad_drg.reachability import DEFAULT_R_CAPTURE, solve_brs

P_BOUND = 25.0
GRID_SHAPE = (31, 31, 21, 21)
TIME_HORIZON = 3.0
N_TIME_STEPS = 21
ACCURACY = "medium"


def v_observed_from(t_idx: int, p_a: np.ndarray, p_d: np.ndarray, r_capture: float) -> float:
    """時刻インデックスt_idx以降で実際に達成された最小距離ベースのコスト。"""
    dist = np.linalg.norm(p_a[t_idx:] - p_d[t_idx:], axis=1)
    return float(dist.min() - r_capture)


def suboptimality_curve(traj, brs, r_capture: float) -> dict:
    duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
    v_star, v_obs, subopt = [], [], []
    for k in range(len(traj.t)):
        dp = traj.p_a[k] - traj.p_d[k]
        dv = traj.v_a[k] - traj.v_d[k]
        x = np.concatenate([dp, dv])
        remaining = np.clip(duration - traj.t[k], 0.0, TIME_HORIZON)
        vs = brs.value_at(x, t=remaining)
        vo = v_observed_from(k, traj.p_a, traj.p_d, r_capture)
        v_star.append(vs)
        v_obs.append(vo)
        subopt.append(vo - vs if not np.isnan(vs) else np.nan)
    return dict(t=traj.t.tolist(), v_star=v_star, v_observed=v_obs, suboptimality=subopt)


def main():
    t0 = time.time()
    all_trajectories = []
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
        print(f"[{match_id}] trajectories={len(trajs)} elapsed={time.time() - t0:.1f}s", flush=True)

    calib = calibration_stats(all_trajectories)
    a_max = calib["a_max_p95"]
    v_max = calib["v_max_p95"]
    v_bound = 2.0 * v_max
    r_capture = DEFAULT_R_CAPTURE

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
    print(f"BRS計算完了 a_max={a_max:.2f} v_bound={v_bound:.2f}")

    print("\n=== 全イベントの初期状態でのSuboptimality ===")
    rows = []
    for traj in all_trajectories:
        duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
        dp0 = traj.p_a[0] - traj.p_d[0]
        dv0 = traj.v_a[0] - traj.v_d[0]
        x0 = np.concatenate([dp0, dv0])
        v_star = brs.value_at(x0, t=min(duration, TIME_HORIZON))
        if np.isnan(v_star):
            continue
        v_obs = v_observed_from(0, traj.p_a, traj.p_d, r_capture)
        rows.append(
            dict(
                match_id=traj.match_id,
                attacker=traj.attacker,
                defender=traj.defender,
                dist0=float(np.linalg.norm(dp0)),
                duration=float(duration),
                v_star=float(v_star),
                v_observed=float(v_obs),
                suboptimality=float(v_obs - v_star),
            )
        )

    subopt = np.array([r["suboptimality"] for r in rows])
    v_star_arr = np.array([r["v_star"] for r in rows])
    v_obs_arr = np.array([r["v_observed"] for r in rows])

    defender_subopt = [r for r in rows if r["v_star"] <= 0 and r["v_observed"] > 0]
    attacker_subopt = [r for r in rows if r["v_star"] > 0 and r["v_observed"] <= 0]
    both_consistent = [r for r in rows if (r["v_star"] <= 0) == (r["v_observed"] <= 0)]

    print(f"有効イベント数: {len(rows)}")
    print(f"Suboptimality: mean={subopt.mean():.3f} median={np.median(subopt):.3f} std={subopt.std():.3f}")
    print(f"V*<=0 の割合(理論上捕捉可能): {(v_star_arr <= 0).mean():.1%}")
    print(f"V_observed<=0 の割合(実際に最接近時にr_capture以内): {(v_obs_arr <= 0).mean():.1%}")
    print(f"\n守備側の非最適性(V*<=0だが実際には捕捉できず): {len(defender_subopt)}件 ({len(defender_subopt) / len(rows):.1%})")
    print(f"攻撃側の非最適性/制約誤り(V*>0だが実際には捕捉): {len(attacker_subopt)}件 ({len(attacker_subopt) / len(rows):.1%})")
    print(f"理論と実際が整合: {len(both_consistent)}件 ({len(both_consistent) / len(rows):.1%})")

    # 代表的な数事例でSuboptimality(t)の時系列を可視化
    print("\n=== 代表事例のSuboptimality(t)時系列を計算 ===")
    examples = {}
    if defender_subopt:
        ex = max(defender_subopt, key=lambda r: r["suboptimality"])
        traj = next(t for t in all_trajectories if t.match_id == ex["match_id"] and t.attacker == ex["attacker"] and t.defender == ex["defender"])
        examples["defender_suboptimal"] = suboptimality_curve(traj, brs, r_capture)
    if attacker_subopt:
        ex = min(attacker_subopt, key=lambda r: r["suboptimality"])
        traj = next(t for t in all_trajectories if t.match_id == ex["match_id"] and t.attacker == ex["attacker"] and t.defender == ex["defender"])
        examples["attacker_suboptimal"] = suboptimality_curve(traj, brs, r_capture)

    out = dict(
        a_max=a_max,
        v_max=v_max,
        v_bound=v_bound,
        r_capture=r_capture,
        n_events=len(rows),
        suboptimality_mean=float(subopt.mean()),
        suboptimality_median=float(np.median(subopt)),
        suboptimality_std=float(subopt.std()),
        frac_v_star_capturable=float((v_star_arr <= 0).mean()),
        frac_v_observed_captured=float((v_obs_arr <= 0).mean()),
        n_defender_suboptimal=len(defender_subopt),
        n_attacker_suboptimal=len(attacker_subopt),
        n_consistent=len(both_consistent),
        rows=rows,
        examples=examples,
    )
    out_path = "documents/phase4_suboptimality_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました")

    # 可視化
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(subopt, bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Suboptimality = V_observed - V*")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Suboptimality distribution (n={len(rows)})")

    for label, curve in examples.items():
        axes[1].plot(curve["t"], curve["v_star"], "--", label=f"{label}: V*")
        axes[1].plot(curve["t"], curve["v_observed"], "-", label=f"{label}: V_observed")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("value")
    axes[1].set_title("Example event(s): V*(t) vs V_observed(t)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig_path = "documents/phase4_suboptimality.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"可視化を {fig_path} に保存しました")


if __name__ == "__main__":
    main()
