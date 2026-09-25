"""フェーズ12: 「正面で抜かれた」事例の軌道の可視化と診断。

フェーズ11で、攻撃者が守備者の正面にいる局面(横ずれ |l0| の下位三分位)は、
モデルでは「ほぼ抜けない」(2.9%)のに、実際には最も多く抜かれていた(15.1%)。
これがラベルの誤判定(接触・交錯で s=0 を横切っただけ)か、本当の突破かを確かめる。

図: ゴール座標系(初期時刻の攻撃者位置からゴール方向を上)で、攻撃者(赤)・守備者(青)の
軌道を描く。丸が開始点、× が s=0 を横切った(抜いたと判定された)時刻の位置。
3秒以降の軌道は薄く描く。

診断指標(抜いたと判定された時刻 t_b について):
  gap_at_beat   : t_b での横方向の間隔 |l|(小さいほど「すれ違い・交錯」寄り)
  min_dist      : t_b までの最小距離
  max_s_after   : t_b 以降(3秒まで)に攻撃者がゴール方向に守備者より何 m 前に出たか
  time_after    : t_b からセグメント終了(ボール保持が終わる)までの時間
  outcome       : 次にボールを持ったチーム(captured = 守備側が奪った)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ad_drg.cache import load_all_trajectories
from ad_drg.calibration import calibration_stats
from ad_drg.evaluation import HORIZON, goal_axis, rotate_to_goal_frame
from ad_drg.reach_avoid import GridSpec
from scripts.phase8_reaction_delay import build_rows

S_PASS = 0.0
N_PLOT = 24
plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]


def diagnose(traj):
    e_g = goal_axis(traj)
    origin = traj.p_a[0]
    pa = rotate_to_goal_frame(traj.p_a - origin, e_g)
    pd = rotate_to_goal_frame(traj.p_d - origin, e_g)
    rel = pa - pd
    n3 = min(len(pa), int(round(HORIZON / traj.dt)) + 1)
    beat = np.nonzero(rel[:n3, 0] >= S_PASS)[0]
    if not len(beat):
        return None
    kb = int(beat[0])
    dist = np.linalg.norm(rel, axis=1)
    return dict(
        pa=pa, pd=pd, kb=kb, n3=n3,
        t_beat=kb * traj.dt,
        gap_at_beat=float(abs(rel[kb, 1])),
        dist_at_beat=float(dist[kb]),
        min_dist=float(dist[: kb + 1].min()),
        max_s_after=float(rel[kb:n3, 0].max()),
        time_after=float((len(pa) - 1 - kb) * traj.dt),
        outcome=traj.outcome,
    )


def main():
    trajs = load_all_trajectories(verbose=False)
    calib = calibration_stats(trajs)
    rows = build_rows(trajs)
    gs, vb = GridSpec(), 2.0 * calib["v_max_p95"]
    rows = [r for r in rows if gs.s_lo <= r["x0"][0] <= gs.s_hi and abs(r["x0"][1]) <= gs.l_bound
            and np.all(np.abs(r["x0"][2:]) <= vb)]
    l_cut = np.percentile([r["l0_abs"] for r in rows], [100 / 3, 200 / 3])

    groups = {"正面 (|l0| 下位1/3)": [], "横ずれ (|l0| 上位1/3)": []}
    for r in rows:
        d = diagnose(r["traj"])
        if d is None:
            continue
        d.update(match_id=r["match_id"], dist0=r["dist0"], l0_abs=r["l0_abs"], s0=r["s0"])
        if r["l0_abs"] <= l_cut[0]:
            groups["正面 (|l0| 下位1/3)"].append(d)
        elif r["l0_abs"] > l_cut[1]:
            groups["横ずれ (|l0| 上位1/3)"].append(d)

    # 診断指標の比較
    summary = {}
    keys = ["dist0", "t_beat", "gap_at_beat", "min_dist", "max_s_after", "time_after"]
    for name, evs in groups.items():
        s = {k: dict(median=float(np.median([e[k] for e in evs])),
                     q25=float(np.percentile([e[k] for e in evs], 25)),
                     q75=float(np.percentile([e[k] for e in evs], 75))) for k in keys}
        s["n"] = len(evs)
        s["frac_captured"] = float(np.mean([e["outcome"] == "captured" for e in evs]))
        s["frac_gap_lt_1m"] = float(np.mean([e["gap_at_beat"] < 1.0 for e in evs]))
        s["frac_min_dist_lt_1m"] = float(np.mean([e["min_dist"] < 1.0 for e in evs]))
        s["frac_max_s_after_lt_1m"] = float(np.mean([e["max_s_after"] < 1.0 for e in evs]))
        s["frac_segment_ends_within_0p5s"] = float(np.mean([e["time_after"] < 0.5 for e in evs]))
        # 「交錯・すれ違い」寄りの判定: 抜いた瞬間の横間隔 < 1m かつ その後 1m 未満しか前に出ない
        s["frac_crossing_like"] = float(np.mean([(e["gap_at_beat"] < 1.0) and (e["max_s_after"] < 1.0) for e in evs]))
        summary[name] = s
        print(f"\n== {name}: 抜かれた {len(evs)} 件 ==")
        for k in keys:
            print(f"  {k:12s} 中央値 {s[k]['median']:.2f}  [{s[k]['q25']:.2f}, {s[k]['q75']:.2f}]")
        for k in ("frac_captured", "frac_gap_lt_1m", "frac_min_dist_lt_1m", "frac_max_s_after_lt_1m",
                  "frac_segment_ends_within_0p5s", "frac_crossing_like"):
            print(f"  {k:30s} {s[k]:.2f}")

    # 図: 正面で抜かれた事例
    evs = sorted(groups["正面 (|l0| 下位1/3)"], key=lambda e: e["dist0"])
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(evs), size=min(N_PLOT, len(evs)), replace=False))
    fig, axes = plt.subplots(4, 6, figsize=(21, 15))
    for ax, i in zip(axes.flat, pick):
        e = evs[i]
        for p, c in ((e["pa"], "tab:red"), (e["pd"], "tab:blue")):
            ax.plot(p[: e["n3"], 1], p[: e["n3"], 0], color=c, lw=1.8)
            ax.plot(p[e["n3"] - 1:, 1], p[e["n3"] - 1:, 0], color=c, lw=1, alpha=0.3)
            ax.plot(p[0, 1], p[0, 0], "o", color=c, ms=6)
            ax.plot(p[e["kb"], 1], p[e["kb"], 0], "x", color=c, ms=9, mew=2)
        ax.plot([e["pa"][e["kb"], 1], e["pd"][e["kb"], 1]], [e["pa"][e["kb"], 0], e["pd"][e["kb"], 0]],
                "k:", lw=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_xaxis()  # l 軸(左手側が正)を、ゴールを上にして見たときの左右に合わせる
        ax.grid(alpha=0.3)
        ax.set_title(f"d0={e['dist0']:.1f}m 抜き{e['t_beat']:.2f}s 横間隔{e['gap_at_beat']:.1f}m\n"
                     f"前進{e['max_s_after']:.1f}m 残{e['time_after']:.1f}s {e['outcome']}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    fig.suptitle("正面(|l0| 下位1/3)で「抜かれた」と判定された事例  "
                 "赤=攻撃者 青=守備者 ○=開始 ×=s=0 を横切った時刻 点線=その時の2人  上がゴール方向 [m]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("documents/phase12_headon_beaten.png", dpi=110)

    with open("documents/phase12_headon_beaten_results.json", "w") as f:
        json.dump(dict(s_pass=S_PASS, l0_cuts=l_cut.tolist(), summary=summary), f, indent=2, ensure_ascii=False)
    print("\n図と結果を保存しました")


if __name__ == "__main__":
    main()
