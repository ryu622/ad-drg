"""フェーズ14: 1対1の結末の4分類と「前へ外した」ラベルの検証。

フェーズ13の探索的診断を受けて、1対1の結末を次の4つに分ける(ad_drg/free_label.classify_duel)。
  前へ外した   : 攻撃者がゴール方向に 1 m/s 以上で進みながら、守備者が T_act 以内にボール・
                 シュートコースに届かない状態が 0.2 s 以上続いた
  横・後ろへ逃げた: 届かない状態にはなったが、ゴール方向に進んでいなかった
  すぐはたいた : 外せないまま、1対1の開始から 0.5 s 以内にボール保持が終わった
  外せなかった : 外せないまま 0.5 s 以上ボールを持ち続けた
1対1の開始 = 守備者が初めてボールに介入可能になった瞬間。主設定は T_act = 0.5 s、
届く距離 1.0 m、シュートコースあり(ゴールまで 30 m 以内)、守備者の能力は実測値。

【データについての注意】利用できるデータは idsse-data の7試合のみで、分類の閾値(1 m/s など)は
フェーズ13で同じデータを見た後に決めた。以下は同じデータでの検証であり、独立データによる確認ではない。
閾値は下記の通り実行前に固定し、感度分析で結論が閾値に依存しないかを確認する。

主な成果: 「脅威」= 1対1の開始から8秒以内に、攻撃側が保持したままシュートまたは PA 進入。
(ゴール方向への10m前進は「ゴール方向に進んでいる」という分類条件とほぼ機械的に結びつくため、
主な成果から外し、参考として報告する。)

【事前に決めた判定基準】(実行前に記述)
  (i)   脅威の発生率が「前へ外した」で「外せなかった」より高い(リスク比の95%CIの下限 > 1)
  (ii)  脅威の発生率が「前へ外した」で「前に進んだが外せなかった」より高い(リスク比のCI下限 > 1)。
        「前に進んだが外せなかった」= 開始から3秒以内にゴール方向 1 m/s 以上で 0.2 s 以上進んだが、
        外した状態にはならなかった(すぐはたいた・外せなかった)イベント。
        前に動いたこと自体ではなく「外した」ことに意味があるかを見る、最も重要な比較。
  (iii) 頑健性: (ii) のリスク比の点推定が、1試合ずつ除いた7通りすべてで > 1、かつ感度分析
        (ゴール方向速度の閾値 0.5 / 2.0 m/s、T_act 0.3 / 0.7 s)すべてで > 1
  すべて満たせば「前へ外した」を突破の主ラベルとして採用する。
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from ad_drg.cache import load_all_with_tracks
from ad_drg.calibration import calibration_stats_by_role
from ad_drg.evaluation import HORIZON, auc_with_ci, baseline_features, lomo_cv_scores, paired_auc_diff
from ad_drg.free_label import CATEGORIES, CATEGORY_JA, FreeParams, classify_duel
from ad_drg.outcomes import outcomes
from scripts.phase13_free_label_validation import risk_ratio
from scripts.phase7_reach_avoid_auc import BASELINE_SET, goal_frame_state

N_BOOT = 2000


def build(trajs, tracks, prm, fwd_thr):
    rows = []
    for traj in trajs:
        c = classify_duel(traj, prm, HORIZON, fwd_thr=fwd_thr)
        if not c["engaged"]:
            continue
        k = c["k_eng"]
        out = outcomes(traj, tracks[traj.match_id], k_start=k)
        x = goal_frame_state(traj, k)
        rows.append(dict(
            match_id=traj.match_id, attacker=traj.attacker, defender=traj.defender,
            category=c["category"], went_forward=c["went_forward"],
            threat=int(out["shot"] or out["pa_entry"]), shot=int(out["shot"]), pa_entry=int(out["pa_entry"]),
            progress10=int(out["progress10"]), captured=int(traj.outcome == "captured"),
            s0=float(x[0]), l0_abs=float(abs(x[1])), vs0=float(x[2]), vl0_abs=float(abs(x[3])),
            **baseline_features(traj, k),
        ))
    return rows


def rr_between(rows, group1, group0, outcome="threat"):
    sub = [r for r in rows if group1(r) or group0(r)]
    lab = np.array([int(group1(r)) for r in sub])
    y = np.array([r[outcome] for r in sub])
    return risk_ratio(lab, y), len(sub), int(lab.sum())


is_ff = lambda r: r["category"] == "forward_free"
is_contained = lambda r: r["category"] == "contained"
is_fwd_covered = lambda r: r["went_forward"] and r["category"] in ("release", "contained")


def main():
    t0 = time.time()
    trajs, tracks = load_all_with_tracks(verbose=False)
    role = calibration_stats_by_role(trajs)["defender"]
    a_def = role["a_max_p95"]
    tau = role["v_max_p95"] / a_def
    prm = FreeParams(a_def, tau, 0.5, 1.0, True)
    rows = build(trajs, tracks, prm, 1.0)
    print(f"1対1成立 {len(rows)} 件 ({time.time() - t0:.0f}s)")

    # 分類ごとの成果率
    table = {}
    print(f"\n{'分類':14s} {'n':>4s} {'脅威':>6s} {'シュート':>6s} {'PA進入':>6s} {'前進10m':>7s} {'奪われた':>6s}")
    for cat in CATEGORIES:
        sub = [r for r in rows if r["category"] == cat]
        table[cat] = dict(n=len(sub), **{o: float(np.mean([r[o] for r in sub])) for o in
                                          ("threat", "shot", "pa_entry", "progress10", "captured")})
        t = table[cat]
        print(f"{CATEGORY_JA[cat]:12s} {t['n']:4d} {t['threat']:6.3f} {t['shot']:6.3f} {t['pa_entry']:6.3f} "
              f"{t['progress10']:7.3f} {t['captured']:6.3f}")
    fc = [r for r in rows if is_fwd_covered(r)]
    table["forward_covered"] = dict(n=len(fc), **{o: float(np.mean([r[o] for r in fc])) for o in
                                                  ("threat", "shot", "pa_entry", "progress10", "captured")})
    t = table["forward_covered"]
    print(f"{'(前に進んだが外せなかった)':12s} {t['n']:4d} {t['threat']:6.3f} {t['shot']:6.3f} {t['pa_entry']:6.3f} "
          f"{t['progress10']:7.3f} {t['captured']:6.3f}")

    # (i)(ii)
    rr_i, n_i, _ = rr_between(rows, is_ff, is_contained)
    rr_ii, n_ii, _ = rr_between(rows, is_ff, is_fwd_covered)
    print(f"\n(i)  前へ外した vs 外せなかった: 脅威 {rr_i['rate_label1']:.3f} vs {rr_i['rate_label0']:.3f} "
          f"RR {rr_i['rr']:.2f} [{rr_i['ci_lo']:.2f}, {rr_i['ci_hi']:.2f}] (n={n_i})")
    print(f"(ii) 前へ外した vs 前に進んだが外せなかった: 脅威 {rr_ii['rate_label1']:.3f} vs {rr_ii['rate_label0']:.3f} "
          f"RR {rr_ii['rr']:.2f} [{rr_ii['ci_lo']:.2f}, {rr_ii['ci_hi']:.2f}] (n={n_ii})")
    # 参考: (ii) の比較で成果ごと
    rr_ii_by = {o: rr_between(rows, is_ff, is_fwd_covered, o)[0] for o in ("shot", "pa_entry", "progress10")}
    for o, v in rr_ii_by.items():
        print(f"     参考 {o:10s} {v['rate_label1']:.3f} vs {v['rate_label0']:.3f} RR {v['rr']:.2f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")

    # 参考: 初期状態で調整した上乗せ((ii) の対象で、ベースライン vs ベースライン+前へ外した)
    sub = [r for r in rows if is_ff(r) or is_fwd_covered(r)]
    y = np.array([r["threat"] for r in sub])
    g = np.array([r["match_id"] for r in sub])
    X = np.array([[r[f] for f in BASELINE_SET] for r in sub])
    lab = np.array([[int(is_ff(r))] for r in sub])
    oof_b, oof_l = lomo_cv_scores(X, y, g), lomo_cv_scores(np.hstack([X, lab]), y, g)
    adj = dict(base=auc_with_ci(y, oof_b)["auc"], plus=auc_with_ci(y, oof_l)["auc"], diff=paired_auc_diff(y, oof_b, oof_l))
    print(f"     参考 初期状態で調整: CV AUC {adj['base']:.3f} → {adj['plus']:.3f} "
          f"Δ {adj['diff']['diff']:+.3f} [{adj['diff']['ci_lo']:+.3f}, {adj['diff']['ci_hi']:+.3f}]")

    # (iii) 1試合除外
    loo = {}
    for m in sorted(set(r["match_id"] for r in rows)):
        v, _, _ = rr_between([r for r in rows if r["match_id"] != m], is_ff, is_fwd_covered)
        loo[m] = v["rr"]
    print("\n(iii) 1試合除外の RR:", {k: round(v, 2) for k, v in loo.items()})
    sens = {}
    for name, (t_act, thr) in {"fwd0.5": (0.5, 0.5), "fwd2.0": (0.5, 2.0), "T0.3": (0.3, 1.0), "T0.7": (0.7, 1.0)}.items():
        rs = build(trajs, tracks, FreeParams(a_def, tau, t_act, 1.0, True), thr)
        v, n, n1 = rr_between(rs, is_ff, is_fwd_covered)
        vi, _, _ = rr_between(rs, is_ff, is_contained)
        cnt = Counter(r["category"] for r in rs)
        sens[name] = dict(rr_ii=v, rr_i=vi, n_duels=len(rs), categories=dict(cnt))
        print(f"      感度 {name}: 1対1 {len(rs)}件 分類 {dict(cnt)} | (ii) RR {v['rr']:.2f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]"
              f" | (i) RR {vi['rr']:.2f} [{vi['ci_lo']:.2f}, {vi['ci_hi']:.2f}]", flush=True)

    crit_i = rr_i["ci_lo"] > 1
    crit_ii = rr_ii["ci_lo"] > 1
    crit_iii = all(v > 1 for v in loo.values()) and all(s["rr_ii"]["rr"] > 1 for s in sens.values())
    print(f"\n判定: (i) {crit_i} / (ii) {crit_ii} / (iii) {crit_iii}")

    out = dict(params=dict(t_act=0.5, r_reach=1.0, fwd_thr=1.0, release_s=0.5, a_max_def=a_def, tau=tau),
               n_duels=len(rows), table=table, rr_i=rr_i, rr_ii=rr_ii, rr_ii_by_outcome=rr_ii_by,
               adjusted_auc=adj, leave_one_match_out=loo, sensitivity=sens,
               criterion_i=crit_i, criterion_ii=crit_ii, criterion_iii=crit_iii)
    with open("documents/phase14_duel_categories_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"保存しました (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
