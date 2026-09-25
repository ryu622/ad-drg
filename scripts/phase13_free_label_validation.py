"""フェーズ13: 「守備者を外した」ラベル(ad_drg/free_label.py)の妥当性検証。

対象: 守備者が一度でもボールに介入可能になったイベント。守備者が初めて介入可能になった
フレームを「1対1の開始」とし、特徴量・ラベル・成果はすべてこの時刻を起点にする。
(当初は「セグメント開始時点で介入可能」を条件にしたが、1,454件中96件しか該当しなかったため
実行途中で変更した。セグメント開始 = ボールを受けた瞬間で、守備者はまだ寄せている途中のことが多い。)
ラベル y_freed: 開始から3秒以内に「守備者が T_act 以内にボール・シュートコースに届かない」状態が
0.2 秒以上続いたか。主設定は T_act = 0.5 s、r_reach = 1.0 m、シュートコースはゴールまで 30 m 以内で適用。
守備者の運動能力はフェーズ8bの役割別実測値(守備者 a_max p95, τ = v_max/a_max)。

成果(ad_drg/outcomes.py): 開始から8秒以内の シュート / PA進入 / ゴール方向へ10m以上前進(いずれも攻撃側保持中)。

【事前に決めた判定基準】(実行前に記述)
  (i)  主設定のラベルで、成果(any)の発生率が「外した」群で高い(リスク比の95%CIが1を超える)、かつ
       初期状態のベースライン(フェーズ7と同じ12特徴量)に対してラベルを加えると成果の予測 AUC が上がる
       (試合単位CV、ΔAUC の95%CIが0を超える)
  (ii) 旧ラベル(ゴール方向の前後関係 s >= 0 / s >= 1 m)より、成果との関連(ΔAUC)が大きい
       (共通の対象 = 1対1成立かつ守備者ゴール側 で比較)
  両方を満たせば、新ラベルを以後の分析の主ラベルとして採用する。
ラベルは成果と同じ時間帯の情報を含むので、これは事前予測ではなく「ラベルがサッカー的に
意味のある出来事を捉えているか」(基準関連妥当性)の検証である。
"""

from __future__ import annotations

import json
import time

import numpy as np

from ad_drg.cache import load_all_with_tracks
from ad_drg.calibration import calibration_stats_by_role
from ad_drg.evaluation import HORIZON, auc_with_ci, goal_axis, baseline_features, lomo_cv_scores, paired_auc_diff
from ad_drg.free_label import FreeParams, free_label
from ad_drg.outcomes import outcomes
from scripts.phase7_reach_avoid_auc import BASELINE_SET, goal_frame_state

OUTCOMES = ["any", "shot", "pa_entry", "progress10"]
N_BOOT = 2000


def risk_ratio(label: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    p1, p0 = y[label == 1].mean(), y[label == 0].mean()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        l, yy = label[idx], y[idx]
        if l.sum() == 0 or (1 - l).sum() == 0 or yy[l == 0].mean() == 0:
            continue
        boots.append(yy[l == 1].mean() / yy[l == 0].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return dict(rate_label1=float(p1), rate_label0=float(p0), rr=float(p1 / p0) if p0 > 0 else None,
                ci_lo=float(lo), ci_hi=float(hi))


def label_validity(rows, label_key, outcome_key):
    y = np.array([r["out"][outcome_key] for r in rows], dtype=int)
    lab = np.array([r[label_key] for r in rows], dtype=int)
    groups = np.array([r["match_id"] for r in rows])
    X = np.array([[r[f] for f in BASELINE_SET] for r in rows])
    oof_base = lomo_cv_scores(X, y, groups)
    oof_lab = lomo_cv_scores(np.hstack([X, lab[:, None]]), y, groups)
    return dict(
        n=len(rows), n_label=int(lab.sum()), n_outcome=int(y.sum()),
        risk_ratio=risk_ratio(lab, y),
        cv_base=auc_with_ci(y, oof_base)["auc"],
        cv_base_plus_label=auc_with_ci(y, oof_lab)["auc"],
        delta_auc=paired_auc_diff(y, oof_base, oof_lab),
    )


def fmt(v):
    rr = v["risk_ratio"]
    d = v["delta_auc"]
    rr_s = f"RR {rr['rr']:.2f} [{rr['ci_lo']:.2f},{rr['ci_hi']:.2f}]" if rr["rr"] else "RR -"
    return (f"n={v['n']} label={v['n_label']} outcome={v['n_outcome']} | 成果率 {rr['rate_label1']:.3f} vs "
            f"{rr['rate_label0']:.3f} {rr_s} | CV {v['cv_base']:.3f}→{v['cv_base_plus_label']:.3f} "
            f"Δ {d['diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]")


def main():
    t0 = time.time()
    trajs, tracks = load_all_with_tracks(verbose=False)
    role = calibration_stats_by_role(trajs)["defender"]
    a_def = role["a_max_p95"]
    tau = role["v_max_p95"] / a_def
    variants = {
        "主設定 T0.5 r1.0 +コース": FreeParams(a_def, tau, 0.5, 1.0, True),
        "T0.3 r1.0 +コース": FreeParams(a_def, tau, 0.3, 1.0, True),
        "T0.7 r1.0 +コース": FreeParams(a_def, tau, 0.7, 1.0, True),
        "T0.5 r0.5 +コース": FreeParams(a_def, tau, 0.5, 0.5, True),
        "T0.5 r1.0 ボールのみ": FreeParams(a_def, tau, 0.5, 1.0, False),
    }
    main_key = "主設定 T0.5 r1.0 +コース"
    print(f"守備者 a_max={a_def:.2f}, τ={tau:.3f}s")

    def row_at(traj, k, prm):
        x = goal_frame_state(traj, k)
        e_g = goal_axis(traj)
        s_series = (traj.p_a - traj.p_d) @ e_g
        n_end = min(len(s_series), k + int(round(HORIZON / traj.dt)) + 1)
        return dict(match_id=traj.match_id, attacker=traj.attacker, defender=traj.defender,
                    s0=float(x[0]), l0_abs=float(abs(x[1])), vs0=float(x[2]), vl0_abs=float(abs(x[3])),
                    **baseline_features(traj, k), out=outcomes(traj, tracks[traj.match_id], k_start=k),
                    beat_s0=int(np.any(s_series[k:n_end] >= 0.0)), beat_s1=int(np.any(s_series[k:n_end] >= 1.0)))

    base_rates = {o: float(np.mean([outcomes(t, tracks[t.match_id])[o] for t in trajs])) for o in OUTCOMES}
    print("全イベント(セグメント開始起点)の成果率:", {k: round(v, 3) for k, v in base_rates.items()})
    results = dict(base_rates=base_rates, variants={})
    rows_by_variant = {}
    for name, prm in variants.items():
        eng = []
        for traj in trajs:
            fl = free_label(traj, prm, HORIZON)
            if not fl["engaged"]:
                continue
            r = row_at(traj, fl["k_eng"], prm)
            r.update(_lab=int(fl["freed"]), t_free=fl["t_free"], k_eng=fl["k_eng"], traj=traj)
            eng.append(r)
        rows_by_variant[name] = eng
        res = dict(n_engaged=len(eng), frac_freed=float(np.mean([r["_lab"] for r in eng])),
                   median_onset_s=float(np.median([r["k_eng"] * r["traj"].dt for r in eng])), by_outcome={})
        print(f"\n=== {name}: 1対1成立 {len(eng)} / {len(trajs)} 件 (開始までの中央値 {res['median_onset_s']:.2f}s), "
              f"外した割合 {res['frac_freed']:.3f} ({time.time() - t0:.0f}s) ===")
        for o in OUTCOMES:
            v = label_validity(eng, "_lab", o)
            res["by_outcome"][o] = v
            print(f"  [{o:10s}] {fmt(v)}", flush=True)
        results["variants"][name] = res

    # (ii) 旧ラベルとの比較(共通の対象: 主設定で1対1成立 かつ 守備者ゴール側)
    common = [r for r in rows_by_variant[main_key] if r["s0"] < 0]
    for r in common:
        r["_new"] = r["_lab"]
    print(f"\n=== 旧ラベルとの比較: 1対1成立かつ守備者ゴール側 {len(common)} 件 ===")
    comp = {}
    for lab_name, key in (("新ラベル(外した)", "_new"), ("旧 s>=0", "beat_s0"), ("旧 s>=1m", "beat_s1")):
        comp[lab_name] = {}
        for o in OUTCOMES:
            v = label_validity(common, key, o)
            comp[lab_name][o] = v
            print(f"  {lab_name:10s} [{o:10s}] {fmt(v)}")
    # 新旧の ΔAUC 差(any)
    y = np.array([r["out"]["any"] for r in common], dtype=int)
    groups = np.array([r["match_id"] for r in common])
    X = np.array([[r[f] for f in BASELINE_SET] for r in common])
    oof = {k: lomo_cv_scores(np.hstack([X, np.array([[r[k]] for r in common])]), y, groups)
           for k in ("_new", "beat_s0", "beat_s1")}
    new_vs_old = {old: paired_auc_diff(y, oof[old], oof["_new"]) for old in ("beat_s0", "beat_s1")}
    for old, d in new_vs_old.items():
        print(f"  新 - {old} (any, ベースライン込み): ΔAUC {d['diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]")
    results["comparison_common"] = dict(n=len(common), labels=comp, new_minus_old=new_vs_old)

    m = results["variants"][main_key]["by_outcome"]["any"]
    crit_i = m["risk_ratio"]["ci_lo"] > 1 and m["delta_auc"]["ci_lo"] > 0
    crit_ii = all(d["diff"] > 0 for d in new_vs_old.values())
    crit_ii_strict = all(d["ci_lo"] > 0 for d in new_vs_old.values())
    print(f"\n判定 (i): {crit_i} / (ii) 点推定で旧ラベルを上回る: {crit_ii} (CIも0超: {crit_ii_strict})")
    results.update(criterion_i=crit_i, criterion_ii=crit_ii, criterion_ii_strict=crit_ii_strict)

    results.update(a_max_def=a_def, tau=tau)
    with open("documents/phase13_free_label_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"保存しました (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
