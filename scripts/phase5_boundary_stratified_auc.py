"""フェーズ5(診断テスト): AD モデルの境界張り付き事例とそれ以外で、HJI捕捉可能集合の
予測力(AUC)に差があるかを検証する。

背景: 6.2節(phase3)の「AD vs HJIの軌道再現誤差比較」は、ADが有利になる比較軸
(記述的モデル vs 規範的方策)だったため、HJIの戦略的適応性が実際にADの弱点を
補っているかを直接には検証できていなかった。

本スクリプトでは、phase1/phase3で既に計算済みのAD境界張り付きフラグ(70イベント、
COBYLA再フィット不要・再利用)を使い、
    - ADが境界に張り付いた事例(戦略的適応性の欠如が疑われる事例)
    - 張り付いていない事例
の2群で、HJIの捕捉可能集合による予測(6.1節と同じ value_at ベースのスコア)の
AUC・適合率・再現率を separately に計算する。境界張り付き群でAUCが高ければ、
「ADが苦手とする局面ほどHJIの方が説明力を持つ」という、当初の仮説(1.3節②)を
支持する直接的な証拠になる。
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from ad_drg.calibration import calibration_stats
from ad_drg.extraction import MATCH_IDS, build_trajectories
from ad_drg.reachability import solve_brs

P_BOUND = 25.0
GRID_SHAPE = (31, 31, 21, 21)
TIME_HORIZON = 3.0
N_TIME_STEPS = 21
ACCURACY = "medium"


def score_subset(rows):
    y_true = np.array([r["y_true"] for r in rows])
    y_pred = np.array([r["y_pred"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    if len(set(y_true.tolist())) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, scores)
    return dict(
        n=len(rows),
        n_positive=int(y_true.sum()),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        auc=float(auc),
    )


def main():
    with open("documents/phase3_validation_results.json") as f:
        phase3 = json.load(f)
    sampled_events = phase3["section3"]["rows"]  # 既存の70イベント(AD境界フラグ込み)
    print(f"phase3から再利用するサンプル数: {len(sampled_events)}")

    all_trajectories = []
    for match_id in MATCH_IDS:
        trajs = build_trajectories(match_id)
        all_trajectories.extend(trajs)
    print(f"全軌道数: {len(all_trajectories)}")

    calib = calibration_stats(all_trajectories)
    a_max = calib["a_max_p95"]
    v_max = calib["v_max_p95"]
    v_bound = 2.0 * v_max

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

    traj_lookup = {(t.match_id, t.attacker, t.defender): t for t in all_trajectories}

    boundary_rows, non_boundary_rows, unmatched = [], [], 0
    for ev in sampled_events:
        key = (ev["match_id"], ev["attacker"], ev["defender"])
        traj = traj_lookup.get(key)
        if traj is None or traj.outcome not in ("captured", "evaded"):
            unmatched += 1
            continue
        dp0 = traj.p_a[0] - traj.p_d[0]
        dv0 = traj.v_a[0] - traj.v_d[0]
        x0 = np.concatenate([dp0, dv0])
        duration = traj.t[-1] if len(traj.t) > 1 else traj.dt
        v = brs.value_at(x0, t=min(duration, TIME_HORIZON))
        if np.isnan(v):
            unmatched += 1
            continue
        row = dict(
            y_true=1 if traj.outcome == "captured" else 0,
            y_pred=1 if v <= 0 else 0,
            score=-v,
        )
        if ev["ad_any_at_bound"]:
            boundary_rows.append(row)
        else:
            non_boundary_rows.append(row)

    print(f"照合できなかった事例: {unmatched}件")
    print(f"境界張り付き群: n={len(boundary_rows)}, 非張り付き群: n={len(non_boundary_rows)}")

    boundary_stats = score_subset(boundary_rows) if boundary_rows else None
    non_boundary_stats = score_subset(non_boundary_rows) if non_boundary_rows else None

    print("\n=== 境界張り付き群(ADの適応性欠如が疑われる事例) ===")
    print(json.dumps(boundary_stats, indent=2, ensure_ascii=False))
    print("\n=== 非張り付き群(ADが正常にフィットできた事例) ===")
    print(json.dumps(non_boundary_stats, indent=2, ensure_ascii=False))

    out = dict(
        n_sampled=len(sampled_events),
        n_unmatched=unmatched,
        boundary=boundary_stats,
        non_boundary=non_boundary_stats,
    )
    with open("documents/phase5_boundary_stratified_auc.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n結果を documents/phase5_boundary_stratified_auc.json に保存しました")


if __name__ == "__main__":
    main()
