"""勝敗予測の評価ユーティリティ(フェーズ6以降)。

- 初期状態からの単純なベースライン特徴量(距離・接近速度など)
- 単一特徴量のAUC(ブートストラップ95%CI)
- 試合単位の leave-one-match-out 交差検証によるロジスティック回帰AUC
- 2つの予測スコアのAUC差のペアブートストラップ
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ad_drg.extraction import Trajectory

HORIZON = 3.0  # s: 予測ホライズン(フェーズ3の TIME_HORIZON と同じ)


def goal_axis(traj: Trajectory) -> np.ndarray:
    """初期時刻の攻撃者位置からゴール中心へ向かう単位ベクトル。"""
    g = traj.goal - traj.p_a[0]
    return g / np.linalg.norm(g)


def rotate_to_goal_frame(vec: np.ndarray, e_g: np.ndarray) -> np.ndarray:
    """ベクトル(…,2)を (ゴール方向成分, 左手側の横成分) に回転する。"""
    e_perp = np.array([-e_g[1], e_g[0]])
    return np.stack([vec @ e_g, vec @ e_perp], axis=-1)


def initial_state(traj: Trajectory, k: int = 0) -> np.ndarray:
    """フレームkの相対状態 (dpx, dpy, dvx, dvy)(ピッチ座標系)。"""
    return np.concatenate([traj.p_a[k] - traj.p_d[k], traj.v_a[k] - traj.v_d[k]])


def baseline_features(traj: Trajectory, k: int = 0) -> dict:
    """フレームkの状態だけから計算できる単純な特徴量(デフォルトは初期フレーム)。"""
    dp = traj.p_a[k] - traj.p_d[k]
    dv = traj.v_a[k] - traj.v_d[k]
    dist = float(np.linalg.norm(dp))
    closing = float(-(dp @ dv) / max(dist, 1e-6))  # 正なら接近中
    # 等速直線運動を仮定したときの、HORIZON 秒以内の最小距離
    t_star = np.clip(-(dp @ dv) / max(dv @ dv, 1e-9), 0.0, HORIZON)
    min_dist_cv = float(np.linalg.norm(dp + dv * t_star))
    e_g = goal_axis(traj)
    # 守備者が攻撃者から見てゴール方向にどれだけ寄っているか(cos)
    to_def = traj.p_d[k] - traj.p_a[k]
    goal_side_cos = float(to_def @ e_g / max(np.linalg.norm(to_def), 1e-6))
    return dict(
        dist0=dist,
        closing_speed=closing,
        rel_speed=float(np.linalg.norm(dv)),
        atk_speed=float(np.linalg.norm(traj.v_a[k])),
        def_speed=float(np.linalg.norm(traj.v_d[k])),
        min_dist_cv=min_dist_cv,
        goal_side_cos=goal_side_cos,
        dist_to_goal=float(np.linalg.norm(traj.goal - traj.p_a[k])),
    )


def auc_with_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(auc=float(auc), ci_lo=float(lo), ci_hi=float(hi))


def paired_auc_diff(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUC(s2) - AUC(s1) のペアブートストラップ。"""
    y, s1, s2 = map(np.asarray, (y, s1, s2))
    diff = roc_auc_score(y, s2) - roc_auc_score(y, s1)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s2[idx]) - roc_auc_score(y[idx], s1[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two_sided = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return dict(diff=float(diff), ci_lo=float(lo), ci_hi=float(hi), p_boot=p_two_sided)


def lomo_cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """試合単位 leave-one-match-out でロジスティック回帰の out-of-fold 予測確率を返す。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        test = groups == g
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[~test], y[~test])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof
