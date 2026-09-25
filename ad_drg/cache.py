"""抽出済み1対1軌道のディスクキャッシュ。

build_trajectories は試合ごとにトラッキングデータを読み込むため数分かかる。
reach-avoid 版(フェーズ6以降)では同じ軌道を何度も使うため、pickle で
`.cache/` に保存して再利用する。抽出ロジック(extraction.py)を変更した場合は
`.cache/` の pickle を削除して作り直すこと。

v2(フェーズ13〜): 軌道にフレーム番号・ボール位置を持たせ、試合全体のボール軌跡
(MatchTrack)も一緒に保存する。
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ad_drg.extraction import MATCH_IDS, MatchTrack, Trajectory, build_trajectories

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _load_match(match_id: str) -> tuple[list[Trajectory], MatchTrack]:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"match_v2_{match_id}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    trajs, track = build_trajectories(match_id, return_track=True)
    with open(path, "wb") as f:
        pickle.dump((trajs, track), f)
    return trajs, track


def load_all_trajectories(verbose: bool = True) -> list[Trajectory]:
    return load_all_with_tracks(verbose)[0]


def load_all_with_tracks(verbose: bool = True) -> tuple[list[Trajectory], dict[str, MatchTrack]]:
    all_trajectories, tracks = [], {}
    for match_id in MATCH_IDS:
        trajs, track = _load_match(match_id)
        if verbose:
            print(f"[{match_id}] trajectories={len(trajs)}", flush=True)
        all_trajectories.extend(trajs)
        tracks[match_id] = track
    return all_trajectories, tracks
