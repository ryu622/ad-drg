"""抽出済み1対1軌道のディスクキャッシュ。

build_trajectories は試合ごとにトラッキングデータを読み込むため数分かかる。
reach-avoid 版(フェーズ6以降)では同じ軌道を何度も使うため、pickle で
`.cache/` に保存して再利用する。抽出ロジック(extraction.py)を変更した場合は
`.cache/trajectories_*.pkl` を削除して作り直すこと。
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ad_drg.extraction import MATCH_IDS, Trajectory, build_trajectories

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def load_all_trajectories(verbose: bool = True) -> list[Trajectory]:
    CACHE_DIR.mkdir(exist_ok=True)
    all_trajectories = []
    for match_id in MATCH_IDS:
        path = CACHE_DIR / f"trajectories_{match_id}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                trajs = pickle.load(f)
        else:
            trajs = build_trajectories(match_id)
            with open(path, "wb") as f:
                pickle.dump(trajs, f)
        if verbose:
            print(f"[{match_id}] trajectories={len(trajs)}", flush=True)
        all_trajectories.extend(trajs)
    return all_trajectories
