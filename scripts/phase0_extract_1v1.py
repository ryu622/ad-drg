"""フェーズ0(a): idsse-data 7試合から抽出可能な1対1ドリブルイベント数を数える。

research_plan.md 4.2節の抽出基準を、トラッキングデータのみから近似的に再現する:
    1. 最近傍守備者がドリブル中に交代しない
    2. ボール保持時間が0.5秒超
    3. 攻撃者・守備者双方の移動距離が5m超

イベントデータに明示的な「ドリブル/デュエル」イベントが存在しないため(J03WPY確認済み:
PassEvent/ShotEvent/RecoveryEvent/BallOutEvent/FoulCommittedEvent/GenericEvent/CardEvent/
SubstitutionEventのみ)、ボール保持者(=ボールに最も近い保持チームの選手)が連続して
同一である区間を「ボール保持セグメント」として抽出し、その区間に対して基準1〜3を適用する。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState, TrackingDataset

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]

MAX_CARRIER_DIST = 3.0  # m: ボールとの距離がこれ以内の保持チーム選手を「保持者」とみなす
MIN_DURATION = 0.5  # s
MIN_DISPLACEMENT = 5.0  # m


@dataclass
class MatchStats:
    match_id: str
    n_frames: int = 0
    n_possession_segments: int = 0
    n_1v1_events: int = 0
    reject_duration: int = 0
    reject_defender_unstable: int = 0
    reject_displacement: int = 0
    events: list = field(default_factory=list)


def frame_positions_m(frame, pitch_length: float, pitch_width: float) -> dict[str, np.ndarray]:
    pos = {}
    for player, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue
        pos[player.player_id] = np.array(
            [pdata.coordinates.x * pitch_length, pdata.coordinates.y * pitch_width]
        )
    return pos


def extract_possession_segments(tracking: TrackingDataset, team_by_player: dict[str, str]):
    """ボール保持者が連続して同一である(フレーム, 座標辞書)の区間を返す。"""
    pitch_length = tracking.metadata.pitch_dimensions.pitch_length
    pitch_width = tracking.metadata.pitch_dimensions.pitch_width

    segments: list[tuple[str, list[dict[str, np.ndarray]]]] = []
    cur_carrier = None
    cur_frames: list[dict[str, np.ndarray]] = []

    def flush():
        if cur_carrier is not None and len(cur_frames) >= 2:
            segments.append((cur_carrier, cur_frames))

    for frame in tracking.records:
        if frame.ball_coordinates is None or frame.ball_state != BallState.ALIVE:
            flush()
            cur_carrier, cur_frames = None, []
            continue

        owning_team = frame.ball_owning_team
        pos = frame_positions_m(frame, pitch_length, pitch_width)
        if owning_team is None or not pos:
            flush()
            cur_carrier, cur_frames = None, []
            continue

        ball_xy = np.array(
            [frame.ball_coordinates.x * pitch_length, frame.ball_coordinates.y * pitch_width]
        )
        candidates = [
            (pid, xy) for pid, xy in pos.items() if team_by_player.get(pid) == owning_team.team_id
        ]
        if not candidates:
            flush()
            cur_carrier, cur_frames = None, []
            continue

        carrier_id, carrier_xy = min(candidates, key=lambda kv: np.linalg.norm(kv[1] - ball_xy))
        if np.linalg.norm(carrier_xy - ball_xy) > MAX_CARRIER_DIST:
            flush()
            cur_carrier, cur_frames = None, []
            continue

        if carrier_id != cur_carrier:
            flush()
            cur_carrier = carrier_id
            cur_frames = []

        cur_frames.append(pos)

    flush()
    return segments


def evaluate_segment(carrier_id: str, frames: list[dict[str, np.ndarray]], fps: float, team_by_player: dict[str, str], stats: MatchStats):
    duration = len(frames) / fps
    if duration <= MIN_DURATION:
        stats.reject_duration += 1
        return None

    attacker_team = team_by_player[carrier_id]
    start_pos = frames[0]
    if carrier_id not in start_pos:
        return None

    def nearest_opponent(pos):
        opp = [(pid, xy) for pid, xy in pos.items() if team_by_player.get(pid) != attacker_team]
        if not opp or carrier_id not in pos:
            return None
        return min(opp, key=lambda kv: np.linalg.norm(kv[1] - pos[carrier_id]))[0]

    defender_id = nearest_opponent(start_pos)
    if defender_id is None:
        return None

    for pos in frames:
        if nearest_opponent(pos) != defender_id:
            stats.reject_defender_unstable += 1
            return None

    atk_positions = np.array([pos[carrier_id] for pos in frames if carrier_id in pos])
    def_positions = np.array([pos[defender_id] for pos in frames if defender_id in pos])
    if len(atk_positions) < 2 or len(def_positions) < 2:
        return None

    atk_dist = float(np.linalg.norm(np.diff(atk_positions, axis=0), axis=1).sum())
    def_dist = float(np.linalg.norm(np.diff(def_positions, axis=0), axis=1).sum())
    if atk_dist <= MIN_DISPLACEMENT or def_dist <= MIN_DISPLACEMENT:
        stats.reject_displacement += 1
        return None

    return dict(
        attacker=carrier_id,
        defender=defender_id,
        duration=round(duration, 2),
        atk_dist=round(atk_dist, 2),
        def_dist=round(def_dist, 2),
        n_frames=len(frames),
    )


def process_match(match_id: str) -> MatchStats:
    t0 = time.time()
    tracking = sportec.load_open_tracking_data(match_id=match_id, only_alive=True)
    fps = tracking.metadata.frame_rate
    team_by_player = {
        p.player_id: team.team_id for team in tracking.metadata.teams for p in team.players
    }

    stats = MatchStats(match_id=match_id, n_frames=len(tracking.records))
    segments = extract_possession_segments(tracking, team_by_player)
    stats.n_possession_segments = len(segments)

    for carrier_id, frames in segments:
        ev = evaluate_segment(carrier_id, frames, fps, team_by_player, stats)
        if ev is not None:
            stats.n_1v1_events += 1
            stats.events.append(ev)

    print(
        f"[{match_id}] frames={stats.n_frames} segments={stats.n_possession_segments} "
        f"1v1={stats.n_1v1_events} "
        f"(reject: duration={stats.reject_duration}, defender_unstable={stats.reject_defender_unstable}, "
        f"displacement={stats.reject_displacement}) "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return stats


def main():
    all_stats = {}
    for match_id in MATCH_IDS:
        stats = process_match(match_id)
        all_stats[match_id] = stats

    total = sum(s.n_1v1_events for s in all_stats.values())
    print(f"\n合計1対1イベント数: {total} 件 (7試合)")
    print(f"1試合あたり平均: {total / len(MATCH_IDS):.1f} 件")

    out = {
        mid: dict(
            n_frames=s.n_frames,
            n_possession_segments=s.n_possession_segments,
            n_1v1_events=s.n_1v1_events,
            reject_duration=s.reject_duration,
            reject_defender_unstable=s.reject_defender_unstable,
            reject_displacement=s.reject_displacement,
            events=s.events,
        )
        for mid, s in all_stats.items()
    }
    out["_summary"] = dict(total_1v1_events=total, mean_per_match=total / len(MATCH_IDS))

    out_path = "documents/phase0_1v1_counts.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"結果を {out_path} に保存しました")


if __name__ == "__main__":
    main()
