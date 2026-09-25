"""1対1区間のあとに起きた攻撃側の成果(フェーズ13の妥当性検証用)。

1対1の開始時刻から window 秒以内に、攻撃側チームが次のいずれかを達成したかを判定する。
  shot        : 攻撃側チームのシュート(イベントデータの ShotEvent)
  pa_entry    : 攻撃側チームが保持したまま、ボールが相手ペナルティエリアに入る(トラッキング)
  progress10  : 攻撃側チームが保持したまま、ボールが開始時点からゴール方向に 10 m 以上進む(トラッキング)
pa_entry / progress10 は、保持チームが相手に替わった時点で判定を打ち切る。
PA進入はパス成功だけでなくドリブル侵入も含む(トラッキングからの近似)。

イベントとトラッキングの時刻は同じ基準(ピリオド内の経過秒)だが、1〜2秒ずれる例がある
(フェーズ13の確認)。判定窓が8秒なので実用上の影響は小さいとみなす。
"""

from __future__ import annotations

import numpy as np
from kloppy import sportec

from ad_drg.extraction import MatchTrack, Trajectory

PA_LENGTH = 16.5
PA_HALF_WIDTH = 20.16
_shot_cache: dict[str, list[tuple[int, float, str]]] = {}


def shots_of(match_id: str) -> list[tuple[int, float, str]]:
    if match_id not in _shot_cache:
        ev = sportec.load_open_event_data(match_id=match_id)
        _shot_cache[match_id] = [
            (e.period.id, e.timestamp.total_seconds(), e.team.team_id)
            for e in ev.events
            if e.event_type.name == "SHOT" and e.team is not None
        ]
    return _shot_cache[match_id]


def outcomes(traj: Trajectory, track: MatchTrack, window: float = 8.0, progress_m: float = 10.0, k_start: int = 0) -> dict:
    """k_start: 判定窓の起点となる軌道内のフレーム(1対1の開始)。"""
    k0 = int(traj.frame_idx[k_start])
    period, t0 = track.period[k0], track.t[k0]
    team = traj.attacker_team
    goal = track.goal_of(team, k0)
    sign = 1.0 if goal[0] > track.pitch_length / 2 else -1.0

    idx = np.nonzero((track.period == period) & (track.t >= t0) & (track.t <= t0 + window))[0]
    owner = track.owning_team[idx]
    lost = np.nonzero((owner != team) & (owner != ""))[0]
    if len(lost):
        idx = idx[: lost[0]]
    ball = track.ball[idx]
    ok = ~np.isnan(ball[:, 0])
    ball = ball[ok]

    progress = sign * (ball[:, 0] - ball[0, 0]) if len(ball) else np.array([0.0])
    in_pa = (np.abs(ball[:, 0] - goal[0]) <= PA_LENGTH) & (np.abs(ball[:, 1] - goal[1]) <= PA_HALF_WIDTH)
    shot = any(p == period and t0 <= t <= t0 + window and tm == team for p, t, tm in shots_of(traj.match_id))
    res = dict(
        shot=bool(shot),
        pa_entry=bool(in_pa.any()) and not bool(in_pa[0]),
        progress10=bool(progress.max() >= progress_m),
        max_progress=float(progress.max()),
        start_in_pa=bool(in_pa[0]) if len(in_pa) else False,
        dist_to_goal0=float(np.linalg.norm(goal - traj.ball[k_start])),
    )
    res["any"] = res["shot"] or res["pa_entry"] or res["progress10"]
    return res
