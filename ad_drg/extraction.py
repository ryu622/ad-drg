"""idsse-data (kloppy/Sportec) から1対1ドリブルの軌道を抽出する。

抽出基準は Narizuka & Yamazaki (2026, arXiv:2512.22596) 2.1節に基づく:
    (i)   最近傍守備者がドリブル中に交代しない
    (ii)  ボール保持時間が0.5秒超
    (iii) 攻撃者・守備者双方の"linear distance traveled"が5m超

(iii) について: 原文 "the linear distance traveled by both players exceeded
5 meters" は、経路長(移動距離の総和)か始点終点間の直線距離(変位)かが
原文だけでは一意に定まらない。本実装では両方を計算して Trajectory に保持し、
デフォルトのフィルタには経路長(より一般的な「移動距離」の解釈)を用いる。
どちらの基準を採用するかはフェーズ1で先行研究の追試結果と突き合わせて再検討する。

idsse-dataのイベントデータには明示的な「ドリブル/デュエル」イベントが存在しないため
(PassEvent/ShotEvent/RecoveryEvent/BallOutEvent/FoulCommittedEvent/GenericEvent/
CardEvent/SubstitutionEventのみ確認)、トラッキングデータからボール保持者
(ボールとの距離3m以内の保持チーム選手)の連続区間を「ボール保持セグメント」として
独自に再構成し、これを1対1ドリブル候補として扱う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kloppy import sportec
from kloppy.domain import AttackingDirection, BallState, Ground
from scipy.signal import savgol_filter

MATCH_IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]

MAX_CARRIER_DIST = 3.0  # m: ボールとの距離がこれ以内の保持チーム選手を「保持者」とみなす
MIN_DURATION = 0.5  # s
MIN_DISPLACEMENT = 5.0  # m
GOAL_Y_FRAC = 0.5  # ゴール中心のy座標(ピッチ幅に対する比率)


@dataclass
class Trajectory:
    match_id: str
    attacker: str
    defender: str
    dt: float
    t: np.ndarray
    p_a: np.ndarray  # (N, 2) m
    p_d: np.ndarray
    v_a: np.ndarray  # savgolフィルタによる速度推定 (N, 2) m/s
    v_d: np.ndarray
    goal: np.ndarray  # (2,) 攻撃者が狙うゴール = 守備者の自陣ゴール
    atk_path_length: float
    def_path_length: float
    atk_displacement: float
    def_displacement: float
    outcome: str | None  # "captured" | "evaded" | None(次のセグメントが取得できない)
    frame_idx: np.ndarray | None = None  # (N,) MatchTrack 上のフレーム番号(フェーズ13以降)
    ball: np.ndarray | None = None  # (N, 2) m: ボール位置(フェーズ13以降)
    attacker_team: str | None = None


@dataclass
class MatchTrack:
    """試合全体(ボールインプレー中のフレーム)のボール軌跡と保持チーム。

    1対1区間の後に何が起きたか(シュート・PA進入・前進)を判定するために使う(フェーズ13)。
    """

    match_id: str
    fps: float
    period: np.ndarray  # (M,) int
    t: np.ndarray  # (M,) s: ピリオド内の経過時間(イベントデータの timestamp と同じ基準)
    ball: np.ndarray  # (M, 2) m
    owning_team: np.ndarray  # (M,) str(保持チーム不明は "")
    home_attacks_ltr: np.ndarray  # (M,) bool
    home_team: str
    pitch_length: float
    pitch_width: float

    def goal_of(self, team_id: str, k: int) -> np.ndarray:
        """フレームkで team_id が攻めるゴールの中心座標。"""
        ltr = self.home_attacks_ltr[k] if team_id == self.home_team else not self.home_attacks_ltr[k]
        return np.array([self.pitch_length if ltr else 0.0, self.pitch_width * GOAL_Y_FRAC])


def _frame_positions_m(frame, pitch_length: float, pitch_width: float) -> dict[str, np.ndarray]:
    pos = {}
    for player, pdata in frame.players_data.items():
        if pdata.coordinates is None:
            continue
        pos[player.player_id] = np.array(
            [pdata.coordinates.x * pitch_length, pdata.coordinates.y * pitch_width]
        )
    return pos


def _goal_position(
    team_ground: Ground, attacking_direction: AttackingDirection, pitch_length: float, pitch_width: float
) -> np.ndarray:
    home_attacks_ltr = attacking_direction == AttackingDirection.LTR
    team_attacks_ltr = home_attacks_ltr if team_ground == Ground.HOME else not home_attacks_ltr
    goal_x = pitch_length if team_attacks_ltr else 0.0
    return np.array([goal_x, pitch_width * GOAL_Y_FRAC])


def _smooth_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    n = len(pos)
    if n < 5:
        return np.gradient(pos, axis=0) * fps
    window = min(9, n if n % 2 == 1 else n - 1)
    if window % 2 == 0:
        window -= 1
    polyorder = min(3, window - 1)
    return savgol_filter(pos, window_length=window, polyorder=polyorder, deriv=1, delta=1.0 / fps, axis=0)


def _path_length(pos: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def build_trajectories(match_id: str, min_displacement: float = MIN_DISPLACEMENT, return_track: bool = False):
    """1試合分のトラッキングデータから1対1ドリブル軌道のリストを構築する。

    min_displacement は基準(iii)のしきい値で、経路長(atk_path_length /
    def_path_length)に対して適用する。return_track=True なら (軌道のリスト, MatchTrack) を返す。
    """
    tracking = sportec.load_open_tracking_data(match_id=match_id, only_alive=True)
    fps = tracking.metadata.frame_rate
    dt = 1.0 / fps
    pitch_length = tracking.metadata.pitch_dimensions.pitch_length
    pitch_width = tracking.metadata.pitch_dimensions.pitch_width
    team_by_player = {p.player_id: team.team_id for team in tracking.metadata.teams for p in team.players}
    ground_by_team = {team.team_id: team.ground for team in tracking.metadata.teams}

    segments: list[tuple[str, list, AttackingDirection, list, list]] = []
    cur_carrier, cur_frames, cur_dir = None, [], None
    cur_idx, cur_ball = [], []
    home_team = next(team.team_id for team in tracking.metadata.teams if team.ground == Ground.HOME)
    track_period, track_t, track_ball, track_owner, track_ltr = [], [], [], [], []

    def flush():
        if cur_carrier is not None and len(cur_frames) >= 2:
            segments.append((cur_carrier, cur_frames, cur_dir, cur_idx, cur_ball))

    for frame_i, frame in enumerate(tracking.records):
        track_period.append(frame.period.id)
        track_t.append(frame.timestamp.total_seconds())
        track_ltr.append(frame.attacking_direction == AttackingDirection.LTR)
        track_owner.append(frame.ball_owning_team.team_id if frame.ball_owning_team is not None else "")
        if frame.ball_coordinates is not None:
            track_ball.append([frame.ball_coordinates.x * pitch_length, frame.ball_coordinates.y * pitch_width])
        else:
            track_ball.append([np.nan, np.nan])
        if frame.ball_coordinates is None or frame.ball_state != BallState.ALIVE:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            cur_idx, cur_ball = [], []
            continue
        owning_team = frame.ball_owning_team
        pos = _frame_positions_m(frame, pitch_length, pitch_width)
        if owning_team is None or not pos:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            cur_idx, cur_ball = [], []
            continue
        ball_xy = np.array([frame.ball_coordinates.x * pitch_length, frame.ball_coordinates.y * pitch_width])
        candidates = [(pid, xy) for pid, xy in pos.items() if team_by_player.get(pid) == owning_team.team_id]
        if not candidates:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            cur_idx, cur_ball = [], []
            continue
        carrier_id, carrier_xy = min(candidates, key=lambda kv: np.linalg.norm(kv[1] - ball_xy))
        if np.linalg.norm(carrier_xy - ball_xy) > MAX_CARRIER_DIST:
            flush()
            cur_carrier, cur_frames, cur_dir = None, [], None
            cur_idx, cur_ball = [], []
            continue
        if carrier_id != cur_carrier:
            flush()
            cur_carrier = carrier_id
            cur_frames = []
            cur_idx, cur_ball = [], []
            cur_dir = frame.attacking_direction
        cur_frames.append(pos)
        cur_idx.append(frame_i)
        cur_ball.append(ball_xy)
    flush()

    trajectories = []
    for seg_idx, (carrier_id, frames, atk_dir, frame_idx, ball_seg) in enumerate(segments):
        duration = len(frames) / fps
        if duration <= MIN_DURATION:
            continue
        attacker_team = team_by_player[carrier_id]
        start_pos = frames[0]
        if carrier_id not in start_pos:
            continue

        def nearest_opponent(pos):
            opp = [(pid, xy) for pid, xy in pos.items() if team_by_player.get(pid) != attacker_team]
            if not opp or carrier_id not in pos:
                return None
            return min(opp, key=lambda kv: np.linalg.norm(kv[1] - pos[carrier_id]))[0]

        defender_id = nearest_opponent(start_pos)
        if defender_id is None:
            continue
        if not all(nearest_opponent(pos) == defender_id for pos in frames):
            continue
        if not all(carrier_id in pos and defender_id in pos for pos in frames):
            continue

        p_a = np.array([pos[carrier_id] for pos in frames])
        p_d = np.array([pos[defender_id] for pos in frames])
        atk_path = _path_length(p_a)
        def_path = _path_length(p_d)
        atk_disp = float(np.linalg.norm(p_a[-1] - p_a[0]))
        def_disp = float(np.linalg.norm(p_d[-1] - p_d[0]))
        if atk_path <= min_displacement or def_path <= min_displacement:
            continue

        v_a = _smooth_velocity(p_a, fps)
        v_d = _smooth_velocity(p_d, fps)
        goal = _goal_position(ground_by_team[attacker_team], atk_dir, pitch_length, pitch_width)
        t = np.arange(len(p_a)) * dt

        # outcome: 次のボール保持セグメントの保持チームで守備成功/失敗を近似する
        # (このセグメントの直後にボールを持つのが守備側チームなら「捕捉」とみなす)。
        # ファウル・アウトオブプレー等を挟んでも次セグメントの取得元チームで判定する
        # 簡易プロキシであり、正式なタックル/インターセプトのイベントラベルではない点に注意。
        outcome = None
        if seg_idx + 1 < len(segments):
            next_carrier_id = segments[seg_idx + 1][0]
            next_team = team_by_player.get(next_carrier_id)
            if next_team is not None:
                outcome = "evaded" if next_team == attacker_team else "captured"

        trajectories.append(
            Trajectory(
                match_id=match_id,
                attacker=carrier_id,
                defender=defender_id,
                dt=dt,
                t=t,
                p_a=p_a,
                p_d=p_d,
                v_a=v_a,
                v_d=v_d,
                goal=goal,
                atk_path_length=atk_path,
                def_path_length=def_path,
                atk_displacement=atk_disp,
                def_displacement=def_disp,
                outcome=outcome,
                frame_idx=np.array(frame_idx),
                ball=np.array(ball_seg),
                attacker_team=attacker_team,
            )
        )
    if not return_track:
        return trajectories
    track = MatchTrack(
        match_id=match_id,
        fps=fps,
        period=np.array(track_period),
        t=np.array(track_t),
        ball=np.array(track_ball),
        owning_team=np.array(track_owner),
        home_attacks_ltr=np.array(track_ltr),
        home_team=home_team,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    return trajectories, track
