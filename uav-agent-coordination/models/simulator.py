import json
import numpy as np

from config import SMOKE_EFFECTIVE_TIME, TIME_STEP_VISUAL
from models.kinematics import calculate_missile_position, calculate_smoke_position
from models.occlusion import is_effectively_occluded


def is_occluded_by_smokes(missile_position: np.ndarray,
                          smoke_positions: list,
                          target_samples: np.ndarray) -> bool:
    for smoke_pos in smoke_positions:
        if is_effectively_occluded(missile_position, smoke_pos, target_samples):
            return True
    return False


def compute_smoke_position_at_time(drop_position: np.ndarray,
                                   uav_speed_vector: np.ndarray,
                                   drop_time: float,
                                   explosion_delay: float,
                                   current_time: float) -> np.ndarray:
    time_since_drop = current_time - drop_time
    if time_since_drop < 0:
        return None
    time_since_explosion = time_since_drop - explosion_delay
    smoke_pos = calculate_smoke_position(
        drop_position, uav_speed_vector,
        time_since_drop, time_since_explosion
    )
    if smoke_pos[2] < 0:
        return None
    return smoke_pos


def compute_occluded_time_points(missile_initial_pos: np.ndarray,
                                 missile_direction: np.ndarray,
                                 missile_speed: float,
                                 smoke_events: list,
                                 target_samples: np.ndarray,
                                 start_time: float = 0.0,
                                 end_time: float = None,
                                 time_step: float = TIME_STEP_VISUAL) -> set:
    if end_time is None:
        end_time = start_time + SMOKE_EFFECTIVE_TIME

    occluded_times = set()
    for t in np.arange(start_time, end_time, time_step):
        missile_pos = calculate_missile_position(
            missile_initial_pos, missile_direction, missile_speed, t
        )

        smoke_positions = []
        for smoke in smoke_events:
            smoke_pos = compute_smoke_position_at_time(
                smoke["drop_position"], smoke["uav_speed_vector"],
                smoke["drop_time"], smoke["explosion_delay"], t
            )
            if smoke_pos is not None:
                smoke_positions.append(smoke_pos)

        if smoke_positions and is_occluded_by_smokes(missile_pos, smoke_positions, target_samples):
            occluded_times.add(round(t, 3))

    return occluded_times


def compute_occlusion_duration(missile_initial_pos: np.ndarray,
                                missile_direction: np.ndarray,
                                missile_speed: float,
                                smoke_events: list,
                                target_samples: np.ndarray,
                                start_time: float = 0.0,
                                end_time: float = None,
                                time_step: float = TIME_STEP_VISUAL) -> float:
    occluded_times = compute_occluded_time_points(
        missile_initial_pos, missile_direction, missile_speed,
        smoke_events, target_samples,
        start_time=start_time, end_time=end_time, time_step=time_step
    )
    return len(occluded_times) * time_step


def serialize_result(result: dict) -> str:
    def _encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"不可序列化对象: {type(obj)}")
    return json.dumps(result, default=_encoder, ensure_ascii=False, indent=2)
