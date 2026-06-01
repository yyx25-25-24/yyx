import numpy as np
import config


def calculate_2d_direction_vector(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """计算水平面内的单位方向向量"""
    direction = np.array([end[0] - start[0], end[1] - start[1], 0.0])
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return direction / norm


def calculate_3d_direction_vector(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """计算三维空间内的单位方向向量"""
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return direction / norm


def calculate_uav_position(initial_pos: np.ndarray, speed: float,
                          direction_angle: float, time: float) -> np.ndarray:
    """计算无人机在时刻t的位置（等高度水平飞行）"""
    direction_rad = np.radians(direction_angle)
    direction = np.array([np.cos(direction_rad), np.sin(direction_rad), 0.0])
    position = initial_pos + speed * time * direction
    position[2] = initial_pos[2]  # 保持高度不变
    return position


def calculate_missile_position(initial_pos: np.ndarray, direction: np.ndarray,
                              speed: float, time: float) -> np.ndarray:
    """计算导弹在时刻t的位置"""
    return initial_pos + speed * time * direction


def calculate_smoke_position(drop_position: np.ndarray, uav_speed_vector: np.ndarray,
                            time_since_drop: float, time_since_explosion: float) -> np.ndarray:
    """计算烟幕弹在时刻t的位置

    Args:
        drop_position: 投放位置
        uav_speed_vector: 无人机速度向量
        time_since_drop: 投放后经过的时间
        time_since_explosion: 爆炸后经过的时间（负数表示未爆炸）
    """
    if time_since_explosion < 0:
        # 平抛运动阶段
        x = drop_position[0] + uav_speed_vector[0] * time_since_drop
        y = drop_position[1] + uav_speed_vector[1] * time_since_drop
        z = drop_position[2] - 0.5 * config.GRAVITY * time_since_drop ** 2
    else:
        # 爆炸后下沉阶段
        explosion_time = time_since_drop - time_since_explosion
        x_explosion = drop_position[0] + uav_speed_vector[0] * explosion_time
        y_explosion = drop_position[1] + uav_speed_vector[1] * explosion_time
        z_explosion = drop_position[2] - 0.5 * config.GRAVITY * explosion_time ** 2

        x = x_explosion
        y = y_explosion
        z = z_explosion - config.SMOKE_SINK_SPEED * time_since_explosion

    return np.array([x, y, z])


def get_uav_speed_vector(direction_angle: float, speed: float) -> np.ndarray:
    """获取无人机速度向量"""
    direction_rad = np.radians(direction_angle)
    return np.array([
        speed * np.cos(direction_rad),
        speed * np.sin(direction_rad),
        0.0
    ])
