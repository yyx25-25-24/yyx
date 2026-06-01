import numpy as np
import config
from models.kinematics import calculate_missile_position, calculate_3d_direction_vector


class MissileAgent:
    def __init__(self, missile_id: str):
        self.missile_id = missile_id
        self.initial_position = np.array(config.MISSILE_INITIAL_POSITIONS[missile_id])
        self.current_position = self.initial_position.copy()

        # 飞行参数
        self.speed = config.MISSILE_SPEED
        self.direction = calculate_3d_direction_vector(
            self.initial_position, np.array(config.DECOY_TARGET)
        )

        # 总飞行时间
        self.total_flight_time = np.linalg.norm(
            self.initial_position - np.array(config.DECOY_TARGET)
        ) / self.speed

        # 状态
        self.is_active = True

    def update(self, current_time: float):
        """更新导弹状态"""
        if current_time > self.total_flight_time:
            self.is_active = False
            return

        self.current_position = calculate_missile_position(
            self.initial_position, self.direction, self.speed, current_time
        )

    def reset(self):
        """重置导弹状态"""
        self.current_position = self.initial_position.copy()
        self.is_active = True
