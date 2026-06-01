import numpy as np
import config
from models.kinematics import calculate_uav_position, calculate_smoke_position, get_uav_speed_vector


class UAVAgent:
    def __init__(self, uav_id: str):
        self.uav_id = uav_id
        self.initial_position = np.array(config.UAV_INITIAL_POSITIONS[uav_id])
        self.current_position = self.initial_position.copy()

        # 飞行参数
        self.speed = 0.0
        self.direction_angle = 0.0  # 与x轴夹角，度
        self.speed_vector = np.array([0.0, 0.0, 0.0])

        # 投弹计划
        self.smoke_plan = []  # 列表，每个元素是(drop_time, explosion_delay)
        self.dropped_smokes = []  # 已投放的烟幕弹信息

        # 烟幕云团状态
        self.smoke_clouds = []  # 列表，每个元素是(explosion_time, explosion_position)

    def set_flight_parameters(self, speed: float, direction_angle: float):
        """设置飞行参数"""
        self.speed = speed
        self.direction_angle = direction_angle
        self.speed_vector = get_uav_speed_vector(direction_angle, speed)

    def add_smoke_plan(self, drop_time: float, explosion_delay: float):
        """添加投弹计划"""
        self.smoke_plan.append((drop_time, explosion_delay))
        # 按投放时间排序
        self.smoke_plan.sort(key=lambda x: x[0])

    def update(self, current_time: float):
        """更新无人机状态和烟幕云团状态"""
        # 更新无人机位置
        self.current_position = calculate_uav_position(
            self.initial_position, self.speed, self.direction_angle, current_time
        )

        # 检查是否需要投弹（投弹时刻已过且尚未记录）
        for drop_time, explosion_delay in self.smoke_plan:
            if current_time >= drop_time and (drop_time, explosion_delay) not in self.dropped_smokes:
                self._drop_smoke(drop_time, explosion_delay)
                self.dropped_smokes.append((drop_time, explosion_delay))

        # 移除过期的烟幕云团（超过有效时间）
        self.smoke_clouds = [
            cloud for cloud in self.smoke_clouds
            if current_time - cloud[0] <= config.SMOKE_EFFECTIVE_TIME
        ]

    def _drop_smoke(self, drop_time: float, explosion_delay: float):
        """投放烟幕弹"""
        drop_position = calculate_uav_position(
            self.initial_position, self.speed, self.direction_angle, drop_time
        )
        explosion_time = drop_time + explosion_delay
        explosion_position = calculate_smoke_position(
            drop_position, self.speed_vector, explosion_delay, 0
        )

        self.smoke_clouds.append((explosion_time, explosion_position))

    def get_active_smoke_positions(self, current_time: float) -> list:
        """获取当前时刻所有有效烟幕云团的位置"""
        active_positions = []

        for explosion_time, explosion_pos in self.smoke_clouds:
            if current_time >= explosion_time:
                time_since_explosion = current_time - explosion_time
                smoke_pos = explosion_pos.copy()
                smoke_pos[2] -= config.SMOKE_SINK_SPEED * time_since_explosion

                if smoke_pos[2] >= 0:
                    active_positions.append(smoke_pos)

        return active_positions

    def reset(self):
        """重置无人机状态"""
        self.current_position = self.initial_position.copy()
        self.speed = 0.0
        self.direction_angle = 0.0
        self.speed_vector = np.array([0.0, 0.0, 0.0])
        self.smoke_plan = []
        self.dropped_smokes = []
        self.smoke_clouds = []
