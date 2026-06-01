import numpy as np
from scipy.optimize import differential_evolution

import config
from models.kinematics import (
    calculate_uav_position,
    calculate_missile_position,
    calculate_smoke_position,
    get_uav_speed_vector,
    calculate_3d_direction_vector,
)
from models.occlusion import generate_target_samples
from models.simulator import compute_occlusion_duration


class DifferentialEvolutionOptimizer:
    def __init__(self, uav_id: str, missile_id: str,
                 popsize: int = None, maxiter: int = None, seed: int = None):
        self.uav_id = uav_id
        self.missile_id = missile_id
        self.uav_initial_pos = np.array(config.UAV_INITIAL_POSITIONS[uav_id])
        self.missile_initial_pos = np.array(config.MISSILE_INITIAL_POSITIONS[missile_id])
        self.missile_direction = calculate_3d_direction_vector(
            self.missile_initial_pos, np.array(config.DECOY_TARGET)
        )
        self.target_samples = generate_target_samples()
        self.popsize = popsize if popsize is not None else config.DE_POPULATION_SIZE
        self.maxiter = maxiter if maxiter is not None else config.DE_MAX_ITERATIONS
        self.seed = seed if seed is not None else config.DEFAULT_RANDOM_SEED

        self.bounds = [
            (config.UAV_SPEED_MIN, config.UAV_SPEED_MAX),
            (0.1, 4.0),
            (0.5, 4.0),
            (0, 2 * np.pi)
        ]

    def objective_function(self, params: np.ndarray) -> float:
        """目标函数：最小化负的遮蔽时间"""
        speed, drop_delay, explosion_delay, direction_angle = params

        # 约束检查
        if drop_delay + explosion_delay > 10:
            return 1000 + (drop_delay + explosion_delay - 10) * 0.5

        flight_distance = speed * drop_delay
        if flight_distance > 1500:
            return 1000 + (flight_distance - 1500) / 1000

        # 计算投放位置
        drop_position = calculate_uav_position(
            self.uav_initial_pos, speed, np.degrees(direction_angle), drop_delay
        )

        # 计算起爆位置
        uav_speed_vector = get_uav_speed_vector(np.degrees(direction_angle), speed)
        explosion_position = calculate_smoke_position(
            drop_position, uav_speed_vector, explosion_delay, 0
        )

        if explosion_position[2] <= 0:
            return 1000

        smoke_event = {
            "drop_position": drop_position,
            "uav_speed_vector": get_uav_speed_vector(np.degrees(direction_angle), speed),
            "drop_time": drop_delay,
            "explosion_delay": explosion_delay,
        }

        occlusion_time = compute_occlusion_duration(
            self.missile_initial_pos,
            self.missile_direction,
            config.MISSILE_SPEED,
            [smoke_event],
            self.target_samples,
            start_time=drop_delay + explosion_delay,
            end_time=drop_delay + explosion_delay + config.SMOKE_EFFECTIVE_TIME,
            time_step=config.TIME_STEP_SCAN
        )

        return -occlusion_time  # 转化为最小化问题

    def optimize(self) -> dict:
        """执行优化"""
        best_result = None
        best_occlusion_time = 0

        for seed in [self.seed, self.seed + 1]:
            result = differential_evolution(
                self.objective_function,
                self.bounds,
                seed=seed,
                popsize=self.popsize,
                maxiter=self.maxiter,
                disp=False,
                polish=True,
                atol=config.DE_TOLERANCE,
                tol=config.DE_TOLERANCE
            )

            occlusion_time = -result.fun
            if occlusion_time > best_occlusion_time:
                best_occlusion_time = occlusion_time
                best_result = result

        if best_result is None:
            return {"success": False}

        speed, drop_delay, explosion_delay, direction_angle = best_result.x

        return {
            "success": True,
            "occlusion_time": best_occlusion_time,
            "uav_speed": speed,
            "drop_delay": drop_delay,
            "explosion_delay": explosion_delay,
            "direction_angle": np.degrees(direction_angle),
            "drop_position": calculate_uav_position(
                self.uav_initial_pos, speed, np.degrees(direction_angle), drop_delay
            ),
            "explosion_position": calculate_smoke_position(
                calculate_uav_position(
                    self.uav_initial_pos, speed, np.degrees(direction_angle), drop_delay
                ),
                get_uav_speed_vector(np.degrees(direction_angle), speed),
                explosion_delay,
                0
            )
        }
