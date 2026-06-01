import numpy as np
import random
import time

import config
from models.kinematics import (
    calculate_3d_direction_vector,
    calculate_missile_position,
    calculate_uav_position,
    calculate_smoke_position,
    get_uav_speed_vector,
)
from models.occlusion import generate_target_samples, is_effectively_occluded
from models.simulator import is_occluded_by_smokes


class EnhancedGeneticAlgorithm:
    def __init__(self, problem_type: str, **kwargs):
        self.problem_type = problem_type
        self.kwargs = kwargs

        # 根据问题类型设置参数
        self.population_size = config.GA_POPULATION_SIZE[problem_type]
        self.max_iterations = config.GA_MAX_ITERATIONS[problem_type]
        self.mutation_rate = config.GA_MUTATION_RATE[problem_type]
        self.crossover_prob = config.GA_CROSSOVER_PROBABILITY
        self.elitism_size = config.GA_ELITISM_SIZE[problem_type]
        self.early_stop = config.GA_EARLY_STOP[problem_type]
        self.random_seed = kwargs.get("random_seed", config.DEFAULT_RANDOM_SEED)

        if "population_size" in kwargs:
            self.population_size = kwargs["population_size"]
        if "max_iterations" in kwargs:
            self.max_iterations = kwargs["max_iterations"]

        # 初始化目标采样点
        self.target_samples = generate_target_samples()

        # 初始化导弹信息
        if "missile_id" in kwargs:
            self.missile_id = kwargs["missile_id"]
            self.missile_initial_pos = np.array(config.MISSILE_INITIAL_POSITIONS[self.missile_id])
            self.missile_direction = calculate_3d_direction_vector(
                self.missile_initial_pos, np.array(config.DECOY_TARGET)
            )

        # 初始化无人机信息
        if "uav_ids" in kwargs:
            self.uav_ids = kwargs["uav_ids"]
            self.uav_initial_positions = {
                uav_id: np.array(config.UAV_INITIAL_POSITIONS[uav_id])
                for uav_id in self.uav_ids
            }

        # 问题三特定参数
        if problem_type == "problem3":
            self.uav_id = kwargs["uav_id"]
            self.uav_initial_pos = np.array(config.UAV_INITIAL_POSITIONS[self.uav_id])
            self.num_smokes = 3
            self.chromosome_length = 8  # [方向, 速度, t1, q1, t2, q2, t3, q3]

        # 问题四特定参数
        elif problem_type == "problem4":
            self.num_uavs = 3
            self.chromosome_length = 12  # 每架无人机4个参数

        # 问题五特定参数
        elif problem_type == "problem5":
            self.num_uavs = 5
            self.missile_ids = ["M1", "M2", "M3"]
            self.missile_info = {
                missile_id: {
                    "initial_pos": np.array(config.MISSILE_INITIAL_POSITIONS[missile_id]),
                    "direction": calculate_3d_direction_vector(
                        np.array(config.MISSILE_INITIAL_POSITIONS[missile_id]),
                        np.array(config.DECOY_TARGET)
                    ),
                    "flight_time": np.linalg.norm(
                        np.array(config.MISSILE_INITIAL_POSITIONS[missile_id]) - np.array(config.DECOY_TARGET)
                    ) / config.MISSILE_SPEED
                }
                for missile_id in self.missile_ids
            }
            self.chromosome_length = 75  # 5架无人机×3枚烟幕弹×5个参数

    def create_individual(self) -> np.ndarray:
        """创建个体"""
        if self.problem_type == "problem3":
            return self._create_problem3_individual()
        elif self.problem_type == "problem4":
            return self._create_problem4_individual()
        elif self.problem_type == "problem5":
            return self._create_problem5_individual()

    def _create_problem3_individual(self) -> np.ndarray:
        """创建问题三的个体"""
        direction = random.uniform(0, 360)
        speed = random.uniform(config.UAV_SPEED_MIN, config.UAV_SPEED_MAX)

        base_time = random.uniform(0.5, 2.0)
        interval = random.uniform(1.2, 3.0)

        individual = [direction, speed]

        for i in range(self.num_smokes):
            drop_time = base_time + i * interval
            if i == 0:
                explosion_delay = random.uniform(2.0, 4.0)
            elif i == 1:
                explosion_delay = random.uniform(3.0, 6.0)
            else:
                explosion_delay = random.uniform(4.0, 8.0)

            individual.extend([drop_time, explosion_delay])

        return np.array(individual)

    def _create_problem4_individual(self) -> np.ndarray:
        """创建问题四的个体"""
        individual = []
        intercept_times = [15, 30, 45]

        for i, uav_id in enumerate(self.uav_ids):
            uav_pos = self.uav_initial_positions[uav_id]
            intercept_time = intercept_times[i]

            missile_pos = calculate_missile_position(
                self.missile_initial_pos, self.missile_direction,
                config.MISSILE_SPEED, intercept_time
            )

            direction_vector = missile_pos - uav_pos
            direction = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
            if direction < 0:
                direction += 360

            if uav_pos[1] < 0:
                direction = random.uniform(30, 150)
            else:
                direction = random.uniform(-30, 30) + direction
                direction %= 360

            speed = random.uniform(config.UAV_SPEED_MIN, config.UAV_SPEED_MAX)
            drop_time = random.uniform(1, 20)
            explosion_delay = random.uniform(3, 10)

            individual.extend([direction, speed, drop_time, explosion_delay])

        return np.array(individual)

    def _create_problem5_individual(self) -> np.ndarray:
        """创建问题五的个体"""
        individual = []
        missile_ids = ["M1", "M2", "M3"]

        uav_assignments = {}
        for i, uav_id in enumerate(["FY1", "FY2", "FY3", "FY4", "FY5"]):
            if i < 3:
                uav_assignments[uav_id] = missile_ids[i]
            else:
                uav_assignments[uav_id] = random.choice(missile_ids)

        for uav_id in ["FY1", "FY2", "FY3", "FY4", "FY5"]:
            target_missile = uav_assignments[uav_id]
            uav_pos = np.array(config.UAV_INITIAL_POSITIONS[uav_id])

            missile_flight_time = self.missile_info[target_missile]["flight_time"]
            intercept_times = np.linspace(
                missile_flight_time * 0.3, missile_flight_time * 0.8, 3
            )

            intercept_pos = calculate_missile_position(
                self.missile_info[target_missile]["initial_pos"],
                self.missile_info[target_missile]["direction"],
                config.MISSILE_SPEED, intercept_times[0]
            )
            direction_vector = intercept_pos - uav_pos
            direction = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
            if direction < 0:
                direction += 360

            if uav_pos[1] < 0:
                direction = random.uniform(30, 150)

            speed = random.uniform(100, config.UAV_SPEED_MAX)

            for i in range(3):
                if i < len(intercept_times):
                    intercept_time = intercept_times[i]
                    distance = np.linalg.norm(intercept_pos[:2] - uav_pos[:2])
                    flight_time = distance / speed
                    drop_time = max(1.0, intercept_time - flight_time - random.uniform(3, 8))
                    explosion_delay = random.uniform(3, 8)
                    active = 1
                else:
                    drop_time = 0
                    explosion_delay = 0
                    active = 0

                individual.extend([direction, speed, drop_time, explosion_delay, active])

        return np.array(individual)

    def repair_individual(self, individual: np.ndarray) -> np.ndarray:
        """修复个体，确保满足约束"""
        if self.problem_type == "problem3":
            return self._repair_problem3_individual(individual)
        elif self.problem_type == "problem4":
            return self._repair_problem4_individual(individual)
        elif self.problem_type == "problem5":
            return self._repair_problem5_individual(individual)

    def _repair_problem3_individual(self, individual: np.ndarray) -> np.ndarray:
        """修复问题三的个体"""
        repaired = individual.copy()

        repaired[1] = np.clip(repaired[1], config.UAV_SPEED_MIN, config.UAV_SPEED_MAX)

        drop_times = repaired[2::2]
        drop_times = np.maximum(drop_times, 0)
        sorted_indices = np.argsort(drop_times)
        sorted_times = drop_times[sorted_indices]

        fixed_times = [max(0.1, sorted_times[0])]
        for i in range(1, len(sorted_times)):
            fixed_time = max(fixed_times[i-1] + config.MIN_DROP_INTERVAL, sorted_times[i])
            fixed_times.append(fixed_time)

        for i, idx in enumerate(sorted_indices):
            repaired[2 + idx * 2] = fixed_times[i]

        for i in range(3):
            repaired[3 + i * 2] = np.clip(repaired[3 + i * 2], 1.0, 10.0)

        return repaired

    def _repair_problem4_individual(self, individual: np.ndarray) -> np.ndarray:
        """修复问题四的个体"""
        repaired = individual.copy().reshape(3, 4)

        for i in range(3):
            repaired[i, 0] = np.clip(repaired[i, 0], 0, 360)
            repaired[i, 1] = np.clip(repaired[i, 1], config.UAV_SPEED_MIN, config.UAV_SPEED_MAX)
            repaired[i, 2] = max(0.5, repaired[i, 2])
            repaired[i, 3] = np.clip(repaired[i, 3], 1.0, 15.0)

            uav_id = self.uav_ids[i]
            uav_pos = self.uav_initial_positions[uav_id]
            drop_pos = calculate_uav_position(
                uav_pos, repaired[i, 1], repaired[i, 0], repaired[i, 2]
            )
            if drop_pos[1] < 0:
                if uav_pos[1] < 0:
                    repaired[i, 0] = random.uniform(30, 150)
                else:
                    repaired[i, 0] = random.uniform(0, 180)

        return repaired.flatten()

    def _repair_problem5_individual(self, individual: np.ndarray) -> np.ndarray:
        """修复问题五的个体"""
        repaired = individual.copy().reshape(5, 3, 5)

        for uav_idx in range(5):
            direction = np.clip(repaired[uav_idx, 0, 0], 0, 360)
            speed = np.clip(repaired[uav_idx, 0, 1], config.UAV_SPEED_MIN, config.UAV_SPEED_MAX)

            uav_id = ["FY1", "FY2", "FY3", "FY4", "FY5"][uav_idx]
            uav_pos = np.array(config.UAV_INITIAL_POSITIONS[uav_id])
            test_pos = calculate_uav_position(uav_pos, speed, direction, 5.0)
            if test_pos[1] < -100:
                if uav_pos[1] < 0:
                    direction = random.uniform(30, 150)
                else:
                    direction = random.uniform(-30, 30) + direction
                    direction %= 360

            for smoke_idx in range(3):
                repaired[uav_idx, smoke_idx, 0] = direction
                repaired[uav_idx, smoke_idx, 1] = speed

            drop_times = []
            for smoke_idx in range(3):
                _, _, drop_time, explosion_delay, active = repaired[uav_idx, smoke_idx]

                if active > 0.5:
                    drop_time = max(0.5, drop_time)
                    explosion_delay = np.clip(explosion_delay, 1.0, 15.0)
                    active = 1
                    drop_times.append((smoke_idx, drop_time))
                else:
                    drop_time = 0
                    explosion_delay = 0
                    active = 0

                repaired[uav_idx, smoke_idx, 2] = drop_time
                repaired[uav_idx, smoke_idx, 3] = explosion_delay
                repaired[uav_idx, smoke_idx, 4] = active

            if len(drop_times) > 1:
                drop_times.sort(key=lambda x: x[1])
                for i in range(1, len(drop_times)):
                    idx, time = drop_times[i]
                    prev_idx, prev_time = drop_times[i-1]
                    if time - prev_time < config.MIN_DROP_INTERVAL:
                        repaired[uav_idx, idx, 2] = prev_time + config.MIN_DROP_INTERVAL

        return repaired.flatten()

    def calculate_fitness(self, individual: np.ndarray) -> float:
        """计算适应度"""
        if self.problem_type == "problem3":
            return self._calculate_problem3_fitness(individual)
        elif self.problem_type == "problem4":
            return self._calculate_problem4_fitness(individual)
        elif self.problem_type == "problem5":
            return self._calculate_problem5_fitness(individual)

    def _calculate_problem3_fitness(self, individual: np.ndarray) -> float:
        """计算问题三的适应度"""
        constraint_violations = self._check_problem3_constraints(individual)
        if constraint_violations > 0:
            return max(0, self._calculate_raw_fitness(individual) - 200 * constraint_violations)

        return self._calculate_raw_fitness(individual)

    def _check_problem3_constraints(self, individual: np.ndarray) -> int:
        """检查问题三的约束"""
        violations = 0

        if not (config.UAV_SPEED_MIN <= individual[1] <= config.UAV_SPEED_MAX):
            violations += 1

        drop_times = individual[2::2]
        for i in range(1, len(drop_times)):
            if drop_times[i] - drop_times[i-1] < config.MIN_DROP_INTERVAL:
                violations += 1

        explosion_delays = individual[3::2]
        for delay in explosion_delays:
            if not (1.0 <= delay <= 10.0):
                violations += 1

        for time in drop_times:
            if time < 0:
                violations += 1

        return violations

    def _calculate_raw_fitness(self, individual: np.ndarray) -> float:
        """计算问题三的原始适应度（总遮蔽时间）"""
        direction, speed = individual[:2]
        smoke_params = individual[2:].reshape(3, 2)

        uav_speed_vector = get_uav_speed_vector(direction, speed)

        coarse_step = 0.2
        fine_step = config.TIME_STEP_SCAN
        max_time = 60

        potential_regions = []
        for t in np.arange(0, max_time, coarse_step):
            missile_pos = calculate_missile_position(
                self.missile_initial_pos, self.missile_direction,
                config.MISSILE_SPEED, t
            )

            if np.linalg.norm(missile_pos - np.array(config.TARGET_CENTER)) < 50:
                break

            for drop_time, explosion_delay in smoke_params:
                time_since_drop = t - drop_time
                time_since_explosion = t - drop_time - explosion_delay

                if (time_since_drop >= 0 and time_since_explosion >= 0 and
                    time_since_explosion <= config.SMOKE_EFFECTIVE_TIME):

                    drop_pos = calculate_uav_position(
                        self.uav_initial_pos, speed, direction, drop_time
                    )
                    smoke_pos = calculate_smoke_position(
                        drop_pos, uav_speed_vector, time_since_drop, time_since_explosion
                    )

                    if is_occluded_by_smokes(missile_pos, [smoke_pos], self.target_samples):
                        potential_regions.append(t)
                        break

        occluded_times = set()
        for coarse_t in potential_regions:
            start = max(0, coarse_t - coarse_step)
            end = min(max_time, coarse_t + coarse_step)

            for t in np.arange(start, end, fine_step):
                missile_pos = calculate_missile_position(
                    self.missile_initial_pos, self.missile_direction,
                    config.MISSILE_SPEED, t
                )

                for drop_time, explosion_delay in smoke_params:
                    time_since_drop = t - drop_time
                    time_since_explosion = t - drop_time - explosion_delay

                    if (time_since_drop >= 0 and time_since_explosion >= 0 and
                        time_since_explosion <= config.SMOKE_EFFECTIVE_TIME):

                        drop_pos = calculate_uav_position(
                            self.uav_initial_pos, speed, direction, drop_time
                        )
                        smoke_pos = calculate_smoke_position(
                            drop_pos, uav_speed_vector, time_since_drop, time_since_explosion
                        )

                        if smoke_pos[2] < 0:
                            continue

                        if is_occluded_by_smokes(missile_pos, [smoke_pos], self.target_samples):
                            occluded_times.add(round(t / fine_step) * fine_step)
                            break

        return len(occluded_times) * fine_step

    def _calculate_problem4_fitness(self, individual: np.ndarray) -> float:
        """计算问题四的适应度"""
        constraint_violations = self._check_problem4_constraints(individual)
        if constraint_violations > 0:
            return max(0, self._calculate_problem4_raw_fitness(individual) - 1000 * constraint_violations)

        return self._calculate_problem4_raw_fitness(individual)

    def _check_problem4_constraints(self, individual: np.ndarray) -> int:
        """检查问题四的约束"""
        violations = 0
        params = individual.reshape(3, 4)

        for i in range(3):
            direction, speed, drop_time, explosion_delay = params[i]

            if not (config.UAV_SPEED_MIN <= speed <= config.UAV_SPEED_MAX):
                violations += 1

            if not (0 <= direction <= 360):
                violations += 1

            if drop_time < 0:
                violations += 1

            if not (1.0 <= explosion_delay <= 15.0):
                violations += 1

        return violations

    def _calculate_problem4_raw_fitness(self, individual: np.ndarray) -> float:
        """计算问题四的原始适应度"""
        params = individual.reshape(3, 4)
        occluded_times = set()
        max_time = 70

        for t in np.arange(0, max_time, config.TIME_STEP_VISUAL):
            missile_pos = calculate_missile_position(
                self.missile_initial_pos, self.missile_direction,
                config.MISSILE_SPEED, t
            )

            for i in range(3):
                direction, speed, drop_time, explosion_delay = params[i]
                uav_id = self.uav_ids[i]
                uav_initial_pos = self.uav_initial_positions[uav_id]

                time_since_drop = t - drop_time
                time_since_explosion = t - drop_time - explosion_delay

                if (time_since_drop >= 0 and time_since_explosion >= 0 and
                    time_since_explosion <= config.SMOKE_EFFECTIVE_TIME):

                    drop_pos = calculate_uav_position(
                        uav_initial_pos, speed, direction, drop_time
                    )
                    uav_speed_vector = get_uav_speed_vector(direction, speed)
                    smoke_pos = calculate_smoke_position(
                        drop_pos, uav_speed_vector, time_since_drop, time_since_explosion
                    )

                    if smoke_pos[2] < 0:
                        continue

                    if is_occluded_by_smokes(missile_pos, [smoke_pos], self.target_samples):
                        occluded_times.add(round(t, 1))
                        break

        return len(occluded_times) * config.TIME_STEP_VISUAL

    def _calculate_problem5_fitness(self, individual: np.ndarray) -> float:
        """计算问题五的适应度"""
        constraint_violations = self._check_problem5_constraints(individual)
        if constraint_violations > 0:
            raw_fitness, _ = self._calculate_problem5_raw_fitness(individual)
            return max(0, raw_fitness - 100 * constraint_violations)

        raw_fitness, occlusion_details = self._calculate_problem5_raw_fitness(individual)

        occlusion_times = [len(times) * config.TIME_STEP_VISUAL for times in occlusion_details.values()]
        balance_bonus = -np.var(occlusion_times)

        synergy_bonus = 0
        for missile_id in self.missile_ids:
            for t in occlusion_details[missile_id]:
                overlap_count = 0
                for uav_idx in range(5):
                    for smoke_idx in range(3):
                        params = individual.reshape(5, 3, 5)[uav_idx, smoke_idx]
                        _, _, drop_time, explosion_delay, active = params

                        if active == 0:
                            continue

                        time_since_explosion = t - drop_time - explosion_delay
                        if 0 <= time_since_explosion <= config.SMOKE_EFFECTIVE_TIME:
                            overlap_count += 1

                synergy_bonus += min(overlap_count, 1) * np.log(1 + overlap_count)

        return raw_fitness + balance_bonus + synergy_bonus * 0.1

    def _check_problem5_constraints(self, individual: np.ndarray) -> int:
        """检查问题五的约束"""
        violations = 0
        params = individual.reshape(5, 3, 5)

        for uav_idx in range(5):
            direction, speed = params[uav_idx, 0, :2]

            if not (config.UAV_SPEED_MIN <= speed <= config.UAV_SPEED_MAX):
                violations += 1

            if not (0 <= direction <= 360):
                violations += 1

            drop_times = []
            for smoke_idx in range(3):
                _, _, drop_time, explosion_delay, active = params[uav_idx, smoke_idx]

                if active > 0.5:
                    if drop_time < 0:
                        violations += 1

                    if not (1.0 <= explosion_delay <= 15.0):
                        violations += 1

                    drop_times.append(drop_time)

            if len(drop_times) > 1:
                drop_times.sort()
                for i in range(1, len(drop_times)):
                    if drop_times[i] - drop_times[i-1] < config.MIN_DROP_INTERVAL:
                        violations += 1

        return violations

    def _calculate_problem5_raw_fitness(self, individual: np.ndarray) -> tuple:
        """计算问题五的原始适应度和详细遮蔽信息"""
        params = individual.reshape(5, 3, 5)
        occlusion_details = {missile_id: set() for missile_id in self.missile_ids}
        max_time = 80

        for t in np.arange(0, max_time, config.TIME_STEP_VISUAL):
            for missile_id in self.missile_ids:
                missile_info = self.missile_info[missile_id]

                if t > missile_info["flight_time"]:
                    continue

                missile_pos = calculate_missile_position(
                    missile_info["initial_pos"], missile_info["direction"],
                    config.MISSILE_SPEED, t
                )

                for uav_idx in range(5):
                    uav_id = ["FY1", "FY2", "FY3", "FY4", "FY5"][uav_idx]
                    uav_initial_pos = np.array(config.UAV_INITIAL_POSITIONS[uav_id])

                    for smoke_idx in range(3):
                        direction, speed, drop_time, explosion_delay, active = params[uav_idx, smoke_idx]

                        if active == 0:
                            continue

                        time_since_drop = t - drop_time
                        time_since_explosion = t - drop_time - explosion_delay

                        if (time_since_drop >= 0 and time_since_explosion >= 0 and
                            time_since_explosion <= config.SMOKE_EFFECTIVE_TIME):

                            drop_pos = calculate_uav_position(
                                uav_initial_pos, speed, direction, drop_time
                            )
                            uav_speed_vector = get_uav_speed_vector(direction, speed)
                            smoke_pos = calculate_smoke_position(
                                drop_pos, uav_speed_vector, time_since_drop, time_since_explosion
                            )

                            if smoke_pos[2] < 0:
                                continue

                            if is_occluded_by_smokes(missile_pos, [smoke_pos], self.target_samples):
                                occlusion_details[missile_id].add(round(t, 1))
                                break
                    else:
                        continue
                    break

        total_time = sum(len(times) * config.TIME_STEP_VISUAL for times in occlusion_details.values())
        return total_time, occlusion_details

    def mutate(self, individual: np.ndarray, generation: int) -> np.ndarray:
        """自适应变异"""
        mutated = individual.copy()
        mutation_strength = 1.0 - 0.8 * (generation / self.max_iterations)

        if self.problem_type == "problem3":
            if random.random() < self.mutation_rate:
                mutated[0] = (mutated[0] + random.gauss(0, 20 * mutation_strength)) % 360

            if random.random() < self.mutation_rate:
                mutated[1] += random.gauss(0, 8 * mutation_strength)

            for i in range(3):
                if random.random() < self.mutation_rate:
                    mutated[2 + i * 2] += random.gauss(0, 0.8 * mutation_strength)
                if random.random() < self.mutation_rate:
                    mutated[3 + i * 2] += random.gauss(0, 0.4 * mutation_strength)

        elif self.problem_type == "problem4":
            for i in range(12):
                if random.random() < self.mutation_rate:
                    param_type = i % 4
                    if param_type == 0:
                        mutated[i] = (mutated[i] + random.gauss(0, 30 * mutation_strength)) % 360
                    elif param_type == 1:
                        mutated[i] += random.gauss(0, 15 * mutation_strength)
                    elif param_type == 2:
                        mutated[i] += random.gauss(0, 3 * mutation_strength)
                    else:
                        mutated[i] += random.gauss(0, 2 * mutation_strength)

        elif self.problem_type == "problem5":
            mutation_rate = self.mutation_rate * (1.2 - generation / self.max_iterations)
            for i in range(75):
                if random.random() < mutation_rate:
                    param_type = i % 5
                    if param_type == 0:
                        mutated[i] = (mutated[i] + random.gauss(0, 45 * mutation_strength)) % 360
                    elif param_type == 1:
                        mutated[i] += random.gauss(0, 20 * mutation_strength)
                    elif param_type == 2:
                        mutated[i] += random.gauss(0, 5 * mutation_strength)
                    elif param_type == 3:
                        mutated[i] += random.gauss(0, 3 * mutation_strength)
                    else:
                        if random.random() < 0.15:
                            mutated[i] = 1 - mutated[i]

        return self.repair_individual(mutated)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """智能交叉"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        if self.problem_type == "problem3":
            alpha = random.uniform(0.3, 0.7)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

        elif self.problem_type == "problem4":
            child1 = parent1.copy()
            child2 = parent2.copy()

            for i in range(3):
                if random.random() < 0.5:
                    start = i * 4
                    end = start + 4
                    child1[start:end] = parent2[start:end]
                    child2[start:end] = parent1[start:end]

        elif self.problem_type == "problem5":
            crossover_type = random.choice(['uav', 'smoke', 'point'])

            if crossover_type == 'uav':
                child1 = parent1.copy()
                child2 = parent2.copy()

                for uav_idx in range(5):
                    if random.random() < 0.5:
                        start = uav_idx * 15
                        end = start + 15
                        child1[start:end] = parent2[start:end]
                        child2[start:end] = parent1[start:end]

            elif crossover_type == 'smoke':
                child1 = parent1.copy()
                child2 = parent2.copy()

                for uav_idx in range(5):
                    for smoke_idx in range(3):
                        if random.random() < 0.5:
                            start = uav_idx * 15 + smoke_idx * 5
                            end = start + 5
                            child1[start:end] = parent2[start:end]
                            child2[start:end] = parent1[start:end]

            else:
                crossover_point = random.randint(1, 74)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return self.repair_individual(child1), self.repair_individual(child2)

    def tournament_selection(self, population: list, fitness_values: list,
                            tournament_size: int = 5) -> np.ndarray:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()

    def optimize(self) -> dict:
        """执行遗传算法优化"""
        print(f"开始{self.problem_type}优化...")
        print(f"种群规模: {self.population_size}, 最大迭代次数: {self.max_iterations}")
        print("=" * 70)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        start_time = time.time()

        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(self.repair_individual(individual))

        best_fitness = -float('inf')
        best_individual = None
        generations_without_improvement = 0

        fitness_history = []
        best_fitness_history = []

        for generation in range(self.max_iterations):
            fitness_values = [self.calculate_fitness(ind) for ind in population]

            avg_fitness = np.mean(fitness_values)
            current_best_fitness = np.max(fitness_values)
            fitness_history.append(avg_fitness)
            best_fitness_history.append(current_best_fitness)

            current_best_index = np.argmax(fitness_values)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_index].copy()
                generations_without_improvement = 0

                if self.problem_type == "problem5":
                    total_time, _ = self._calculate_problem5_raw_fitness(best_individual)
                    print(f"第{generation:3d}代: 总遮蔽时间={total_time:.2f}s, "
                          f"适应度={current_best_fitness:.2f} ⭐")
                else:
                    print(f"第{generation:3d}代: 最佳遮蔽时间={current_best_fitness:.3f}s, "
                          f"平均={avg_fitness:.3f}s ⭐")
            else:
                generations_without_improvement += 1

                if generation % 20 == 0:
                    if self.problem_type == "problem5" and best_individual is not None:
                        total_time, _ = self._calculate_problem5_raw_fitness(best_individual)
                        print(f"第{generation:3d}代: 总遮蔽时间={total_time:.2f}s, "
                              f"适应度={current_best_fitness:.2f}")
                    else:
                        print(f"第{generation:3d}代: 最佳={current_best_fitness:.3f}s, "
                              f"平均={avg_fitness:.3f}s")

            if generations_without_improvement >= self.early_stop:
                print(f"\n连续{self.early_stop}代无改善，提前结束优化")
                break

            new_population = []

            elite_indices = np.argsort(fitness_values)[-self.elitism_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)

                child1, child2 = self.crossover(parent1, parent2)

                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child1, generation))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2, generation))

            population = new_population

        end_time = time.time()
        optimization_time = end_time - start_time

        print(f"\n优化完成! 耗时: {optimization_time:.2f}秒")

        if best_individual is None:
            return {"success": False}

        result = {
            "success": True,
            "best_fitness": best_fitness,
            "best_individual": best_individual,
            "fitness_history": fitness_history,
            "best_fitness_history": best_fitness_history,
            "optimization_time": optimization_time
        }

        if self.problem_type == "problem3":
            result.update(self._get_problem3_results(best_individual))
        elif self.problem_type == "problem4":
            result.update(self._get_problem4_results(best_individual))
        elif self.problem_type == "problem5":
            result.update(self._get_problem5_results(best_individual))

        return result

    def _get_problem3_results(self, individual: np.ndarray) -> dict:
        """获取问题三的详细结果"""
        direction, speed = individual[:2]
        smoke_params = individual[2:].reshape(3, 2)

        uav_speed_vector = get_uav_speed_vector(direction, speed)

        smoke_details = []
        for i, (drop_time, explosion_delay) in enumerate(smoke_params):
            drop_pos = calculate_uav_position(
                self.uav_initial_pos, speed, direction, drop_time
            )
            explosion_pos = calculate_smoke_position(
                drop_pos, uav_speed_vector, explosion_delay, 0
            )

            smoke_details.append({
                "smoke_id": i + 1,
                "drop_time": drop_time,
                "explosion_delay": explosion_delay,
                "drop_position": drop_pos,
                "explosion_position": explosion_pos
            })

        return {
            "uav_direction": direction,
            "uav_speed": speed,
            "smoke_details": smoke_details,
            "total_occlusion_time": self._calculate_raw_fitness(individual)
        }

    def _get_problem4_results(self, individual: np.ndarray) -> dict:
        """获取问题四的详细结果"""
        params = individual.reshape(3, 4)

        uav_details = []
        for i, uav_id in enumerate(self.uav_ids):
            direction, speed, drop_time, explosion_delay = params[i]
            uav_initial_pos = self.uav_initial_positions[uav_id]

            drop_pos = calculate_uav_position(
                uav_initial_pos, speed, direction, drop_time
            )
            uav_speed_vector = get_uav_speed_vector(direction, speed)
            explosion_pos = calculate_smoke_position(
                drop_pos, uav_speed_vector, explosion_delay, 0
            )

            uav_details.append({
                "uav_id": uav_id,
                "direction": direction,
                "speed": speed,
                "drop_time": drop_time,
                "explosion_delay": explosion_delay,
                "drop_position": drop_pos,
                "explosion_position": explosion_pos
            })

        return {
            "uav_details": uav_details,
            "total_occlusion_time": self._calculate_problem4_raw_fitness(individual)
        }

    def _get_problem5_results(self, individual: np.ndarray) -> dict:
        """获取问题五的详细结果"""
        params = individual.reshape(5, 3, 5)
        total_time, occlusion_details = self._calculate_problem5_raw_fitness(individual)

        uav_details = []
        total_smokes = 0

        for uav_idx, uav_id in enumerate(["FY1", "FY2", "FY3", "FY4", "FY5"]):
            direction, speed = params[uav_idx, 0, :2]
            uav_initial_pos = np.array(config.UAV_INITIAL_POSITIONS[uav_id])

            smoke_details = []
            active_smokes = 0

            for smoke_idx in range(3):
                _, _, drop_time, explosion_delay, active = params[uav_idx, smoke_idx]

                if active > 0.5:
                    active_smokes += 1
                    total_smokes += 1

                    drop_pos = calculate_uav_position(
                        uav_initial_pos, speed, direction, drop_time
                    )
                    uav_speed_vector = get_uav_speed_vector(direction, speed)
                    explosion_pos = calculate_smoke_position(
                        drop_pos, uav_speed_vector, explosion_delay, 0
                    )

                    smoke_details.append({
                        "smoke_id": smoke_idx + 1,
                        "drop_time": drop_time,
                        "explosion_delay": explosion_delay,
                        "drop_position": drop_pos,
                        "explosion_position": explosion_pos
                    })

            uav_details.append({
                "uav_id": uav_id,
                "direction": direction,
                "speed": speed,
                "active_smokes": active_smokes,
                "smoke_details": smoke_details
            })

        missile_occlusion_times = {
            missile_id: len(times) * config.TIME_STEP_VISUAL
            for missile_id, times in occlusion_details.items()
        }

        return {
            "uav_details": uav_details,
            "total_occlusion_time": total_time,
            "missile_occlusion_times": missile_occlusion_times,
            "occlusion_details": {missile_id: sorted(list(times)) for missile_id, times in occlusion_details.items()},
            "total_smokes_used": total_smokes
        }
