import numpy as np

from config import TIME_STEP_VISUAL
from models.occlusion import generate_target_samples
from models.simulator import is_occluded_by_smokes


class MultiAgentEnvironment:
    def __init__(self, uav_agents: list, missile_agents: list,
                 target_samples=None, time_step: float = TIME_STEP_VISUAL):
        self.uav_agents = uav_agents
        self.missile_agents = missile_agents
        self.target_samples = target_samples if target_samples is not None else generate_target_samples()
        self.time_step = time_step

    def reset(self):
        for uav in self.uav_agents:
            uav.reset()
        for missile in self.missile_agents:
            missile.reset()

    def step(self, current_time: float):
        for uav in self.uav_agents:
            uav.update(current_time)
        for missile in self.missile_agents:
            missile.update(current_time)

    def get_active_smoke_positions(self, current_time: float) -> list:
        active_positions = []
        for uav in self.uav_agents:
            active_positions.extend(uav.get_active_smoke_positions(current_time))
        return active_positions

    def evaluate_occlusion(self, start_time: float, end_time: float,
                           time_step: float = None) -> tuple:
        if time_step is None:
            time_step = self.time_step

        occlusion_details = {missile.missile_id: set() for missile in self.missile_agents}

        for current_time in np.arange(start_time, end_time, time_step):
            self.step(current_time)
            active_smokes = self.get_active_smoke_positions(current_time)

            for missile in self.missile_agents:
                if not missile.is_active:
                    continue

                if is_occluded_by_smokes(missile.current_position, active_smokes, self.target_samples):
                    occlusion_details[missile.missile_id].add(round(current_time, 3))

        occlusion_times = {
            missile_id: len(times) * time_step
            for missile_id, times in occlusion_details.items()
        }
        total_time = sum(occlusion_times.values())
        return occlusion_times, occlusion_details, total_time
