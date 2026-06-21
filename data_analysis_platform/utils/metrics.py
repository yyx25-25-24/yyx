# utils/metrics.py
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timings = defaultdict(list)
    
    def record_call(self, tool_name: str, duration: float, success: bool):
        self.counters[f"{tool_name}_calls"] += 1
        self.counters[f"{tool_name}_{'success' if success else 'failure'}"] += 1
        self.timings[tool_name].append(duration)
    
    def get_stats(self, tool_name: str) -> dict:
        times = self.timings.get(tool_name, [])
        return {
            "total_calls": self.counters.get(f"{tool_name}_calls", 0),
            "avg_duration": sum(times) / len(times) if times else 0,
            "max_duration": max(times) if times else 0,
        }