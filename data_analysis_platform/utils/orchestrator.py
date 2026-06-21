# utils/orchestrator.py
from typing import List, Dict,Any
from tool_registry import ToolRegistry
class ToolOrchestrator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def execute_pipeline(self, pipeline: List[Dict]) -> Any:
        """
        执行工具流水线
        
        pipeline = [
            {"tool": "data_preprocessing", "args": {...}},
            {"tool": "statistical_analysis", "args": {...}},
            {"tool": "chart_generation", "args": {...}}
        ]
        """
        result = None
        for step in pipeline:
            tool_name = step["tool"]
            args = step.get("args", {})
            
            if result is not None and "df" not in args:
                args["df"] = result
            
            result = self.registry.execute(tool_name, **args)
        
        return result