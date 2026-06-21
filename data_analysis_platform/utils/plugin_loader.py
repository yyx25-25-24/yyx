# utils/plugin_loader.py
import importlib
import os
from pathlib import Path
from tool_registry import registry

class PluginLoader:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins = {}
    
    def discover_plugins(self):
        """自动发现并加载插件"""
        if not self.plugin_dir.exists():
            return
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            module_name = f"plugins.{plugin_file.stem}"
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'register'):
                    module.register(registry)  # 使用全局 registry
                    self.plugins[plugin_file.stem] = module
                    print(f"✅ 加载插件: {plugin_file.stem}")
            except Exception as e:
                print(f"❌ 插件加载失败 {plugin_file.stem}: {e}")