"""
工具注册中心 - 统一管理所有可用的分析工具
支持动态注册、发现和执行工具
"""
import inspect
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ToolInfo:
    """工具元信息"""
    name: str
    description: str
    func: Callable
    category: str = "general"  # 分类：analysis, visualization, export, database等
    requires_df: bool = True   # 是否需要DataFrame输入
    tags: List[str] = field(default_factory=list)


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, ToolInfo] = {}
    
    def register(self, 
                 name: str = None,
                 description: str = "",
                 category: str = "general",
                 requires_df: bool = True,
                 tags: List[str] = None):
        """
        装饰器：注册工具
        
        用法：
        @registry.register(name="correlation_analysis", description="相关性分析")
        def analyze_correlation(df, columns):
            ...
        """
        def decorator(func: Callable):
            tool_name = name or func.__name__
            self._tools[tool_name] = ToolInfo(
                name=tool_name,
                description=description or func.__doc__ or "",
                func=func,
                category=category,
                requires_df=requires_df,
                tags=tags or []
            )
            return func
        return decorator
    
    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """获取工具信息"""
        return self._tools.get(name)
    
    def list_tools(self, category: str = None) -> List[ToolInfo]:
        """列出所有工具，可按分类过滤"""
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())
    
    def execute(self, tool_name: str, *args, **kwargs) -> Any:
        """执行指定工具"""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"工具 '{tool_name}' 未注册")
        return tool.func(*args, **kwargs)
    
    def search_tools(self, keyword: str) -> List[ToolInfo]:
        """根据关键词搜索工具"""
        keyword_lower = keyword.lower()
        results = []
        for tool in self._tools.values():
            if (keyword_lower in tool.name.lower() or 
                keyword_lower in tool.description.lower() or
                any(keyword_lower in tag.lower() for tag in tool.tags)):
                results.append(tool)
        return results


# 全局工具注册实例
registry = ToolRegistry()