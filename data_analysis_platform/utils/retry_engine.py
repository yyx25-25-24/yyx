"""
异常重试引擎 - 统一的错误处理和自动重试机制
"""
import time
import logging
from typing import Callable, Any, Optional, Tuple, Type
from functools import wraps

logger = logging.getLogger(__name__)


class RetryConfig:
    """重试配置"""
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 max_delay: float = 60.0,
                 retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
                 on_retry_callback: Callable = None):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions
        self.on_retry_callback = on_retry_callback


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """计算指数退避延迟时间"""
    delay = config.base_delay * (config.backoff_factor ** attempt)
    return min(delay, config.max_delay)


def retry_with_config(config: RetryConfig = None):
    """
    重试装饰器
    
    用法：
    @retry_with_config(RetryConfig(max_retries=3, base_delay=1.0))
    def call_llm(prompt):
        ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"函数 {func.__name__} 第{attempt + 1}次尝试失败: {e}，"
                            f"{delay:.2f}秒后重试..."
                        )
                        
                        if config.on_retry_callback:
                            config.on_retry_callback(attempt, e, delay)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在{config.max_retries + 1}次尝试后仍然失败"
                        )
            
            raise last_exception
        return wrapper
    return decorator


class RetryEngine:
    """重试引擎类（适合复杂场景）"""
    
    def __init__(self, default_config: RetryConfig = None):
        self.default_config = default_config or RetryConfig()
    
    def execute_with_retry(self, 
                          func: Callable, 
                          *args, 
                          config: RetryConfig = None,
                          **kwargs) -> Any:
        """带重试执行函数"""
        if config is None:
            config = self.default_config
        
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"函数 {func.__name__} 在第{attempt + 1}次尝试成功")
                return result
            except config.retryable_exceptions as e:
                last_exception = e
                
                if attempt < config.max_retries:
                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"第{attempt + 1}次尝试失败: {e}，{delay:.2f}秒后重试"
                    )
                    
                    if config.on_retry_callback:
                        config.on_retry_callback(attempt, e, delay)
                    
                    time.sleep(delay)
        
        raise last_exception
    
    def execute_with_fallback(self,
                             primary_func: Callable,
                             fallback_func: Callable,
                             *args,
                             primary_config: RetryConfig = None,
                             **kwargs) -> Any:
        """
        主函数失败后执行降级函数
        
        用法：
        engine.execute_with_fallback(
            primary_func=call_tongyi_llm,
            fallback_func=call_local_engine,
            prompt=user_query
        )
        """
        try:
            return self.execute_with_retry(primary_func, *args, config=primary_config, **kwargs)
        except Exception as e:
            logger.warning(f"主函数执行失败，切换到降级函数: {e}")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"降级函数也执行失败: {fallback_error}")
                raise


# 默认重试引擎实例
default_retry_engine = RetryEngine()