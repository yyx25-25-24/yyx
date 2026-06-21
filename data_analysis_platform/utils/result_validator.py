"""
结果校验模块 - 验证分析结果的安全性和格式正确性
"""
import re
import ast
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """校验结果"""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    sanitized_output: Any = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class CodeSecurityValidator:
    """代码安全性校验器"""
    
    # 危险的Python内置函数和模块
    DANGEROUS_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'input',
        'globals', 'locals', 'vars', 'dir'
    }
    
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pickle', 
        'marshal', 'ctypes', 'multiprocessing'
    }
    
    @classmethod
    def validate_python_code(cls, code: str) -> ValidationResult:
        """验证Python代码的安全性"""
        errors = []
        warnings = []
        
        # 1. 检查危险关键字
        dangerous_patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bos\.system\s*\(',
            r'\bsubprocess\.',
            r'\bpickle\.',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                errors.append(f"检测到危险代码模式: {pattern}")
        
        # 2. 尝试AST解析检查
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # 检查导入语句
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in cls.DANGEROUS_MODULES:
                            errors.append(f"禁止导入危险模块: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in cls.DANGEROUS_MODULES:
                        errors.append(f"禁止从危险模块导入: {node.module}")
                
                # 检查函数调用
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in cls.DANGEROUS_BUILTINS:
                            errors.append(f"禁止调用危险函数: {node.func.id}")
        except SyntaxError as e:
            errors.append(f"代码语法错误: {e}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_output=code if is_valid else None
        )


class OutputFormatValidator:
    """输出格式校验器"""
    
    @classmethod
    def validate_text_report(cls, text: str) -> ValidationResult:
        """验证文本报告格式"""
        errors = []
        warnings = []
        
        if not text or not text.strip():
            errors.append("报告内容为空")
            return ValidationResult(is_valid=False, errors=errors)
        
        # 检查是否包含基本的结构
        if len(text) < 10:
            warnings.append("报告内容过短，可能不完整")
        
        # 检查是否有明显的乱码
        if '\x00' in text or '\\x' in text:
            warnings.append("检测到可能的乱码字符")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_output=text.strip()
        )
    
    @classmethod
    def validate_json_output(cls, data: Any, expected_schema: Dict = None) -> ValidationResult:
        """验证JSON输出格式"""
        errors = []
        warnings = []
        
        if not isinstance(data, (dict, list)):
            errors.append(f"期望dict或list类型，实际得到: {type(data).__name__}")
            return ValidationResult(is_valid=False, errors=errors)
        
        # 如果提供了schema，进行schema验证
        if expected_schema:
            try:
                import jsonschema
                jsonschema.validate(instance=data, schema=expected_schema)
            except ImportError:
                warnings.append("jsonschema未安装，跳过schema验证")
            except Exception as e:
                errors.append(f"Schema验证失败: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_output=data
        )
    
    @classmethod
    def validate_dataframe_result(cls, df) -> ValidationResult:
        """验证DataFrame结果"""
        errors = []
        warnings = []
        
        try:
            import pandas as pd
            if not isinstance(df, pd.DataFrame):
                errors.append("结果不是DataFrame类型")
                return ValidationResult(is_valid=False, errors=errors)
            
            if df.empty:
                warnings.append("DataFrame为空")
            
            if len(df.columns) == 0:
                errors.append("DataFrame没有列")
            
            # 检查是否有NaN值过多
            nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[1] > 0 else 0
            if nan_ratio > 0.5:
                warnings.append(f"DataFrame中NaN值占比过高: {nan_ratio:.2%}")
                
        except ImportError:
            errors.append("pandas未安装")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_output=df
        )


class ResultValidator:
    """统一的结果校验器"""
    
    def __init__(self):
        self.code_validator = CodeSecurityValidator()
        self.format_validator = OutputFormatValidator()
    
    def validate(self, result: Any, result_type: str = "auto") -> ValidationResult:
        """
        根据类型自动选择合适的校验器
        
        Args:
            result: 待校验的结果
            result_type: 结果类型 ('code', 'text', 'json', 'dataframe', 'auto')
        """
        if result_type == "auto":
            if isinstance(result, str):
                # 判断是代码还是普通文本
                if any(keyword in result for keyword in ['def ', 'import ', 'return ']):
                    result_type = "code"
                else:
                    result_type = "text"
            elif hasattr(result, 'to_dict'):  # DataFrame-like
                result_type = "dataframe"
            else:
                result_type = "json"
        
        if result_type == "code":
            return self.code_validator.validate_python_code(result)
        elif result_type == "text":
            return self.format_validator.validate_text_report(result)
        elif result_type == "json":
            return self.format_validator.validate_json_output(result)
        elif result_type == "dataframe":
            return self.format_validator.validate_dataframe_result(result)
        else:
            return ValidationResult(
                is_valid=True,
                warnings=[f"未知的结果类型: {result_type}，跳过校验"],
                sanitized_output=result
            )


# 全局校验器实例
validator = ResultValidator()