"""
文档解析器基类 - 符合 Linux 哲学的极简设计

单一职责：只定义解析接口
格式支持由 readers_parser_map.py 统一管理
"""

from abc import ABC, abstractmethod
from typing import Optional, Type, Dict
import importlib
from .utils.parser_map import get_parser_for_file

class BaseParser(ABC):
    """
    极简的文档解析器基类

    Linux 哲学：
    1. Do one thing well - 只负责定义解析接口
    2. KISS - 只有一个必需的方法
    3. 配置与代码分离 - 格式映射由 readers_parser_map.py 管理
    """

    @abstractmethod
    def parse(self, file_path: str) -> Optional[str]:
        """
        解析文档并提取纯文本内容。

        Args:
            file_path: 文档文件路径

        Returns:
            提取的纯文本内容，解析失败时返回None
        """
        pass

    # 不再需要 get_supported_extensions() - 由 readers_parser_map.py 管理
    # 不再需要 is_supported() - 由 readers_parser_map.py 管理
    # Linux 哲学：减法优于加法

class ParserFactory:
    """
    极简的解析器工厂 - 符合 Linux 哲学

    设计原则：
    1. 配置驱动 - 从 readers_parser_map.py 读取映射
    2. 单一职责 - 只负责创建解析器实例
    3. 无状态注册 - 不维护内部注册表，配置即真相
    4. 缓存优化 - 避免重复导入，提升性能
    """

    # 简单的类缓存（避免重复导入）
    _cache: Dict[str, Type[BaseParser]] = {}

    @classmethod
    def get_parser(cls, file_path: str) -> Optional[BaseParser]:
        """
        根据文件路径获取合适的解析器

        Linux 哲学：Simple is better than complex

        Args:
            file_path: 文件路径

        Returns:
            解析器实例，如果没有合适的解析器则返回 None
        """
        # 从配置获取解析器信息
        parser_info = get_parser_for_file(file_path)
        if not parser_info:
            return None

        module_path, class_name = parser_info
        cache_key = f"{module_path}.{class_name}"

        # 检查缓存
        if cache_key not in cls._cache:
            # 动态导入
            module = importlib.import_module(module_path)
            cls._cache[cache_key] = getattr(module, class_name)

        # 创建实例
        return cls._cache[cache_key]()

    @classmethod
    def clear_cache(cls):
        """清空缓存 - 用于测试或重新加载"""
        cls._cache.clear()