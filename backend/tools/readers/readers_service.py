"""
统一文档读取服务

提供统一的文档解析接口，支持延迟加载各种解析器
支持同步和异步两种模式
"""

import importlib
import logging
from pathlib import Path
from typing import cast

from .readers_base import BaseParser
from .utils.parser_map import get_parser_for_file, get_parser_map

logger = logging.getLogger(__name__)

class ReadersService:
    """统一文档读取服务"""

    def __init__(self):
        self._parsers = {}
        # 使用集中的映射配置 - Linux 哲学：单一职责
        self._parser_map = get_parser_map()

    def parse_document(self, file_path: str, **options) -> str | None:
        """
        解析文档并返回文本内容（同步方法）

        Args:
            file_path: 文档文件路径
            **options: 保留用于向后兼容，但不再使用

        Returns:
            解析的文本内容，失败返回None
        """
        parser = self._get_parser(file_path)
        if parser:
            try:
                # Linux 哲学：所有解析器现在都自主处理编码
                # 不再需要传递 options - 简单即美
                return parser.parse(file_path)
            except Exception as e:
                logger.error(f"Parse failed for {file_path}: {e}")
        return None

    async def parse_document_async(self, file_path: str, **options) -> str | None:
        """
        异步解析文档并返回文本内容

        Args:
            file_path: 文档文件路径
            **options: 保留用于向后兼容，但不再使用

        Returns:
            解析的文本内容，失败返回None
        """
        parser = self._get_parser(file_path)
        if parser:
            try:
                # 调用解析器的异步方法
                # 如果解析器重写了 parse_async，则使用真正的异步
                # 否则使用基类的默认实现（asyncio.to_thread）
                return await parser.parse_async(file_path)
            except Exception as e:
                logger.error(f"Async parse failed for {file_path}: {e}")
        return None

    def _get_parser(self, file_path: str) -> BaseParser | None:
        """获取解析器（延迟加载）"""
        # 使用统一的映射逻辑，包括处理无扩展名文件
        parser_info = get_parser_for_file(file_path)
        if not parser_info:
            return None

        module_path, class_name = parser_info

        # 使用 (module_path, class_name) 作为缓存键
        cache_key = f"{module_path}.{class_name}"

        if cache_key not in self._parsers:
            try:
                module = importlib.import_module(module_path)
                parser_class = getattr(module, class_name)
                self._parsers[cache_key] = cast("BaseParser", parser_class())
            except Exception as e:
                logger.info(f"Parser {cache_key} not available: {e}")
                self._parsers[cache_key] = None

        return cast("BaseParser | None", self._parsers[cache_key])

    def get_supported_formats(self) -> list[str]:
        """获取支持的文件格式列表"""
        return list(self._parser_map.keys())

    def is_format_supported(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = Path(file_path).suffix.lower()
        return ext in self._parser_map

    def get_available_parsers(self) -> dict[str, bool]:
        """获取各格式解析器的可用性"""
        availability = {}
        for ext, (module_path, _class_name) in self._parser_map.items():
            importlib.import_module(module_path)
            availability[ext] = True
        return availability

# 全局实例
_readers_service = None

def get_readers_service() -> ReadersService:
    """获取全局读取服务实例"""
    global _readers_service
    if _readers_service is None:
        _readers_service = ReadersService()
    return _readers_service

# 便捷函数
def parse_document(file_path: str, **options) -> str | None:
    """便捷函数：解析文档（同步）"""
    return get_readers_service().parse_document(file_path, **options)

async def parse_document_async(file_path: str, **options) -> str | None:
    """便捷函数：解析文档（异步）"""
    return await get_readers_service().parse_document_async(file_path, **options)

def is_format_supported(file_path: str) -> bool:
    """便捷函数：检查格式是否支持"""
    return get_readers_service().is_format_supported(file_path)

def get_supported_formats() -> list[str]:
    """便捷函数：获取支持的格式列表"""
    return get_readers_service().get_supported_formats()
