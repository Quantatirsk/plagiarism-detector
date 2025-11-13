"""
文件读取器模块 - 简化版本

提供各种文档格式的解析功能：
- PDF (支持扫描文档OCR)
- Word文档 (DOC, DOCX)
- Excel文件 (XLS, XLSX, XLSB, ODS)
- 文本格式 (TXT, MD, JSON, CSV等)
- 图片 (支持OCR)
- 音视频 (通过媒体模块)
"""

import importlib
from typing import Any

from backend.tools.media.media_audio_parser import AudioParser
from backend.tools.media.media_video_parser import VideoParser
from backend.tools.readers.readers_doc import DOCParser
from backend.tools.readers.readers_docx import DOCXParser
from backend.tools.readers.readers_excel import ExcelParser
from backend.tools.readers.readers_image import ImageParser
from backend.tools.readers.readers_json import JSONParser
from backend.tools.readers.readers_metadata import MetadataOnlyParser as MetadataParser
from backend.tools.readers.readers_pdf import PDFParser
from backend.tools.readers.readers_text import EnhancedTextParser, PlainTextParser

# 只导入必要的基础类
from .readers_base import BaseParser, ParserFactory
from .readers_csv import CSVParser
from .readers_service import (
    ReadersService,
    get_readers_service,
    get_supported_formats,
    is_format_supported,
    parse_document,
)

# 类型检查时导入以支持静态分析

# 保持向后兼容的动态导入
def __getattr__(name: str) -> Any:
    """动态导入解析器类"""
    parser_imports = {
        'PDFParser': ('backend.tools.readers.readers_pdf', 'PDFParser'),
        'DOCXParser': ('backend.tools.readers.readers_docx', 'DOCXParser'),
        'DOCParser': ('backend.tools.readers.readers_doc', 'DOCParser'),
        'ExcelParser': ('backend.tools.readers.readers_excel', 'ExcelParser'),
        'EnhancedTextParser': ('backend.tools.readers.readers_text', 'EnhancedTextParser'),
        'PlainTextParser': ('backend.tools.readers.readers_text', 'PlainTextParser'),
        'JSONParser': ('backend.tools.readers.readers_json', 'JSONParser'),
        'ImageParser': ('backend.tools.readers.readers_image', 'ImageParser'),
        'MetadataParser': ('backend.tools.readers.readers_metadata', 'MetadataOnlyParser'),
        'AudioParser': ('backend.tools.media.media_audio_parser', 'AudioParser'),
        'VideoParser': ('backend.tools.media.media_video_parser', 'VideoParser'),
    }

    if name in parser_imports:
        module_path, class_name = parser_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'AudioParser',
    # 核心类
    'BaseParser',
    'CSVParser',
    'DOCParser',
    'DOCXParser',
    'EnhancedTextParser',
    'ExcelParser',
    'ImageParser',
    'JSONParser',
    'MetadataParser',
    # 解析器类（动态导入）
    'PDFParser',
    'ParserFactory',
    'PlainTextParser',
    # 统一服务
    'ReadersService',
    'VideoParser',
    'get_readers_service',
    'get_supported_formats',
    'is_format_supported',
    # 便捷函数
    'parse_document',
]
