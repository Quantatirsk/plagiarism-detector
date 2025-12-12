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

from tools.media.media_audio_parser import AudioParser
from tools.media.media_video_parser import VideoParser
from tools.readers.readers_doc import DOCParser
from tools.readers.readers_docx import DOCXParser
from tools.readers.readers_excel import ExcelParser
from tools.readers.readers_image import ImageParser
from tools.readers.readers_json import JSONParser
from tools.readers.readers_metadata import MetadataOnlyParser as MetadataParser
from tools.readers.readers_pdf import PDFParser
from tools.readers.readers_text import EnhancedTextParser, PlainTextParser

from .readers_base import BaseParser, ParserFactory
from .readers_csv import CSVParser
from .readers_service import (
    ReadersService,
    get_readers_service,
    get_supported_formats,
    is_format_supported,
    parse_document,
    parse_document_async,
)

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
    'parse_document_async',
]
