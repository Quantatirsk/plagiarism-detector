"""
文档解析器工具包

支持的文档格式：
- PDF (.pdf)
- Word文档 (.docx, .doc)
- 纯文本 (.txt, .md, .rst 等)
- 编程语言文件 (.py, .js, .java 等)
"""

from .readers_service import ReadersService, get_readers_service, parse_document, is_format_supported, get_supported_formats

__all__ = [
    'ReadersService',
    'get_readers_service',
    'parse_document',
    'is_format_supported',
    'get_supported_formats'
]