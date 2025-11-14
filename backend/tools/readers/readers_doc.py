"""
DOC parser using doc2txt library.

Using the doc2txt library which provides a Python wrapper around antiword
with cross-platform binary files and text optimization features.
"""

import logging

from doc2txt import extract_text

from .readers_base import BaseParser

# 配置日志记录器
logger = logging.getLogger(__name__)
class DOCParser(BaseParser):
    """
    Parser for legacy Word documents (.doc) using doc2txt library.

    The doc2txt library provides a Python wrapper around antiword
    with built-in cross-platform support and text optimization.
    """

    def parse(self, file_path: str) -> str | None:
        """
        Parse a DOC file and extract optimized text.

        Args:
            file_path: Path to the DOC file

        Returns:
            Extracted and optimized text or None if parsing fails
        """
        try:
            return extract_text(file_path, optimize_format=True)
        except Exception as e:
            logger.error(f"解析DOC文件错误 {file_path}: {e}")
            return None

    # 不再需要 get_supported_extensions() - 由 readers_parser_map.py 管理
    # Linux 哲学：单一职责，只负责解析
