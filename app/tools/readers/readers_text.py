"""
支持所有基于文本的文件格式的增强文本解析器。

此解析器支持全面的基于文本的文件格式列表，包括：
- 编程语言 (Python, JavaScript, Java, C/C++等)
- 配置文件 (JSON, YAML, XML, INI等)
- 文档文件 (Markdown, reStructuredText等)
- Web技术 (HTML, CSS等)
- Shell脚本和其他文本格式
"""

from typing import Optional
from .readers_base import BaseParser
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)


class EnhancedTextParser(BaseParser):
    """
    Enhanced parser for all text-based file formats.

    Supports programming languages, configuration files, documentation,
    and other text-based formats commonly found in software projects.
    """

    def parse(self, file_path: str) -> Optional[str]:
        """
        Parse any text-based file with automatic encoding detection.

        Args:
            file_path: Path to the text file

        Returns:
            File content as string or None if parsing fails
        """
        try:
            # 自动尝试多种编码 - Unix 哲学：健壮性原则
            # UTF-8 是默认和首选，符合现代标准
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'gb2312', 'gbk']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        # If we successfully read content, return it
                        if content:
                            return content
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue

            # If all encodings fail, try reading as binary and decode with errors ignored
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                return raw_content.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"解析文本文件错误 {file_path}: {e}")
            return None

    # 不再需要 is_supported() - 由 readers_parser_map.py 管理
    # 不再需要 get_supported_extensions() - 由 readers_parser_map.py 管理
    # Linux 哲学：单一职责 - 只负责解析，不管格式映射

# Update the basic PlainTextParser to be replaced by EnhancedTextParser
class PlainTextParser(EnhancedTextParser):
    """
    Backward compatible plain text parser.
    Now inherits from EnhancedTextParser for extended functionality.
    """
    pass