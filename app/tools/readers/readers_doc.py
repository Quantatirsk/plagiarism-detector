"""
DOC parser using doc2txt library or python-docx as fallback.

For legacy Word documents (.doc) format support.
"""

from typing import Optional
from .readers_base import BaseParser
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)


class DOCParser(BaseParser):
    """
    Parser for legacy Word documents (.doc).

    Tries to use doc2txt library first, falls back to python-docx if available.
    """

    def parse(self, file_path: str) -> Optional[str]:
        """
        Parse a DOC file and extract text.

        Args:
            file_path: Path to the DOC file

        Returns:
            Extracted text or None if parsing fails
        """
        # Try doc2txt first (most efficient for DOC files)
        text = self._try_doc2txt(file_path)
        if text:
            return text

        # Fallback to python-docx (may not work well with old DOC files)
        text = self._try_python_docx(file_path)
        if text:
            return text

        logger.error(f"无法解析DOC文件 {file_path}: 缺少必要的依赖库")
        return None

    def _try_doc2txt(self, file_path: str) -> Optional[str]:
        """尝试使用 doc2txt 库解析"""
        try:
            from doc2txt import extract_text
            return extract_text(file_path, optimize_format=True)
        except ImportError:
            logger.debug("doc2txt library not available")
            return None
        except Exception as e:
            logger.debug(f"doc2txt 解析失败: {e}")
            return None

    def _try_python_docx(self, file_path: str) -> Optional[str]:
        """尝试使用 python-docx 库解析（可能不兼容所有DOC文件）"""
        try:
            from docx import Document
            doc = Document(file_path)

            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            return '\n\n'.join(paragraphs)
        except ImportError:
            logger.debug("python-docx library not available")
            return None
        except Exception as e:
            logger.debug(f"python-docx 解析失败: {e}")
            return None