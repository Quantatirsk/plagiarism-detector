"""
PDF 解析器 - 使用 PyMuPDF 进行基本文本提取
"""

from typing import Optional
import re
from .readers_base import BaseParser
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """PDF 解析器 - 基本文本提取版本"""

    def parse(self, file_path: str) -> Optional[str]:
        """
        解析PDF文件并提取文本

        Args:
            file_path: PDF文件路径

        Returns:
            提取的文本内容，解析失败时返回None
        """
        try:
            # 尝试导入 PyMuPDF
            try:
                import pymupdf
            except ImportError:
                logger.error("PyMuPDF 未安装，无法解析PDF文件")
                return None

            # 打开PDF文档
            doc = pymupdf.open(file_path)

            # 提取所有页面的文本
            all_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    all_text.append(text)

            # 关闭文档
            doc.close()

            if not all_text:
                return None

            # 合并所有文本
            raw_text = '\n\n'.join(all_text)

            # 基本清理：移除多余的空白字符
            raw_text = re.sub(r'\n\s*\n\s*\n', '\n\n', raw_text)  # 合并多个空行为双空行
            raw_text = re.sub(r'[ \t]+', ' ', raw_text)  # 合并多个空格/制表符

            return raw_text.strip()

        except Exception as e:
            logger.error(f"解析PDF文件错误 {file_path}: {e}")
            return None