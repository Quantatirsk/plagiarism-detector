"""
图片 OCR 解析器 - 简洁实现
"""

from typing import Optional
from pathlib import Path
from .readers_base import BaseParser
from ..ocr.ocr_service import get_ocr_service
import logging


logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """图片 OCR 解析器"""

    def __init__(self):
        self.ocr_service = get_ocr_service()

    def parse(self, file_path: str) -> Optional[str]:
        """同步解析图片文件"""
        try:
            if not Path(file_path).exists():
                logger.warning(f"图片文件不存在: {file_path}")
                return None

            text_content = self.ocr_service.extract_text_sync(file_path)
            return self._process_result(text_content)

        except Exception as e:
            logger.error(f"图片 OCR 解析错误 {file_path}: {e}")
            return None

    async def parse_async(self, file_path: str) -> Optional[str]:
        """异步解析图片文件（推荐在 FastAPI 中使用）"""
        try:
            if not Path(file_path).exists():
                logger.warning(f"图片文件不存在: {file_path}")
                return None

            text_content = await self.ocr_service.extract_text(file_path)
            return self._process_result(text_content)

        except Exception as e:
            logger.error(f"图片 OCR 解析错误 {file_path}: {e}")
            return None

    def _process_result(self, text_content) -> Optional[str]:
        """处理 OCR 结果"""
        if not text_content:
            return None

        if isinstance(text_content, dict):
            actual_text = text_content.get('output_markdown') or text_content.get('json_content', '')
        else:
            actual_text = text_content

        if actual_text:
            return self._clean_text(actual_text)
        return None
    
    def _clean_text(self, text: str) -> str:
        """清理 OCR 提取的文本"""
        if not text:
            return ""
        
        # 移除多余空行
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)