"""
图片 OCR 解析器 - 简洁实现
"""

from typing import Optional
import asyncio
from pathlib import Path
from .readers_base import BaseParser
from ..ocr.ocr_service import get_ocr_service
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)
class ImageParser(BaseParser):
    """图片 OCR 解析器"""
    
    def __init__(self):
        """初始化 OCR 服务"""
        self.ocr_service = get_ocr_service()
    
    def parse(self, file_path: str) -> Optional[str]:
        """
        使用 OCR 解析图片文件
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            提取的文本内容，失败返回 None
        """
        try:
            # 检查文件存在
            if not Path(file_path).exists():
                logger.warning(f"图片文件不存在: {file_path}")
                return None
            
            # 使用 OCR 服务提取文本
            # 在同步代码中调用异步OCR服务
            text_content = asyncio.run(self.ocr_service.extract_text(file_path))
            
            if text_content:
                # 如果返回的是字典，提取文本内容
                if isinstance(text_content, dict):
                    # 优先使用 output_markdown，其次是 json_content
                    actual_text = text_content.get('output_markdown') or text_content.get('json_content', '')
                else:
                    actual_text = text_content
                
                # 简单清理文本
                if actual_text:
                    cleaned = self._clean_text(actual_text)
                    return cleaned
            
            return None
                
        except Exception as e:
            logger.error(f"图片 OCR 解析错误 {file_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """清理 OCR 提取的文本"""
        if not text:
            return ""
        
        # 移除多余空行
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)