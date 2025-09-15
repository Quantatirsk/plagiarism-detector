"""
文档解析服务 - 统一多格式文档解析

支持格式：
- PDF (.pdf)
- Word文档 (.docx, .doc)
- Markdown (.md)
- 纯文本 (.txt)
- 其他文本格式 (.rst, .py, .js 等)
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from app.tools.readers import get_readers_service

logger = logging.getLogger(__name__)


class DocumentParser:
    """文档解析器 - 支持多种格式"""

    def __init__(self):
        self.readers_service = get_readers_service()

    def parse_document(self, file_path: str) -> Optional[str]:
        """
        解析文档并返回纯文本内容

        Args:
            file_path: 文档文件路径

        Returns:
            解析的纯文本内容，失败返回None
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                logger.error(f"文件不存在: {file_path}")
                return None

            # 使用readers_service解析文档
            content = self.readers_service.parse_document(file_path)

            if content is None:
                logger.warning(f"无法解析文档: {file_path}")
                return None

            # 基本文本清理
            content = self._clean_text(content)

            logger.info(f"成功解析文档 {file_path}, 提取了 {len(content)} 个字符")
            return content

        except Exception as e:
            logger.error(f"解析文档时出错 {file_path}: {e}")
            return None

    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文档信息

        Args:
            file_path: 文档文件路径

        Returns:
            文档信息字典
        """
        try:
            path = Path(file_path)

            info = {
                'filename': path.name,
                'extension': path.suffix.lower(),
                'size': path.stat().st_size if path.exists() else 0,
                'is_supported': self.is_format_supported(file_path),
                'exists': path.exists()
            }

            # 如果支持该格式，尝试获取文本长度
            if info['is_supported'] and info['exists']:
                content = self.parse_document(file_path)
                info['text_length'] = len(content) if content else 0
                info['is_parseable'] = content is not None
            else:
                info['text_length'] = 0
                info['is_parseable'] = False

            return info

        except Exception as e:
            logger.error(f"获取文档信息时出错 {file_path}: {e}")
            return {
                'filename': Path(file_path).name,
                'extension': '',
                'size': 0,
                'is_supported': False,
                'exists': False,
                'text_length': 0,
                'is_parseable': False,
                'error': str(e)
            }

    def is_format_supported(self, file_path: str) -> bool:
        """
        检查文件格式是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持该格式
        """
        return self.readers_service.is_format_supported(file_path)

    def get_supported_formats(self) -> list:
        """
        获取支持的文件格式列表

        Returns:
            支持的文件扩展名列表
        """
        return self.readers_service.get_supported_formats()

    def _clean_text(self, text: str) -> str:
        """
        清理文本内容

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 移除过多的空行（保留段落分隔）
        lines = text.split('\n')
        cleaned_lines = []
        empty_line_count = 0

        for line in lines:
            stripped_line = line.strip()

            if stripped_line:
                # 非空行：重置空行计数并添加该行
                empty_line_count = 0
                cleaned_lines.append(line.rstrip())  # 移除行尾空白
            else:
                # 空行：最多连续保留两个空行
                empty_line_count += 1
                if empty_line_count <= 2:
                    cleaned_lines.append('')

        # 移除开头和结尾的空行
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return '\n'.join(cleaned_lines)


# 全局实例
_document_parser = None

def get_document_parser() -> DocumentParser:
    """获取全局文档解析器实例"""
    global _document_parser
    if _document_parser is None:
        _document_parser = DocumentParser()
    return _document_parser