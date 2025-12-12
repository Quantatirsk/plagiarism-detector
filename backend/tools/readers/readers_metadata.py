"""
仅元数据解析器 - 通用文件类型支持

【当前状态】
⚠️ 此解析器当前未被使用，因为：
1. 没有在 readers_parser_map.py 中配置映射
2. 没有作为 readers_service.py 的默认后备解析器
3. 只在 __init__.py 中有动态导入引用，但无实际调用路径

【设计意图】
作为通用后备解析器，为所有文件类型提供基础支持：
- 仅提取文件元数据（路径、大小、时间）
- 不尝试解析文本内容
- 返回空字符串，允许文件在 docs_meta 表中被索引
- 支持通过路径搜索找到文件

【潜在用途】
1. 作为默认后备解析器处理未知文件类型
2. 处理二进制文件（.exe, .dll, .so）
3. 处理媒体文件（当不需要转写时）
4. 处理压缩文件（当不需要解压时）

【激活方式】
如需启用，可选择：
- 方案1：在 readers_service.py 中作为后备解析器
- 方案2：在 readers_parser_map.py 中为特定格式配置
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional, List
from .readers_base import BaseParser
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)
class MetadataOnlyParser(BaseParser):
    """
    通用解析器，仅提取文件元数据。

    此解析器支持所有文件类型并返回空内容，
    允许文件通过元数据（路径、大小、创建/修改时间）被索引，
    同时可通过路径搜索功能找到文件。
    """

    def parse(self, file_path: str) -> Optional[str]:
        """
        从任何文件类型仅提取元数据。

        Args:
            file_path: 文件路径

        Returns:
            空字符串（无文本内容）或 None（如果文件不可访问）
        """
        try:
            file_path_obj = Path(file_path)

            # 检查文件是否存在和可访问
            if not file_path_obj.exists():
                return None

            if not file_path_obj.is_file():
                return None

            # 检查文件是否可读
            if not os.access(file_path, os.R_OK):
                return None

            # 仅用于元数据索引，返回空字符串
            # 这允许文件在docs_meta表中被索引
            # 但不在docs_fts表中
            return ""

        except Exception as e:
            logger.error(f"访问文件元数据错误 {file_path}: {e}")
            return None

    # 不再需要 get_supported_extensions() - 由 readers_parser_map.py 管理
    # 不再需要 is_supported() - 由 readers_parser_map.py 管理
    # Linux 哲学：单一职责，只负责解析

    def get_file_mime_type(self, file_path: str) -> Optional[str]:
        """
        获取文件的 MIME 类型。

        Args:
            file_path: 文件路径

        Returns:
            MIME 类型字符串，如果无法确定则返回 None
        """
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type
        except Exception:
            return None

    def get_file_category(self, file_path: str) -> str:
        """
        根据扩展名和 MIME 类型对文件进行分类。

        Args:
            file_path: 文件路径

        Returns:
            文件类别字符串
        """
        try:
            file_path_obj = Path(file_path)
            extension = file_path_obj.suffix.lower()
            mime_type = self.get_file_mime_type(file_path)

            # 图像文件
            if mime_type and mime_type.startswith('image/'):
                return 'image'
            elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp', '.ico']:
                return 'image'

            # 音频文件
            elif mime_type and mime_type.startswith('audio/'):
                return 'audio'
            elif extension in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']:
                return 'audio'

            # 视频文件
            elif mime_type and mime_type.startswith('video/'):
                return 'video'
            elif extension in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']:
                return 'video'

            # 归档/压缩文件
            elif extension in ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tar.gz', '.tar.bz2']:
                return 'archive'

            # 可执行文件
            elif extension in ['.exe', '.msi', '.dmg', '.app', '.deb', '.rpm', '.appimage']:
                return 'executable'

            # 文档文件（有专门的文本内容解析器）
            elif extension in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
                return 'document'

            # 文本/代码文件（有专门的文本内容解析器）
            elif extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                return 'text'

            # 其他/未知
            else:
                return 'other'

        except Exception:
            return 'other'

# 不再需要注册到 ParserFactory - 由 readers_parser_map.py 管理
