"""
CSV 文件解析器
返回结构化的字典列表，每行是一个完整的字典

Linux 哲学：只负责解析，不管格式映射
"""

import csv
import json
from typing import Optional
from .readers_base import BaseParser
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)
class CSVParser(BaseParser):
    """CSV 文件解析器 - 极简实现"""
    
    def parse(self, file_path: str) -> Optional[str]:
        """
        解析 CSV 文件为结构化字典列表
        
        Args:
            file_path: CSV 文件路径
            
        Returns:
            JSON 字符串，包含字典列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"解析 CSV 文件错误 {file_path}: {e}")
            return None
    
    # 不再需要 get_supported_extensions() - 由 readers_parser_map.py 管理
    # Linux 哲学：单一职责