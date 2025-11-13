"""
OCR 服务配置管理
"""

import os
from typing import Optional

# 导入配置加载器确保环境变量已加载
try:
    from ..config_loader import load_tools_config
    # 自动加载配置（如果还未加载）
    load_tools_config()
except ImportError:
    # 如果无法导入配置加载器，继续使用默认方式
    pass

class OCRConfig:
    """OCR 服务配置类 - MinerU API"""

    def __init__(self):
        # MinerU 服务基础配置
        self.base_url: str = os.getenv("OCR_BASE_URL", "http://home.teea.cn:7000")
        self.api_key: Optional[str] = os.getenv("OCR_API_KEY")  # 如果需要API Key

        # 超时配置（单一超时设置，涵盖所有操作）
        self.timeout: int = int(os.getenv("OCR_TIMEOUT", "1200"))  # 默认20分钟

        # 重试配置
        self.max_retries: int = int(os.getenv("OCR_MAX_RETRIES", "3"))
        self.retry_delay: int = int(os.getenv("OCR_RETRY_DELAY", "2"))

        # MinerU 特定配置
        self.backend: str = os.getenv("MINERU_BACKEND", "pipeline")  # pipeline/txt/ocr
        self.parse_method: str = os.getenv("MINERU_PARSE_METHOD", "auto")  # auto/txt/ocr
        self.formula_enable: bool = os.getenv("MINERU_FORMULA_ENABLE", "true").lower() == "true"
        self.table_enable: bool = os.getenv("MINERU_TABLE_ENABLE", "true").lower() == "true"
        self.response_format_zip: bool = os.getenv("MINERU_RESPONSE_FORMAT_ZIP", "true").lower() == "true"

        # 缓存配置
        self.cache_enabled: bool = os.getenv("OCR_CACHE_ENABLED", "false").lower() == "true"
        self.cache_dir: str = os.getenv("OCR_CACHE_DIR", ".ocr_cache")
        self.cache_ttl: int = int(os.getenv("OCR_CACHE_TTL", "3600"))  # 缓存过期时间(秒)

        # 支持的文件格式
        self.supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}

        # 最大文件大小 (MB)
        self.max_file_size: int = int(os.getenv("OCR_MAX_FILE_SIZE", "50"))

        # 调试模式
        self.debug_mode: bool = os.getenv("OCR_DEBUG", "false").lower() == "true"
    
    @property
    def timeout_tuple(self) -> tuple:
        """获取超时配置元组（兼容性保留，MinerU 使用单一超时）"""
        return (self.timeout, self.timeout)
    
    def get_headers(self) -> dict:
        """获取请求头"""
        headers = {
            'User-Agent': 'Refly-AI OCR Client/1.0'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def validate(self) -> list[str]:
        """验证配置"""
        errors = []

        # 检查基础URL
        if not self.base_url or not self.base_url.startswith(('http://', 'https://')):
            errors.append("OCR_BASE_URL 必须是有效的HTTP/HTTPS URL")

        # 检查超时配置
        if self.timeout <= 0:
            errors.append("OCR_TIMEOUT 必须大于0")

        # 检查重试配置
        if self.max_retries < 0:
            errors.append("OCR_MAX_RETRIES 不能小于0")

        if self.retry_delay < 0:
            errors.append("OCR_RETRY_DELAY 不能小于0")

        # 检查文件大小限制
        if self.max_file_size <= 0:
            errors.append("OCR_MAX_FILE_SIZE 必须大于0")

        # 检查 MinerU 特定配置
        valid_backends = ['pipeline', 'txt', 'ocr']
        if self.backend not in valid_backends:
            errors.append(f"MINERU_BACKEND 必须是以下之一: {valid_backends}")

        valid_parse_methods = ['auto', 'txt', 'ocr']
        if self.parse_method not in valid_parse_methods:
            errors.append(f"MINERU_PARSE_METHOD 必须是以下之一: {valid_parse_methods}")

        return errors
    
    def is_file_size_valid(self, file_path: str) -> bool:
        """检查文件大小是否在限制范围内"""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return file_size_mb <= self.max_file_size
        except OSError:
            return False
    
    def is_supported_format(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        from pathlib import Path
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions
    
    def get_cache_config(self) -> dict:
        """获取缓存配置"""
        return {
            'enabled': self.cache_enabled,
            'cache_dir': self.cache_dir,
            'ttl': self.cache_ttl
        }
    
    def __str__(self) -> str:
        """配置信息字符串表示"""
        return f"""MinerU OCR Configuration:
  Base URL: {self.base_url}
  API Key: {'设置' if self.api_key else '未设置'}
  Timeout: {self.timeout}s
  Max Retries: {self.max_retries}
  Retry Delay: {self.retry_delay}s
  Backend: {self.backend}
  Parse Method: {self.parse_method}
  Formula Enable: {self.formula_enable}
  Table Enable: {self.table_enable}
  Response Format ZIP: {self.response_format_zip}
  Max File Size: {self.max_file_size}MB
  Cache Enabled: {self.cache_enabled}
  Debug Mode: {self.debug_mode}
"""

# 全局配置实例
ocr_config = OCRConfig()