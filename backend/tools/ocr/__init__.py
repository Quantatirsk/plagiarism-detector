"""
OCR服务模块

提供图像识别和文档OCR功能，支持环境变量配置
"""

from .ocr_config import OCRConfig, ocr_config
from .ocr_service import OCRService, get_ocr_service

__all__ = [
    'OCRConfig',
    'OCRService',
    'get_ocr_service',
    'ocr_config'
]
