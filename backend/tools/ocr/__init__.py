"""
OCR服务模块

提供图像识别和文档OCR功能，支持环境变量配置
"""

from .ocr_service import OCRService, get_ocr_service 
from .ocr_config import ocr_config, OCRConfig

__all__ = [
    'OCRService',
    'get_ocr_service', 
    'ocr_config',
    'OCRConfig'
]