"""
PDF 解析器 - 使用 PyMuPDF
支持同步和异步两种模式
"""

from typing import Optional, Any, cast
import re
import asyncio
import pymupdf
from collections import defaultdict
from .readers_base import BaseParser
from ..ocr.ocr_service import get_ocr_service
from .utils.para_optimizer import ParagraphOptimizer
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)
class PDFParser(BaseParser):
    """PDF 解析器"""
    
    def __init__(self):
        """初始化PDF解析器和文本优化器。"""
        super().__init__()
        self.text_optimizer = ParagraphOptimizer()

    def _is_scanned_pdf(self, doc: pymupdf.Document) -> bool:
        """
        检测PDF是否为扫描件（主要包含图像而非文本）

        Args:
            doc: PyMuPDF文档对象

        Returns:
            如果PDF是扫描件返回True
        """
        try:
            total_pages = len(doc)
            if total_pages == 0:
                return True

            # 检查前几页的文本内容
            pages_to_check = min(3, total_pages)
            total_text_chars = 0
            total_image_count = 0

            for page_num in range(pages_to_check):
                page = cast(Any, doc.load_page(page_num))

                # 获取页面的文本内容
                text = page.get_text().strip()
                total_text_chars += len(text)

                # 获取页面的图像数量
                image_list = page.get_images()
                total_image_count += len(image_list)

            # 判断标准：
            # 1. 平均每页文本少于100个字符
            # 2. 有图像存在
            avg_chars_per_page = total_text_chars / pages_to_check
            is_scanned = avg_chars_per_page < 100 and total_image_count > 0

            if is_scanned:
                logger.info(f"检测到扫描PDF：平均每页 {avg_chars_per_page:.0f} 字符, {total_image_count} 张图像")
            
            return is_scanned

        except Exception as e:
            logger.info(f"检测PDF类型时出错: {e}")
            return False

    def _extract_text_from_scanned_pdf(self, file_path: str) -> str:
        """
        从扫描的PDF中提取文本（使用OCR）- 同步方法

        Args:
            file_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        try:
            ocr_service = get_ocr_service()
            if not ocr_service:
                logger.info("OCR服务不可用，无法处理扫描PDF")
                return ""

            # 使用OCR服务的同步方法
            logger.debug(f"正在使用OCR处理PDF文件...")
            ocr_result = ocr_service.extract_text_sync(file_path)

            # 如果返回的是字典，提取文本内容
            if isinstance(ocr_result, dict):
                ocr_text = ocr_result.get('output_markdown') or ocr_result.get('json_content', '')
            else:
                ocr_text = ocr_result

            if ocr_text:
                logger.info(f"OCR处理成功，提取到 {len(ocr_text)} 个字符")
                return ocr_text
            else:
                logger.info("OCR未能提取到文本")
                return ""

        except Exception as e:
            logger.error(f"OCR处理失败: {e}")
            return ""

    async def _extract_text_from_scanned_pdf_async(self, file_path: str) -> str:
        """
        从扫描的PDF中提取文本（使用OCR）- 异步方法

        Args:
            file_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        try:
            ocr_service = get_ocr_service()
            if not ocr_service:
                logger.info("OCR服务不可用，无法处理扫描PDF")
                return ""

            # 使用OCR服务的异步方法
            logger.debug(f"正在使用OCR异步处理PDF文件...")
            ocr_result = await ocr_service.extract_text(file_path)

            # 如果返回的是字典，提取文本内容
            if isinstance(ocr_result, dict):
                ocr_text = ocr_result.get('output_markdown') or ocr_result.get('json_content', '')
            else:
                ocr_text = ocr_result

            if ocr_text:
                logger.info(f"OCR异步处理成功，提取到 {len(ocr_text)} 个字符")
                return ocr_text
            else:
                logger.info("OCR未能提取到文本")
                return ""

        except Exception as e:
            logger.error(f"OCR异步处理失败: {e}")
            return ""

    def _extract_text_from_native_pdf(self, doc: pymupdf.Document) -> str:
        """
        从原生PDF中提取文本（包含可搜索文本）

        Args:
            doc: PyMuPDF文档对象

        Returns:
            提取的文本内容
        """
        all_text = []

        for page_num in range(len(doc)):
            page = cast(Any, doc.load_page(page_num))
            text = page.get_text()
            if text.strip():
                all_text.append(text)

        return '\n\n'.join(all_text)

    def _apply_paragraph_optimization(self, text: str) -> str:
        """
        应用段落优化

        Args:
            text: 原始文本

        Returns:
            优化后的文本
        """
        if not self.text_optimizer:
            return text
        
        try:
            return self.text_optimizer.optimize_text(text)
        except Exception as e:
            logger.error(f"段落优化失败: {e}")
            return text

    def _remove_headers_footers(self, text: str) -> str:
        """
        移除页眉页脚

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        # 简单的启发式方法：跳过重复出现的短行
        line_counts = defaultdict(int)
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:  # 只统计短行
                line_counts[stripped] += 1
        
        # 找出重复出现的行（可能是页眉页脚）
        repeated_lines = {line for line, count in line_counts.items() if count > 2}
        
        for line in lines:
            stripped = line.strip()
            if stripped not in repeated_lines:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _remove_page_numbers(self, text: str) -> str:
        """
        移除页码

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        # 移除常见的页码模式
        patterns = [
            r'^\d+$',  # 纯数字
            r'^第\s*\d+\s*页$',  # 第 X 页
            r'^Page\s*\d+$',  # Page X
            r'^\d+\s*/\s*\d+$',  # X/Y
            r'^-\s*\d+\s*-$',  # - X -
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            is_page_number = False
            
            for pattern in patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_page_number = True
                    break
            
            if not is_page_number:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def parse(self, file_path: str, **options) -> Optional[str]:
        """
        解析PDF文件（同步方法）

        Args:
            file_path: PDF文件路径
            **options: 可选参数
                - optimize_text: 是否优化文本段落（默认True）
                - remove_headers_footers: 是否移除页眉页脚（默认False）
                - remove_page_numbers: 是否移除页码（默认False）
                - use_ocr: 是否对扫描件使用OCR（默认True）

        Returns:
            提取的文本内容
        """
        try:
            # 打开PDF文档
            doc = pymupdf.open(file_path)

            use_ocr = options.get('use_ocr', True)
            # 检测是否为扫描PDF
            if use_ocr and self._is_scanned_pdf(doc):
                raw_text = self._extract_text_from_scanned_pdf(file_path)
            else:
                raw_text = self._extract_text_from_native_pdf(doc)

            # 关闭文档
            doc.close()

            if not raw_text:
                return None

            # 应用文本处理选项
            if options.get('remove_headers_footers', False):
                raw_text = self._remove_headers_footers(raw_text)

            if options.get('remove_page_numbers', False):
                raw_text = self._remove_page_numbers(raw_text)

            # 应用段落优化
            if options.get('optimize_text', True):
                return self.text_optimizer.optimize_text(raw_text)
            else:
                return raw_text

        except Exception as e:
            logger.error(f"解析PDF文件错误 {file_path}: {e}")
            return None

    async def parse_async(self, file_path: str, **options) -> Optional[str]:
        """
        异步解析PDF文件

        Args:
            file_path: PDF文件路径
            **options: 可选参数
                - optimize_text: 是否优化文本段落（默认True）
                - remove_headers_footers: 是否移除页眉页脚（默认False）
                - remove_page_numbers: 是否移除页码（默认False）
                - use_ocr: 是否对扫描件使用OCR（默认True）

        Returns:
            提取的文本内容
        """
        try:
            # 在线程池中打开PDF文档（IO操作）
            doc = await asyncio.to_thread(pymupdf.open, file_path)

            use_ocr = options.get('use_ocr', True)
            # 检测是否为扫描PDF
            is_scanned = await asyncio.to_thread(self._is_scanned_pdf, doc)

            if use_ocr and is_scanned:
                # 使用异步OCR提取文本
                raw_text = await self._extract_text_from_scanned_pdf_async(file_path)
            else:
                # 在线程池中提取原生PDF文本
                raw_text = await asyncio.to_thread(self._extract_text_from_native_pdf, doc)

            # 关闭文档
            await asyncio.to_thread(doc.close)

            if not raw_text:
                return None

            # 应用文本处理选项（在线程池中执行）
            if options.get('remove_headers_footers', False):
                raw_text = await asyncio.to_thread(self._remove_headers_footers, raw_text)

            if options.get('remove_page_numbers', False):
                raw_text = await asyncio.to_thread(self._remove_page_numbers, raw_text)

            # 应用段落优化
            if options.get('optimize_text', True):
                return await asyncio.to_thread(self.text_optimizer.optimize_text, raw_text)
            else:
                return raw_text

        except Exception as e:
            logger.error(f"异步解析PDF文件错误 {file_path}: {e}")
            return None