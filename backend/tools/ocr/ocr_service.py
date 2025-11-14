"""
OCR服务类 - 封装MinerU API调用

基于MinerU API实现的OCR服务，支持文本提取功能。
使用同步ZIP返回模式，提供与MonkeyOCR兼容的接口。
"""

import asyncio
import io
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import aiofiles
import httpx

from backend.infrastructure.http.async_client_manager import get_async_client

from .ocr_config import ocr_config

logger = logging.getLogger(__name__)

class OCRService:
    """
    MinerU API服务封装类 - 异步实现
    """

    def __init__(self, base_url: str | None = None):
        """
        初始化OCR服务

        Args:
            base_url: MinerU API基础URL (可选，默认使用环境变量配置)
        """
        # 使用传入的URL或配置文件中的URL
        self.base_url = base_url or ocr_config.base_url
        self.config = ocr_config

        # 配置验证
        config_errors = self.config.validate()
        if config_errors:
            logger.warning(f"MinerU OCR配置验证警告: {config_errors}")

        # 设置超时（OCR 操作耗时较长，使用单一超时值）
        self.timeout = self.config.timeout

        if self.config.debug_mode:
            logger.info(f"MinerU OCR服务已初始化: {self.base_url}")
            logger.debug(str(self.config))

    async def extract_text(self, file_path: str) -> dict | None:
        """
        从图像或PDF中提取文本 - 使用MinerU API（异步ZIP模式，带重试机制）

        Args:
            file_path: 文件路径

        Returns:
            提取的结果字典，包含以下字段（兼容MonkeyOCR格式）:
            {
                'output_markdown': str,  # Markdown格式文本
                'json_content': str,     # JSON格式的详细结构数据
                'content_list': str,     # 内容列表（MinerU新增）
                'task_id': None,         # MinerU不需要task_id
                'download_url': None     # MinerU不需要download_url
            }
            失败返回None
        """
        import asyncio

        filename = Path(file_path).name
        logger.info(f"🔧 MinerU OCR 处理文件: {file_path}")

        # 重试循环
        for attempt in range(self.config.max_retries):
            try:
                # 异步读取文件内容
                async with aiofiles.open(file_path, 'rb') as f:
                    file_content = await f.read()
                    files = {
                        'files': (filename, file_content, self._get_mime_type(file_path))
                    }

                    # MinerU API 参数
                    data = {
                        'return_md': 'true',
                        'return_middle_json': 'true',
                        'return_content_list': 'true',
                        'response_format_zip': str(self.config.response_format_zip).lower(),
                        'backend': self.config.backend,
                        'parse_method': self.config.parse_method,
                        'formula_enable': str(self.config.formula_enable).lower(),
                        'table_enable': str(self.config.table_enable).lower()
                    }

                    # 异步调用 MinerU API
                    client = await get_async_client()
                    headers = self.config.get_headers()

                    response = await client.post(
                        f"{self.base_url}/file_parse",
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=self.timeout
                    )

                # 验证响应
                if response.status_code != 200:
                    logger.error(f"❌ MinerU API 错误: HTTP {response.status_code}")
                    if attempt < self.config.max_retries - 1:
                        logger.info(f"⏳ 重试 {attempt + 1}/{self.config.max_retries - 1}...")
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    return None

                # 解析 ZIP 响应
                if self.config.response_format_zip:
                    result = await self._parse_mineru_zip(response.content, filename)
                else:
                    # JSON 模式（可选支持）
                    result = await self._parse_mineru_json(response.json(), filename)

                logger.info(f"✅ MinerU OCR 处理成功: {filename}")
                return result

            except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
                # 网络相关错误，可以重试
                error_type = type(e).__name__
                logger.warning(f"⚠️ MinerU OCR {error_type}: {e!s}")

                if attempt < self.config.max_retries - 1:
                    retry_delay = self.config.retry_delay * (attempt + 1)  # 指数退避
                    logger.info(f"⏳ 等待 {retry_delay}秒 后重试 ({attempt + 1}/{self.config.max_retries - 1})...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"❌ MinerU OCR 在 {self.config.max_retries} 次重试后仍失败")
                    return None

            except Exception as e:
                # 其他错误，不重试
                logger.error(f"❌ MinerU OCR 错误: {e}", exc_info=True)
                return None

        return None

    def extract_text_sync(self, file_path: str) -> dict | None:
        """
        从图像或PDF中提取文本 - 同步版本（用于同步上下文）

        在同步代码中调用异步OCR服务时使用此方法。
        它会在新线程中创建独立的事件循环来运行异步代码。

        Args:
            file_path: 文件路径

        Returns:
            提取的结果字典，失败返回None
        """
        import concurrent.futures

        def run_in_thread():
            """在新线程中运行异步OCR，避免事件循环冲突"""
            return asyncio.run(self.extract_text(file_path))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                # 使用配置的超时时间
                return cast("dict | None", future.result(timeout=self.timeout))
        except concurrent.futures.TimeoutError:
            logger.error(f"❌ OCR同步调用超时: {file_path} (超时时间: {self.timeout}秒)")
            return None
        except Exception as e:
            logger.error(f"❌ OCR同步调用失败: {e}", exc_info=True)
            return None

    def is_supported_format(self, file_path: str) -> bool:
        """
        检查文件格式是否支持OCR

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        return self.config.is_supported_format(file_path)

    def is_file_size_valid(self, file_path: str) -> bool:
        """
        检查文件大小是否在限制范围内

        Args:
            file_path: 文件路径

        Returns:
            文件大小是否有效
        """
        return self.config.is_file_size_valid(file_path)

    def _get_mime_type(self, file_path: str) -> str:
        """获取文件的MIME类型"""
        file_ext = Path(file_path).suffix.lower()
        mime_type, _ = mimetypes.guess_type(file_path)

        # 常见文件类型映射
        ext_to_mime = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.gif': 'image/gif'
        }

        return ext_to_mime.get(file_ext, mime_type or 'application/octet-stream')

    async def _parse_mineru_zip(self, zip_content: bytes, filename: str) -> dict | None:
        """解析 MinerU ZIP 响应（异步版本）

        Args:
            zip_content: ZIP文件二进制内容
            filename: 原始文件名

        Returns:
            解析结果字典，包含兼容MonkeyOCR的字段
        """
        temp_extract_dir = None

        try:
            # 创建临时解压目录
            temp_extract_dir = tempfile.mkdtemp()

            # 解压 ZIP（从内存）
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_file:
                zip_file.extractall(temp_extract_dir)
                file_list = zip_file.namelist()
                logger.debug(f"📦 ZIP文件内容: {', '.join(file_list)}")

            # MinerU ZIP 结构: filename_stem/filename_stem.md, filename_stem_middle.json, filename_stem_content_list.json
            base_name = Path(filename).stem
            folder_path = Path(temp_extract_dir) / base_name

            # 读取文件（异步）
            md_file = folder_path / f"{base_name}.md"
            middle_json_file = folder_path / f"{base_name}_middle.json"
            content_list_file = folder_path / f"{base_name}_content_list.json"

            output_markdown = None
            json_content = None
            content_list = None

            if md_file.exists():
                async with aiofiles.open(md_file, encoding='utf-8') as f:
                    output_markdown = await f.read()
                logger.debug(f"📄 读取 Markdown: {md_file.name}")

            if middle_json_file.exists():
                async with aiofiles.open(middle_json_file, encoding='utf-8') as f:
                    json_content = await f.read()
                logger.debug(f"📄 读取 middle.json: {middle_json_file.name}")

            if content_list_file.exists():
                async with aiofiles.open(content_list_file, encoding='utf-8') as f:
                    content_list = await f.read()
                logger.debug(f"📄 读取 content_list.json: {content_list_file.name}")

            # 返回兼容MonkeyOCR的格式
            return {
                'output_markdown': output_markdown,
                'json_content': json_content,
                'content_list': content_list,  # MinerU 新增字段
                'task_id': None,  # MinerU 不需要
                'download_url': None,  # MinerU 不需要
                'zip_files': file_list  # 保留用于调试
            }

        except Exception as e:
            logger.error(f"❌ 解析 MinerU ZIP 失败: {e}", exc_info=True)
            return None

        finally:
            # 清理临时文件
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                try:
                    shutil.rmtree(temp_extract_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ 清理临时文件失败: {cleanup_error}")

    async def _parse_mineru_json(self, response_data: dict, filename: str) -> dict | None:
        """解析 MinerU JSON 响应（可选模式，异步）

        Args:
            response_data: API 返回的 JSON 数据
            filename: 原始文件名

        Returns:
            解析结果字典
        """
        try:
            base_name = Path(filename).stem
            results = response_data.get('results', {})

            if base_name not in results:
                logger.error(f"❌ 结果中未找到文件: {base_name}")
                return None

            file_result = results[base_name]

            # 提取字段
            md_content = file_result.get('md_content', '')
            middle_json = file_result.get('middle_json', '')  # JSON字符串
            content_list = file_result.get('content_list', [])

            # 转换为兼容格式
            return {
                'output_markdown': md_content,
                'json_content': middle_json,  # 保持字符串格式，与MonkeyOCR一致
                'content_list': json.dumps(content_list, ensure_ascii=False) if isinstance(content_list, list) else content_list,
                'task_id': None,
                'download_url': None
            }

        except Exception as e:
            logger.error(f"❌ 解析 MinerU JSON 失败: {e}", exc_info=True)
            return None

# 全局OCR服务实例
_ocr_service = None

def get_ocr_service() -> OCRService:
    """
    获取全局OCR服务实例（单例模式）

    Returns:
        OCR服务实例
    """
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service

if __name__ == "__main__":
    # 测试示例
    ocr = OCRService()
