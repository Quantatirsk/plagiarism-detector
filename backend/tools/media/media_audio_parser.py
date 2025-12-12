"""
音频文件解析器

使用阿里百炼ASR服务对各种音频格式进行语音识别，提取文字内容。
支持的格式：WAV、MP3、M4A、FLAC、AAC、OGG等。
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from ..readers.readers_base import BaseParser
from .media_asr_service import ASRService, get_asr_service

# 配置日志记录器
logger = logging.getLogger(__name__)
class AudioParser(BaseParser):
    """
    音频文件解析器

    使用阿里百炼ASR服务对音频进行语音识别，提取其中的文字内容。
    支持多种音频格式，包括录音、播客、会议音频等。
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化音频解析器

        Args:
            api_key: 阿里云API密钥，可选
        """
        self.api_key = api_key
        self._asr_service: Optional[ASRService] = None

    @property
    def asr_service(self):
        """延迟初始化ASR服务"""
        if self._asr_service is None:
            self._asr_service = get_asr_service(self.api_key)
        return self._asr_service

    def parse(self, file_path: str) -> Optional[str]:
        """
        使用ASR解析音频文件并提取文字

        Args:
            file_path: 音频文件路径

        Returns:
            提取的文字内容，失败返回None
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                logger.warning(f"音频文件不存在: {file_path}")
                return None

            # 检查文件格式是否支持
            if not self.is_supported(file_path):
                logger.warning(f"不支持的音频格式: {file_path}")
                return None

            logger.debug(f"开始语音识别音频文件: {file_path}")

            # 使用ASR服务提取文字
            text_content = asyncio.run(self.asr_service.recognize_file(file_path))

            if text_content:
                # 清理提取的文字
                cleaned_text = self._clean_extracted_text(text_content)
                logger.info(f"语音识别成功，提取文字长度: {len(cleaned_text)} 字符")
                return cleaned_text
            else:
                logger.warning(f"语音识别失败，未提取到文字内容: {file_path}")
                return None

        except Exception as e:
            logger.error(f"音频解析错误 {file_path}: {e}")
            return None

    def parse_with_details(self, file_path: str, options: Optional[dict] = None) -> Optional[dict]:
        """
        使用详细选项解析音频文件

        Args:
            file_path: 音频文件路径
            options: 解析选项字典
                - language: 语言代码('zh', 'en')
                - enable_words: 是否返回词级别信息
                - enable_speaker_diarization: 是否启用说话人分离

        Returns:
            包含详细信息的解析结果字典，失败返回None
        """
        try:
            if not self.is_supported(file_path):
                logger.warning(f"不支持的音频格式: {file_path}")
                return None

            logger.debug(f"开始详细语音识别: {file_path}")

            # 使用高级ASR服务
            result = asyncio.run(self.asr_service.recognize_file_advanced(file_path, options))

            if result:
                # 添加文件信息
                file_info = self.get_audio_info(file_path)
                if file_info:
                    result['file_info'] = file_info

                logger.info(f"详细语音识别成功，文字长度: {len(result.get('text', ''))} 字符")
                return result
            else:
                logger.warning(f"详细语音识别失败: {file_path}")
                return None

        except Exception as e:
            logger.error(f"详细音频解析错误 {file_path}: {e}")
            return None

    def batch_parse(self, file_paths: list[str], language: str = 'zh') -> dict[str, Optional[str]]:
        """
        批量解析多个音频文件

        Args:
            file_paths: 音频文件路径列表
            language: 语言代码

        Returns:
            文件路径到解析结果的映射字典
        """
        try:
            # 过滤支持的文件
            supported_files = [f for f in file_paths if self.is_supported(f)]
            if not supported_files:
                logger.info("没有支持的音频文件")
                return {}

            logger.debug(f"开始批量音频解析 {len(supported_files)} 个文件")

            # 使用ASR服务批量处理
            results = asyncio.run(self.asr_service.batch_recognize(supported_files, language))

            # 清理结果
            cleaned_results: dict[str, Optional[str]] = {}
            for file_path, text in results.items():
                if text:
                    cleaned_results[file_path] = self._clean_extracted_text(text)
                else:
                    cleaned_results[file_path] = None

            logger.info("批量音频解析完成")
            return cleaned_results

        except Exception as e:
            logger.error(f"批量音频解析错误: {e}")
            return {}

    def _clean_extracted_text(self, text: str) -> str:
        """
        清理ASR提取的文字内容

        Args:
            text: 原始ASR文字

        Returns:
            清理后的文字
        """
        if not text:
            return ""

        # 移除多余的空白字符
        text = text.strip()

        # 将多个连续的空格合并为单个空格
        import re
        text = re.sub(r'\s+', ' ', text)

        # 移除可能的ASR标记符号（如果有的话）
        text = re.sub(r'[<>{}]', '', text)

        return text

    def get_audio_info(self, file_path: str) -> Optional[dict]:
        """
        获取音频文件信息

        Args:
            file_path: 音频文件路径

        Returns:
            音频信息字典，失败返回None
        """
        try:
            return self.asr_service.get_file_info(file_path)
        except Exception as e:
            logger.error(f"获取音频信息错误 {file_path}: {e}")
            return None

    def estimate_recognition_time(self, file_path: str) -> Optional[str]:
        """
        估算语音识别所需时间

        Args:
            file_path: 音频文件路径

        Returns:
            估算时间的描述字符串
        """
        try:
            file_info = self.get_audio_info(file_path)
            if not file_info or file_info.get('duration') is None:
                return "无法估算（无法获取音频时长）"

            duration = file_info['duration']

            # 简单估算：通常ASR处理时间约为音频时长的10%-50%
            min_time = duration * 0.1
            max_time = duration * 0.5

            return f"预计 {min_time:.1f} - {max_time:.1f} 秒（音频时长: {file_info.get('duration_formatted', 'N/A')}）"

        except Exception:
            return "无法估算"

    def get_supported_extensions(self) -> list[str]:
        """获取支持的音频文件扩展名"""
        return [
            '.wav', '.mp3', '.m4a', '.flac', '.aac',
            '.ogg', '.wma', '.amr', '.opus', '.webm'
        ]

    def is_supported(self, file_path: str) -> bool:
        """
        检查文件是否为支持的音频格式

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_extensions()

    def get_language_options(self) -> dict[str, str]:
        """
        获取支持的语言选项

        Returns:
            语言代码到描述的映射字典
        """
        return {
            'zh': '中文（简体）',
            'en': '英文',
            'zh-tw': '中文（繁体）',
            'ja': '日文',
            'ko': '韩文',
            'auto': '自动检测'
        }

    def validate_audio_file(self, file_path: str) -> dict[str, Any]:
        """
        验证音频文件是否适合进行语音识别

        Args:
            file_path: 音频文件路径

        Returns:
            验证结果字典，包含是否有效和相关信息
        """
        validation_result: dict[str, Any] = {
            'valid': False,
            'file_exists': False,
            'format_supported': False,
            'file_readable': False,
            'size_appropriate': False,
            'warnings': [],
            'recommendations': []
        }

        try:
            # 检查文件是否存在
            file_path_obj = Path(file_path)
            validation_result['file_exists'] = file_path_obj.exists()

            if not validation_result['file_exists']:
                validation_result['warnings'].append("文件不存在")
                return validation_result

            # 检查格式支持
            validation_result['format_supported'] = self.is_supported(file_path)
            if not validation_result['format_supported']:
                validation_result['warnings'].append(f"不支持的音频格式: {file_path_obj.suffix}")
                return validation_result

            # 检查文件是否可读
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # 读取前1KB检查
                validation_result['file_readable'] = True
            except Exception:
                validation_result['warnings'].append("文件无法读取或已损坏")
                return validation_result

            # 检查文件大小
            file_size = file_path_obj.stat().st_size
            size_mb = file_size / (1024 * 1024)

            if size_mb > 100:  # 大于100MB
                validation_result['warnings'].append(f"文件较大 ({size_mb:.1f}MB)，处理时间可能较长")
            elif size_mb < 0.1:  # 小于100KB
                validation_result['warnings'].append(f"文件较小 ({size_mb:.1f}MB)，可能识别效果不佳")
            else:
                validation_result['size_appropriate'] = True

            # 获取音频信息并给出建议
            audio_info = self.get_audio_info(file_path)
            if audio_info:
                duration = audio_info.get('duration')
                if duration:
                    if duration > 3600:  # 超过1小时
                        validation_result['recommendations'].append("建议将长音频分段处理以提高效率")
                    elif duration < 5:  # 少于5秒
                        validation_result['recommendations'].append("音频时长较短，识别准确度可能受影响")

            # 综合判断
            validation_result['valid'] = (
                validation_result['file_exists'] and
                validation_result['format_supported'] and
                validation_result['file_readable']
            )

            if validation_result['valid'] and not validation_result['warnings']:
                validation_result['recommendations'].append("文件验证通过，可以进行语音识别")

            return validation_result

        except Exception as e:
            validation_result['warnings'].append(f"验证过程出错: {e}")
            return validation_result

# 注册解析器到工厂 - 已移除，现在通过配置管理
# ParserFactory.register_parser(AudioParser)

if __name__ == "__main__":
    # 测试示例
    try:
        parser = AudioParser()

        # 测试音频文件验证
        test_file = "test_audio.wav"
        validation = parser.validate_audio_file(test_file)

        logger.info("音频文件验证结果:")
        for key, value in validation.items():
            if key not in ['warnings', 'recommendations']:
                logger.info(f"  {key}: {value}")

        if validation['warnings']:
            logger.warning("  警告:")
            for warning in validation['warnings']:
                logger.warning(f"    - {warning}")

        if validation['recommendations']:
            logger.info("  建议:")
            for rec in validation['recommendations']:
                logger.info(f"    - {rec}")

        # 如果文件存在且有效，测试识别
        if validation['valid'] and Path(test_file).exists():
            logger.debug("\n开始识别测试...")
            result = parser.parse(test_file)
            if result:
                logger.info(f"识别结果: {result[:100]}...")

        # 显示支持的格式和语言
        logger.info(f"\n支持的音频格式: {', '.join(parser.get_supported_extensions())}")
        logger.info("\n支持的语言:")
        for code, name in parser.get_language_options().items():
            logger.info(f"  {code}: {name}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.info("请确保设置了DASHSCOPE_API_KEY环境变量")
