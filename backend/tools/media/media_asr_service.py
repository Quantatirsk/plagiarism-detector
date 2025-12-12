"""
阿里百炼ASR服务封装类

提供语音识别服务，支持：
- 实时语音识别
- 文件语音识别
- 多种音频格式支持
- 批量处理
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any, cast

# 配置日志记录器
logger = logging.getLogger(__name__)
try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback
    dashscope_available = True
except ImportError as e:
    logger.error(f"Warning: dashscope import error: {e}")
    dashscope = None
    Recognition = None
    RecognitionCallback = None
    dashscope_available = False

class ASRService:
    """
    阿里百炼自动语音识别服务封装类
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化ASR服务

        Args:
            api_key: 阿里云API密钥，如果不提供则从环境变量获取
        """
        if not dashscope_available:
            raise ImportError("dashscope library is not installed. Please install it with: pip install dashscope")

        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请提供DASHSCOPE_API_KEY环境变量或直接传入api_key参数")

        # 设置API密钥
        os.environ['DASHSCOPE_API_KEY'] = self.api_key
        if dashscope:
            dashscope.api_key = self.api_key

        # 线程池用于执行同步的 dashscope SDK 调用
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 支持的音频格式
        self.supported_formats = {
            '.wav', '.mp3', '.m4a', '.flac', '.aac',
            '.ogg', '.wma', '.amr', '.mp4', '.avi',
            '.mov', '.mkv', '.wmv', '.flv'
        }

    def _recognize_file_sync(self, file_path: str, language: str = 'zh') -> Optional[str]:
        """
        同步方法：识别音频文件中的语音并转换为文字（内部使用）

        Args:
            file_path: 音频文件路径
            language: 语言代码 ('zh' 中文, 'en' 英文)

        Returns:
            识别的文字内容，失败返回None
        """
        _ = language  # 参数保留用于将来扩展，当前模型自动检测语言
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                logger.warning(f"音频文件不存在: {file_path}")
                return None

            # 检查文件格式
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                logger.warning(f"不支持的音频格式: {file_ext}")
                return None

            logger.debug(f"开始识别音频文件: {file_path}")

            # 使用Recognition进行实时识别
            callback_base = cast(Any, RecognitionCallback)

            class ASRCallback(callback_base):
                def __init__(self):
                    self.full_text: list[str] = []

                def on_open(self) -> None:
                    pass

                def on_complete(self) -> None:
                    pass

                def on_error(self, result: Any) -> None:
                    logger.error(f"识别错误: {result}")

                def on_event(self, result: Any) -> None:
                    # 收集识别结果
                    if result.get_sentence():
                        sentence = result.get_sentence()
                        if 'text' in sentence:
                            self.full_text.append(sentence['text'])

            # 创建回调实例
            callback = ASRCallback()

            # 获取文件格式
            format_name = file_ext[1:] if file_ext else 'mp3'

            # 创建识别实例
            recognition = cast(Any, Recognition)(
                model='paraformer-realtime-v2',
                format=format_name,
                sample_rate=16000,
                callback=callback
            )

            # 读取音频文件
            with open(file_path, 'rb') as f:
                audio_data = f.read()

            # 开始识别
            recognition.start()
            recognition.send_audio_frame(audio_data)
            recognition.stop()

            # 返回识别结果
            if callback.full_text:
                text_result = ''.join(callback.full_text)
                logger.info(f"语音识别成功，识别文字长度: {len(text_result)} 字符")
                return text_result
            else:
                logger.info("语音识别结果为空")
                return None

        except Exception as e:
            logger.info(f"语音识别异常 {file_path}: {e}")
            return None

    async def recognize_file(self, file_path: str, language: str = 'zh') -> Optional[str]:
        """
        识别音频文件中的语音并转换为文字

        Args:
            file_path: 音频文件路径
            language: 语言代码 ('zh' 中文, 'en' 英文)

        Returns:
            识别的文字内容，失败返回None
        """
        # 在线程池中执行同步的 dashscope SDK 调用
        return await asyncio.to_thread(self._recognize_file_sync, file_path, language)

    def _recognize_file_advanced_sync(self, file_path: str, options: Optional[dict] = None) -> Optional[dict]:
        """
        高级语音识别功能，返回详细信息

        Args:
            file_path: 音频文件路径
            options: 识别选项字典
                - language: 语言代码
                - enable_words: 是否返回词级别时间戳
                - enable_speaker_diarization: 是否启用说话人分离
                - max_sentence_length: 最大句子长度

        Returns:
            包含详细识别信息的字典，失败返回None
        """
        try:
            # 默认选项
            default_options = {
                'language': 'zh',
                'enable_words': True,
                'enable_speaker_diarization': False,
                'max_sentence_length': 200
            }

            if options:
                default_options.update(options)

            # 检查文件
            if not Path(file_path).exists():
                logger.warning(f"音频文件不存在: {file_path}")
                return None

            logger.debug(f"开始高级语音识别: {file_path}")

            # 使用Recognition进行高级识别
            callback_base = cast(Any, RecognitionCallback)

            class AdvancedASRCallback(callback_base):
                def __init__(self):
                    self.sentences: list[dict] = []
                    self.words: list[dict] = []
                    self.full_text: list[str] = []

                def on_open(self) -> None:
                    pass

                def on_complete(self) -> None:
                    pass

                def on_error(self, result: Any) -> None:
                    logger.error(f"识别错误: {result}")

                def on_event(self, result: Any) -> None:
                    # 收集识别结果
                    if result.get_sentence():
                        sentence = result.get_sentence()
                        if 'text' in sentence:
                            sentence_info = {
                                'text': sentence['text'],
                                'start_time': (sentence.get('begin_time') or 0) / 1000.0,  # 转换为秒
                                'end_time': (sentence.get('end_time') or 0) / 1000.0,
                                'confidence': sentence.get('confidence', 1.0)
                            }
                            self.sentences.append(sentence_info)
                            self.full_text.append(sentence['text'])

                            # 收集词级别信息
                            if 'words' in sentence:
                                for word in sentence['words']:
                                    word_info = {
                                        'word': word.get('text', ''),
                                        'start_time': (word.get('begin_time') or 0) / 1000.0,
                                        'end_time': (word.get('end_time') or 0) / 1000.0,
                                        'confidence': word.get('confidence', 1.0)
                                    }
                                    self.words.append(word_info)

            # 创建回调实例
            callback = AdvancedASRCallback()

            # 获取文件格式
            file_ext = Path(file_path).suffix.lower()
            format_name = file_ext[1:] if file_ext else 'mp3'

            # 创建识别实例
            recognition = cast(Any, Recognition)(
                model='paraformer-realtime-v2',
                format=format_name,
                sample_rate=16000,
                callback=callback
            )

            # 读取音频文件
            with open(file_path, 'rb') as f:
                audio_data = f.read()

            # 开始识别
            recognition.start()
            recognition.send_audio_frame(audio_data)
            recognition.stop()

            # 构建返回结果
            recognition_result = {
                'text': ''.join(callback.full_text),
                'sentences': callback.sentences,
                'words': callback.words,
                'speakers': [],
                'confidence': 1.0,
                'duration': 0.0
            }

            # 计算总时长
            if callback.sentences:
                last_sentence = callback.sentences[-1]
                recognition_result['duration'] = last_sentence.get('end_time', 0)

            if recognition_result['text']:
                logger.info(f"高级语音识别成功，文字长度: {len(str(recognition_result['text']))} 字符")
                return recognition_result
            else:
                logger.warning("高级语音识别失败: 无识别结果")
                return None

        except Exception as e:
            logger.info(f"高级语音识别异常 {file_path}: {e}")
            return None

    async def recognize_file_advanced(self, file_path: str, options: Optional[dict] = None) -> Optional[dict]:
        """
        高级语音识别功能，返回详细信息

        Args:
            file_path: 音频文件路径
            options: 识别选项字典

        Returns:
            包含详细识别信息的字典，失败返回None
        """
        # 在线程池中执行同步的 dashscope SDK 调用
        return await asyncio.to_thread(self._recognize_file_advanced_sync, file_path, options)

    async def batch_recognize(self, file_paths: list[str], language: str = 'zh') -> dict[str, Optional[str]]:
        """
        批量识别多个音频文件

        Args:
            file_paths: 音频文件路径列表
            language: 语言代码

        Returns:
            文件路径到识别结果的映射字典
        """
        results = {}

        logger.debug(f"开始批量识别 {len(file_paths)} 个音频文件")

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"处理第 {i}/{len(file_paths)} 个文件: {Path(file_path).name}")

            result = await self.recognize_file(file_path, language)
            results[file_path] = result

            # 添加延迟以避免API限流
            if i < len(file_paths):
                await asyncio.sleep(1)

        logger.info("批量识别完成")
        return results

    def _read_audio_file(self, file_path: str) -> bytes:
        """
        读取音频文件为字节数据

        Args:
            file_path: 音频文件路径

        Returns:
            音频文件的字节数据
        """
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取音频文件失败 {file_path}: {e}")
            raise

    def get_supported_formats(self) -> list[str]:
        """
        获取支持的音频格式列表

        Returns:
            支持的音频格式扩展名列表
        """
        return list(self.supported_formats)

    def is_supported_format(self, file_path: str) -> bool:
        """
        检查文件格式是否支持

        Args:
            file_path: 文件路径

        Returns:
            是否支持该格式
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats

    def get_file_info(self, file_path: str) -> Optional[dict]:
        """
        获取音频文件信息

        Args:
            file_path: 音频文件路径

        Returns:
            文件信息字典，包含大小、格式等信息
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                return None

            file_info = {
                'filename': file_path_obj.name,
                'size': file_path_obj.stat().st_size,
                'format': file_path_obj.suffix.lower(),
                'size_mb': round(file_path_obj.stat().st_size / (1024 * 1024), 2),
                'supported': self.is_supported_format(file_path)
            }

            # 尝试获取音频时长（需要额外库支持）
            try:
                import mutagen
                audio_file = cast(Any, mutagen).File(file_path)
                if audio_file is not None:
                    file_info['duration'] = round(audio_file.info.length, 2)
                    file_info['duration_formatted'] = self._format_duration(audio_file.info.length)
            except ImportError:
                file_info['duration'] = None
                file_info['duration_formatted'] = "未安装mutagen库"
            except Exception:
                file_info['duration'] = None
                file_info['duration_formatted'] = "无法获取时长"

            return file_info

        except Exception as e:
            logger.error(f"获取文件信息失败 {file_path}: {e}")
            return None

    def _format_duration(self, seconds: float) -> str:
        """
        格式化时长显示

        Args:
            seconds: 秒数

        Returns:
            格式化的时长字符串
        """
        if seconds < 60:
            return f"{seconds:.1f} 秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes} 分 {secs:.1f} 秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours} 小时 {minutes} 分 {secs:.1f} 秒"

# 全局ASR服务实例（单例模式）
_asr_service = None

def get_asr_service(api_key: Optional[str] = None) -> ASRService:
    """
    获取全局ASR服务实例（单例模式）

    Args:
        api_key: API密钥，仅在首次调用时有效

    Returns:
        ASR服务实例
    """
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService(api_key)
    return _asr_service

async def asr_recognize_file(file_path: str, language: str = 'zh') -> Optional[str]:
    """
    便捷函数：识别音频文件

    Args:
        file_path: 音频文件路径
        language: 语言代码

    Returns:
        识别的文字内容，失败返回None
    """
    return await get_asr_service().recognize_file(file_path, language)

if __name__ == "__main__":
    # 测试示例
    try:
        asr = ASRService()

        # 测试文件信息获取
        test_file = "test_audio.wav"
        if Path(test_file).exists():
            info = asr.get_file_info(test_file)
            if info:
                logger.info("音频文件信息:")
                for key, value in info.items():
                    logger.info(f"  {key}: {value}")

            # 测试语音识别
            result = asr.recognize_file(test_file)
            if result:
                logger.info("\n识别结果:")
                logger.info(result)

        # 显示支持的格式
        logger.info(f"\n支持的音频格式: {', '.join(asr.get_supported_formats())}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.info("请确保设置了DASHSCOPE_API_KEY环境变量")
