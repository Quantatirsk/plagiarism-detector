"""
音视频处理模块

提供音视频文件的语音识别和文本提取功能：
- 音频文件识别转文字 (支持多种格式)
- 视频文件音轨提取和识别
- 调用阿里百炼ASR模型
"""
from typing import Optional

from .media_asr_service import ASRService, get_asr_service
from .media_audio_parser import AudioParser
from .media_video_parser import VideoParser


class MediaService:
    """
    统一的媒体服务接口，整合音频、视频处理功能
    """

    def __init__(self, api_key: Optional[str] = None):
        self.asr_service: Optional[ASRService] = None
        self.audio_parser: Optional[AudioParser] = None
        self.video_parser: Optional[VideoParser] = None
        self.api_key = api_key
        self._init_services()

    def _init_services(self):
        """初始化各种服务"""
        try:
            self.asr_service = get_asr_service(self.api_key)
        except Exception:
            self.asr_service = None

        try:
            self.audio_parser = AudioParser()
        except Exception:
            self.audio_parser = None

        try:
            self.video_parser = VideoParser()
        except Exception:
            self.video_parser = None

    def transcribe_audio(self, file_path: str, **kwargs):
        """音频转写"""
        if self.asr_service:
            return self.asr_service.recognize_file(file_path, **kwargs)
        return None

    def speech_to_text(self, file_path: str, **kwargs):
        """语音转文字（别名）"""
        return self.transcribe_audio(file_path, **kwargs)

    def process_audio(self, file_path: str, task: str = 'transcribe', **kwargs):
        """处理音频文件"""
        if task == 'transcribe':
            return self.transcribe_audio(file_path, **kwargs)
        elif task == 'parse' and self.audio_parser:
            return self.audio_parser.parse(file_path)
        return None

    def extract_audio_from_video(self, video_path: str, **kwargs):
        """从视频提取音频"""
        if self.video_parser:
            return self.video_parser.extract_audio(video_path)
        return None

    def process_video(self, file_path: str, **kwargs):
        """处理视频文件"""
        if self.video_parser:
            return self.video_parser.parse(file_path, **kwargs)
        return None

    def get_supported_formats(self):
        """获取支持的格式"""
        formats: list[str] = []
        if self.asr_service:
            formats.extend(self.asr_service.get_supported_formats())
        if self.audio_parser:
            formats.extend(['wav', 'mp3', 'flac', 'm4a'])
        if self.video_parser:
            formats.extend(['mp4', 'avi', 'mov', 'mkv'])
        return list(set(formats))  # 去重

# 单例实例
_media_service_instance = None

def get_media_service(api_key: Optional[str] = None) -> MediaService:
    """
    获取媒体服务单例实例

    Args:
        api_key: API密钥

    Returns:
        MediaService实例
    """
    global _media_service_instance
    if _media_service_instance is None:
        _media_service_instance = MediaService(api_key)
    return _media_service_instance

__all__ = [
    'ASRService',
    'AudioParser',
    'MediaService',
    'VideoParser',
    'get_asr_service',
    'get_media_service',
]
