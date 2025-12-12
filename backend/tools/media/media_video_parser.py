"""
视频文件解析器

从视频文件中提取音轨并使用阿里百炼ASR服务进行语音识别。
支持的格式：MP4、AVI、MOV、MKV、WMV、FLV等。
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..readers.readers_base import BaseParser
from .media_asr_service import ASRService, get_asr_service

# 配置日志记录器
logger = logging.getLogger(__name__)

class VideoParser(BaseParser):
    """
    视频文件解析器

    从视频文件中提取音轨，然后使用阿里百炼ASR服务进行语音识别。
    支持多种视频格式，包括电影、会议录像、教学视频等。
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化视频解析器

        Args:
            api_key: 阿里云API密钥，可选
        """
        self.api_key = api_key
        self._asr_service: Optional[ASRService] = None
        self.temp_dir = tempfile.gettempdir()

    @property
    def asr_service(self):
        """延迟初始化ASR服务"""
        if self._asr_service is None:
            self._asr_service = get_asr_service(self.api_key)
        return self._asr_service

    def parse(self, file_path: str) -> Optional[str]:
        """
        解析视频文件并提取其中的语音文字

        Args:
            file_path: 视频文件路径

        Returns:
            提取的文字内容，失败返回None
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                logger.warning(f"视频文件不存在: {file_path}")
                return None

            # 检查文件格式是否支持
            if not self.is_supported(file_path):
                logger.warning(f"不支持的视频格式: {file_path}")
                return None

            logger.info(f"开始处理视频文件: {file_path}")

            # 提取音轨
            audio_file_path = self.extract_audio(file_path)
            if not audio_file_path:
                logger.warning("音轨提取失败")
                # 检查是否是因为缺少依赖
                deps = self.check_dependencies()
                if not any(deps.values()):
                    return "视频解析需要FFmpeg或moviepy依赖，当前环境不可用"
                return None

            try:
                # 使用ASR服务识别音轨
                logger.debug("开始语音识别...")
                text_content = asyncio.run(self.asr_service.recognize_file(audio_file_path))

                if text_content:
                    # 清理提取的文字
                    cleaned_text = self._clean_extracted_text(text_content)
                    logger.info(f"视频语音识别成功，文字长度: {len(cleaned_text)} 字符")
                    return cleaned_text
                else:
                    logger.warning("视频语音识别失败")
                    return None

            finally:
                # 清理临时音频文件
                self._cleanup_temp_file(audio_file_path)

        except Exception as e:
            logger.error(f"视频解析错误 {file_path}: {e}")
            return None

    def parse_with_details(self, file_path: str, options: Optional[dict] = None) -> Optional[dict]:
        """
        使用详细选项解析视频文件

        Args:
            file_path: 视频文件路径
            options: 解析选项字典
                - language: 语言代码
                - extract_audio_only: 是否只提取音频不识别
                - audio_quality: 音频质量 ('low', 'medium', 'high')
                - start_time: 开始时间（秒）
                - duration: 持续时间（秒）

        Returns:
            包含详细信息的解析结果字典，失败返回None
        """
        try:
            if not self.is_supported(file_path):
                logger.warning(f"不支持的视频格式: {file_path}")
                return None

            # 默认选项
            default_options = {
                'language': 'zh',
                'extract_audio_only': False,
                'audio_quality': 'medium',
                'start_time': None,
                'duration': None
            }

            if options:
                default_options.update(options)

            logger.info(f"开始详细处理视频文件: {file_path}")

            # 获取视频信息
            video_info = self.get_video_info(file_path)

            # 提取音轨（带选项）
            audio_file_path = self._extract_audio_with_options(file_path, default_options)
            if not audio_file_path:
                logger.warning("音轨提取失败")
                return None

            try:
                result = {
                    'video_info': video_info,
                    'audio_file': audio_file_path,
                    'text': '',
                    'recognition_details': {}
                }

                # 如果只是提取音频，返回音频文件路径
                if default_options['extract_audio_only']:
                    result['text'] = f"音频已提取到: {audio_file_path}"
                    return result

                # 进行语音识别
                recognition_result = asyncio.run(self.asr_service.recognize_file_advanced(
                    audio_file_path,
                    {'language': default_options['language']}
                ))

                if recognition_result:
                    result['text'] = recognition_result.get('text', '')
                    result['recognition_details'] = recognition_result
                    logger.info(f"详细视频语音识别成功，文字长度: {len(str(result['text']))} 字符")
                    return result
                else:
                    logger.warning("详细视频语音识别失败")
                    return None

            finally:
                # 如果不是只提取音频，清理临时文件
                if not default_options['extract_audio_only']:
                    self._cleanup_temp_file(audio_file_path)

        except Exception as e:
            logger.error(f"详细视频解析错误 {file_path}: {e}")
            return None

    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        从视频文件中提取音轨

        Args:
            video_path: 视频文件路径

        Returns:
            提取的音频文件路径，失败返回None
        """
        try:
            # 生成临时音频文件名
            video_name = Path(video_path).stem
            temp_audio_path = os.path.join(self.temp_dir, f"{video_name}_audio.wav")

            # 尝试使用不同的音频提取方法
            success = False

            # 方法1: 使用moviepy（推荐）
            try:
                from moviepy import VideoFileClip

                logger.debug("使用MoviePy提取音轨...")
                with VideoFileClip(video_path) as video:
                    if video.audio is not None:
                        video.audio.write_audiofile(
                            temp_audio_path,
                            verbose=False,
                            logger=None
                        )
                        success = True
                    else:
                        logger.warning("视频文件没有音轨")
                        return None

            except ImportError:
                logger.debug("MoviePy未安装，尝试其他方法...")
            except Exception as e:
                logger.debug(f"MoviePy提取失败: {e}，尝试其他方法...")

            # 方法2: 使用ffmpeg-python
            if not success:
                try:
                    import ffmpeg

                    logger.debug("使用FFmpeg提取音轨...")
                    (
                        ffmpeg
                        .input(video_path)
                        .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    success = True

                except ImportError:
                    logger.debug("ffmpeg-python未安装，尝试系统FFmpeg...")
                except Exception as e:
                    logger.debug(f"ffmpeg-python提取失败: {e}，尝试系统FFmpeg...")

            # 方法3: 使用系统FFmpeg命令
            if not success:
                try:
                    logger.debug("使用系统FFmpeg提取音轨...")
                    cmd = [
                        'ffmpeg',
                        '-i', video_path,
                        '-vn',  # 不要视频
                        '-acodec', 'pcm_s16le',  # 音频编码
                        '-ac', '1',  # 单声道
                        '-ar', '16000',  # 采样率
                        '-y',  # 覆盖输出文件
                        temp_audio_path
                    ]

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5分钟超时
                    )

                    if result.returncode == 0:
                        success = True
                    else:
                        logger.error(f"FFmpeg命令失败: {result.stderr}")

                except FileNotFoundError:
                    logger.warning("系统未安装FFmpeg")
                except subprocess.TimeoutExpired:
                    logger.error("FFmpeg处理超时")
                except Exception as e:
                    logger.error(f"系统FFmpeg提取失败: {e}")

            # 检查是否成功生成音频文件
            if success and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                logger.info(f"音轨提取成功: {temp_audio_path}")
                return temp_audio_path
            else:
                logger.warning("音轨提取失败")
                self._cleanup_temp_file(temp_audio_path)
                return None

        except Exception as e:
            logger.error(f"音轨提取异常: {e}")
            return None

    def _extract_audio_with_options(self, video_path: str, options: dict) -> Optional[str]:
        """
        使用选项提取音轨

        Args:
            video_path: 视频文件路径
            options: 提取选项

        Returns:
            提取的音频文件路径，失败返回None
        """
        try:
            # 生成临时音频文件名
            video_name = Path(video_path).stem
            temp_audio_path = os.path.join(self.temp_dir, f"{video_name}_audio_options.wav")

            # 根据质量设置参数
            quality_settings = {
                'low': {'ar': '8000', 'ab': '64k'},
                'medium': {'ar': '16000', 'ab': '128k'},
                'high': {'ar': '22050', 'ab': '192k'}
            }

            quality = options.get('audio_quality', 'medium')
            settings = quality_settings.get(quality, quality_settings['medium'])

            # 构建FFmpeg命令
            try:
                import subprocess

                cmd = ['ffmpeg', '-i', video_path]

                # 添加时间范围选项
                if options.get('start_time') is not None:
                    cmd.extend(['-ss', str(options['start_time'])])

                if options.get('duration') is not None:
                    cmd.extend(['-t', str(options['duration'])])

                # 音频选项
                cmd.extend([
                    '-vn',  # 不要视频
                    '-acodec', 'pcm_s16le',
                    '-ac', '1',  # 单声道
                    '-ar', settings['ar'],
                    '-y',
                    temp_audio_path
                ])

                logger.debug(f"使用选项提取音轨（质量: {quality}）...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0 and os.path.exists(temp_audio_path):
                    logger.info(f"选项音轨提取成功: {temp_audio_path}")
                    return temp_audio_path
                else:
                    logger.error(f"选项音轨提取失败: {result.stderr}")
                    return self.extract_audio(video_path)  # 回退到基础方法

            except Exception as e:
                logger.debug(f"选项音轨提取异常: {e}，使用基础方法")
                return self.extract_audio(video_path)

        except Exception as e:
            logger.error(f"选项音轨提取错误: {e}")
            return None

    def _cleanup_temp_file(self, file_path: str):
        """
        清理临时文件

        Args:
            file_path: 要清理的文件路径
        """
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"临时文件已清理: {file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")

    def _clean_extracted_text(self, text: str) -> str:
        """
        清理提取的文字内容

        Args:
            text: 原始文字

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

        return text

    def get_video_info(self, file_path: str) -> Optional[dict]:
        """
        获取视频文件信息

        Args:
            file_path: 视频文件路径

        Returns:
            视频信息字典，失败返回None
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                return None

            basic_info = {
                'filename': file_path_obj.name,
                'size': file_path_obj.stat().st_size,
                'format': file_path_obj.suffix.lower(),
                'size_mb': round(file_path_obj.stat().st_size / (1024 * 1024), 2),
                'supported': self.is_supported(file_path)
            }

            # 尝试获取详细视频信息
            try:
                from moviepy import VideoFileClip

                with VideoFileClip(file_path) as video:
                    basic_info.update({
                        'duration': round(video.duration, 2),
                        'duration_formatted': self._format_duration(video.duration),
                        'fps': video.fps,
                        'size_pixels': (video.w, video.h),
                        'has_audio': video.audio is not None
                    })

            except ImportError:
                basic_info['duration'] = None
                basic_info['duration_formatted'] = "需安装MoviePy库获取详细信息"
            except Exception:
                basic_info['duration'] = None
                basic_info['duration_formatted'] = "无法获取视频信息"

            return basic_info

        except Exception as e:
            logger.error(f"获取视频信息失败 {file_path}: {e}")
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

    def get_supported_extensions(self) -> list[str]:
        """获取支持的视频文件扩展名"""
        return [
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',
            '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts',
            '.vob', '.rm', '.rmvb', '.asf', '.divx', '.xvid'
        ]

    def is_supported(self, file_path: str) -> bool:
        """
        检查文件是否为支持的视频格式

        Args:
            file_path: 文件路径

        Returns:
            是否支持
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_extensions()

    def estimate_processing_time(self, file_path: str) -> Optional[str]:
        """
        估算视频处理所需时间

        Args:
            file_path: 视频文件路径

        Returns:
            估算时间的描述字符串
        """
        try:
            video_info = self.get_video_info(file_path)
            if not video_info or video_info.get('duration') is None:
                return "无法估算（无法获取视频时长）"

            duration = video_info['duration']
            size_mb = video_info['size_mb']

            # 估算音频提取时间（通常为视频时长的5%-20%）
            extract_time_min = duration * 0.05
            extract_time_max = duration * 0.2

            # 估算ASR处理时间（通常为音频时长的10%-50%）
            asr_time_min = duration * 0.1
            asr_time_max = duration * 0.5

            total_min = extract_time_min + asr_time_min
            total_max = extract_time_max + asr_time_max

            return (f"预计 {total_min:.1f} - {total_max:.1f} 秒\n"
                   f"（视频时长: {video_info.get('duration_formatted', 'N/A')}，"
                   f"文件大小: {size_mb:.1f}MB）")

        except Exception:
            return "无法估算"

    def check_dependencies(self) -> dict[str, bool]:
        """
        检查依赖库的安装状态

        Returns:
            依赖库安装状态字典
        """
        dependencies = {
            'moviepy': False,
            'ffmpeg-python': False,
            'system_ffmpeg': False
        }

        # 检查MoviePy
        import importlib.util
        dependencies['moviepy'] = importlib.util.find_spec('moviepy') is not None

        # 检查ffmpeg-python
        dependencies['ffmpeg-python'] = importlib.util.find_spec('ffmpeg') is not None

        # 检查系统FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True, timeout=5)
            dependencies['system_ffmpeg'] = result.returncode == 0
        except Exception:
            pass

        return dependencies

# 注册解析器到工厂 - 已移除，现在通过配置管理
# ParserFactory.register_parser(VideoParser)

if __name__ == "__main__":
    # 测试示例
    try:
        parser = VideoParser()

        # 检查依赖
        deps = parser.check_dependencies()
        logger.info("依赖库状态:")
        for lib, status in deps.items():
            status_text = "✓ 已安装" if status else "✗ 未安装"
            logger.info(f"  {lib}: {status_text}")

        # 如果有可用的依赖，测试视频文件
        if any(deps.values()):
            test_file = "test_video.mp4"
            if Path(test_file).exists():
                logger.info(f"\n开始测试视频文件: {test_file}")

                # 获取视频信息
                info = parser.get_video_info(test_file)
                if info:
                    logger.info("视频信息:")
                    for key, value in info.items():
                        logger.info(f"  {key}: {value}")

                # 估算处理时间
                estimate = parser.estimate_processing_time(test_file)
                logger.info(f"\n处理时间估算: {estimate}")

                # 如果视频较短，进行实际处理
                if info and info.get('duration', 0) < 60:  # 小于1分钟
                    logger.info("\n开始处理短视频...")
                    result = parser.parse(test_file)
                    if result:
                        logger.info(f"识别结果: {result[:200]}...")
        else:
            logger.warning("\n请安装以下任一依赖库以使用视频处理功能:")
            logger.warning("  pip install moviepy")
            logger.warning("  pip install ffmpeg-python")
            logger.warning("  或安装系统FFmpeg")

        # 显示支持的格式
        logger.info(f"\n支持的视频格式: {', '.join(parser.get_supported_extensions())}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.error("请确保设置了DASHSCOPE_API_KEY环境变量")
