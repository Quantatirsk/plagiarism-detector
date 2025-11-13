# 音视频处理工具 (Media)

Refly-AI 的音视频处理工具集，基于阿里百炼ASR服务，提供音频和视频文件的语音识别和文本转换功能。

## 功能特性

- 🎵 **多格式支持**: 支持WAV、MP3、M4A、FLAC、AAC、OGG等音频格式
- 🎬 **视频处理**: 支持MP4、AVI、MOV、MKV等视频格式的音轨提取和识别
- 🗣️ **语音识别**: 高精度的语音转文字功能，支持中英文识别
- ⚡ **高性能**: 基于阿里百炼ASR引擎，识别速度快、准确率高
- 🔄 **批量处理**: 支持多文件批量识别和处理
- 📊 **实时处理**: 支持实时音频流识别
- 🌐 **多语言**: 支持中文、英文等多种语言识别
- 📁 **智能解析**: 自动识别文件格式，选择最佳处理策略

## 支持的文件格式

### 音频格式
| 格式 | 扩展名 | 特点 | 推荐场景 |
|------|--------|------|----------|
| **WAV** | `.wav` | 无损格式，质量最高 | 高质量录音、专业音频 |
| **MP3** | `.mp3` | 压缩格式，文件小 | 常规音频文件、网络传输 |
| **M4A** | `.m4a` | 苹果格式，音质好 | 苹果设备录音、播客 |
| **FLAC** | `.flac` | 无损压缩 | 音乐、高质量音频 |
| **AAC** | `.aac` | 高效压缩 | 流媒体、移动设备 |
| **OGG** | `.ogg` | 开源格式 | 游戏音频、开源项目 |
| **AMR** | `.amr` | 移动通话格式 | 通话录音、语音消息 |

### 视频格式  
| 格式 | 扩展名 | 特点 | 处理方式 |
|------|--------|------|----------|
| **MP4** | `.mp4` | 最常用视频格式 | 提取音轨后识别 |
| **AVI** | `.avi` | 传统视频格式 | 提取音轨后识别 |
| **MOV** | `.mov` | 苹果视频格式 | 提取音轨后识别 |
| **MKV** | `.mkv` | 开源视频格式 | 提取音轨后识别 |
| **WMV** | `.wmv` | 微软视频格式 | 提取音轨后识别 |
| **FLV** | `.flv` | Flash视频格式 | 提取音轨后识别 |

## 快速开始

### 基础音频识别

```python
from tools.media import ASRService

# 方法1: 创建ASR服务实例
asr = ASRService()
text = asr.recognize_file("recording.wav")
if text:
    print("识别结果:", text)

# 方法2: 使用单例模式
from tools.media import get_asr_service
asr = get_asr_service()
text = asr.recognize_file("interview.mp3")
```

### 使用解析器类

```python
from tools.media import AudioParser, VideoParser

# 音频文件解析
audio_parser = AudioParser()
text = audio_parser.parse("speech.wav")

# 视频文件解析（提取音轨并识别）
video_parser = VideoParser()
text = video_parser.parse("meeting_recording.mp4")
```

## 详细用法

### 1. 音频文件识别

```python
from tools.media import ASRService

# 创建ASR服务（需要配置DASHSCOPE_API_KEY环境变量）
asr = ASRService()

# 基础识别
text = asr.recognize_file("audio.wav")
print("识别文本:", text)

# 指定语言识别
chinese_text = asr.recognize_file("chinese_speech.mp3", language='zh')
english_text = asr.recognize_file("english_speech.wav", language='en')

# 检查文件格式支持
if asr.is_supported_format("unknown_file.xyz"):
    print("支持该文件格式")
else:
    print("不支持该文件格式")
```

**语言代码**:
- `zh`: 中文（普通话）
- `en`: 英文
- `zh-yue`: 粤语
- `zh-tw`: 繁体中文

### 2. 视频文件音轨识别

```python
from tools.media import VideoParser

video_parser = VideoParser()

# 视频文件语音识别
text = video_parser.parse("conference.mp4")
if text:
    print("会议录音转文字:")
    print(text)

# 长视频文件处理
long_video_text = video_parser.parse("long_lecture.avi")
```

**视频处理特性**:
- 自动提取音轨
- 支持多种视频编码
- 智能音频质量优化
- 长时间视频分段处理

### 3. 批量文件处理

```python
from tools.media import ASRService
import os
from pathlib import Path

def batch_audio_recognition(input_dir, output_dir):
    """批量音频识别"""
    asr = ASRService()
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    results = []
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if asr.is_supported_format(file_path):
            print(f"正在识别: {filename}")
            
            try:
                text = asr.recognize_file(file_path)
                if text:
                    # 保存识别结果
                    output_file = os.path.join(output_dir, f"{filename}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    results.append({
                        "file": filename,
                        "status": "success",
                        "text_length": len(text),
                        "output_file": output_file
                    })
                    print(f"✓ 成功: {filename}")
                else:
                    results.append({
                        "file": filename,
                        "status": "failed",
                        "error": "识别结果为空"
                    })
                    print(f"✗ 失败: {filename}")
                    
            except Exception as e:
                results.append({
                    "file": filename,
                    "status": "error",
                    "error": str(e)
                })
                print(f"✗ 异常: {filename} - {e}")
    
    return results

# 使用示例
results = batch_audio_recognition("./audio_files", "./transcripts")

# 生成处理报告
successful = [r for r in results if r["status"] == "success"]
failed = [r for r in results if r["status"] != "success"]

print(f"\\n处理完成: 成功 {len(successful)} 个，失败 {len(failed)} 个")
```

### 4. 多语言识别

```python
from tools.media import ASRService

def multi_language_recognition(file_path):
    """多语言音频识别"""
    asr = ASRService()
    
    languages = {
        'zh': '中文',
        'en': '英文',
        'zh-yue': '粤语'
    }
    
    results = {}
    
    for lang_code, lang_name in languages.items():
        try:
            print(f"尝试 {lang_name} 识别...")
            text = asr.recognize_file(file_path, language=lang_code)
            
            if text and len(text.strip()) > 0:
                results[lang_name] = {
                    "text": text,
                    "length": len(text),
                    "confidence": estimate_confidence(text, lang_code)
                }
                print(f"✓ {lang_name} 识别成功: {len(text)} 字符")
            else:
                print(f"✗ {lang_name} 识别失败")
                
        except Exception as e:
            print(f"✗ {lang_name} 识别异常: {e}")
    
    # 选择最佳识别结果
    if results:
        best_result = max(results.items(), key=lambda x: x[1]['confidence'])
        print(f"\\n最佳识别结果: {best_result[0]}")
        return best_result[1]['text']
    
    return None

def estimate_confidence(text, language):
    """估算识别置信度"""
    if not text:
        return 0
    
    # 基于文本长度和语言特征简单估算
    base_score = min(len(text) / 100, 1.0)
    
    if language == 'zh':
        # 中文字符比例
        chinese_chars = sum(1 for c in text if '\\u4e00' <= c <= '\\u9fff')
        if len(text) > 0:
            chinese_ratio = chinese_chars / len(text)
            base_score *= (0.5 + 0.5 * chinese_ratio)
    
    return base_score

# 使用示例
best_text = multi_language_recognition("mixed_language_audio.wav")
if best_text:
    print("识别文本:", best_text)
```

### 5. 音频质量检测和预处理

```python
from tools.media import ASRService
import librosa
import numpy as np

class AudioProcessor:
    """音频预处理器"""
    
    def __init__(self):
        self.asr = ASRService()
    
    def analyze_audio_quality(self, file_path):
        """分析音频质量"""
        try:
            # 加载音频文件
            y, sr = librosa.load(file_path)
            
            # 计算音频特征
            duration = len(y) / sr
            rms_energy = np.sqrt(np.mean(y**2))
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            quality_info = {
                "duration": duration,
                "sample_rate": sr,
                "rms_energy": float(rms_energy),
                "avg_zcr": float(np.mean(zero_crossing_rate)),
                "avg_spectral_centroid": float(np.mean(spectral_centroid))
            }
            
            # 质量评估
            quality_score = self._calculate_quality_score(quality_info)
            quality_info["quality_score"] = quality_score
            quality_info["quality_level"] = self._get_quality_level(quality_score)
            
            return quality_info
            
        except Exception as e:
            return {"error": f"音频分析失败: {e}"}
    
    def _calculate_quality_score(self, info):
        """计算质量分数"""
        score = 0.5  # 基础分数
        
        # 基于采样率
        if info["sample_rate"] >= 16000:
            score += 0.2
        elif info["sample_rate"] >= 8000:
            score += 0.1
        
        # 基于能量
        if info["rms_energy"] > 0.01:
            score += 0.2
        elif info["rms_energy"] > 0.005:
            score += 0.1
        
        # 基于时长
        if 5 <= info["duration"] <= 300:  # 5秒到5分钟
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_quality_level(self, score):
        """获取质量等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "较差"
    
    def recognize_with_quality_check(self, file_path):
        """带质量检查的识别"""
        # 分析音频质量
        quality_info = self.analyze_audio_quality(file_path)
        
        if "error" in quality_info:
            return {"error": quality_info["error"]}
        
        # 执行识别
        text = self.asr.recognize_file(file_path)
        
        result = {
            "text": text,
            "audio_quality": quality_info,
            "success": text is not None and len(text.strip()) > 0
        }
        
        if quality_info["quality_score"] < 0.4:
            result["warning"] = "音频质量较差，识别结果可能不准确"
        
        return result

# 使用示例
processor = AudioProcessor()
result = processor.recognize_with_quality_check("low_quality_audio.wav")

print("识别结果:", result["text"])
print("音频质量:", result["audio_quality"]["quality_level"])
if "warning" in result:
    print("警告:", result["warning"])
```

### 6. 实时音频流处理

```python
from tools.media import ASRService
import threading
import queue
import time

class RealTimeASR:
    """实时语音识别"""
    
    def __init__(self):
        self.asr = ASRService()
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_running = False
        
    def start_recognition(self):
        """开始实时识别"""
        self.is_running = True
        
        # 启动识别线程
        recognition_thread = threading.Thread(target=self._recognition_worker)
        recognition_thread.daemon = True
        recognition_thread.start()
        
        print("实时识别已启动")
    
    def stop_recognition(self):
        """停止实时识别"""
        self.is_running = False
        print("实时识别已停止")
    
    def add_audio_chunk(self, audio_file_path):
        """添加音频片段"""
        if self.is_running:
            self.audio_queue.put(audio_file_path)
    
    def get_recognition_result(self):
        """获取识别结果"""
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _recognition_worker(self):
        """识别工作线程"""
        while self.is_running:
            try:
                # 获取音频文件
                audio_file = self.audio_queue.get(timeout=1.0)
                
                # 执行识别
                text = self.asr.recognize_file(audio_file)
                
                if text:
                    # 添加时间戳
                    result = {
                        "timestamp": time.time(),
                        "text": text,
                        "source_file": audio_file
                    }
                    self.text_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                error_result = {
                    "timestamp": time.time(),
                    "error": str(e),
                    "source_file": audio_file if 'audio_file' in locals() else None
                }
                self.text_queue.put(error_result)

# 使用示例（需要配合音频录制）
real_time_asr = RealTimeASR()
real_time_asr.start_recognition()

# 模拟添加音频片段
audio_files = ["chunk1.wav", "chunk2.wav", "chunk3.wav"]
for audio_file in audio_files:
    real_time_asr.add_audio_chunk(audio_file)
    time.sleep(2)

# 获取识别结果
for i in range(10):  # 检查10次
    result = real_time_asr.get_recognition_result()
    if result:
        if "error" in result:
            print(f"识别错误: {result['error']}")
        else:
            print(f"识别结果: {result['text']}")
    time.sleep(1)

real_time_asr.stop_recognition()
```

## 高级功能

### 1. 语音分段和标记

```python
from tools.media import ASRService
import re
from datetime import datetime, timedelta

def segment_and_timestamp_audio(file_path, segment_duration=30):
    """对长音频进行分段识别并添加时间戳"""
    asr = ASRService()
    
    try:
        # 这里需要先分割音频文件
        # segments = split_audio_file(file_path, segment_duration)
        # 为了演示，假设已经有分段文件
        
        segments_text = []
        current_time = 0
        
        # 假设有分段文件 segment_0.wav, segment_1.wav, ...
        segment_index = 0
        
        while True:
            segment_file = f"segment_{segment_index}.wav"
            
            if not os.path.exists(segment_file):
                break
            
            print(f"识别分段 {segment_index + 1}...")
            text = asr.recognize_file(segment_file)
            
            if text:
                timestamp = str(timedelta(seconds=current_time))
                segments_text.append({
                    "timestamp": timestamp,
                    "duration": segment_duration,
                    "text": text.strip()
                })
            
            current_time += segment_duration
            segment_index += 1
        
        return segments_text
        
    except Exception as e:
        print(f"分段识别失败: {e}")
        return []

def format_transcription(segments):
    """格式化转录结果"""
    formatted_text = "# 语音转录结果\\n\\n"
    formatted_text += f"转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
    
    for i, segment in enumerate(segments, 1):
        formatted_text += f"## 第{i}段 ({segment['timestamp']})\\n\\n"
        formatted_text += f"{segment['text']}\\n\\n"
    
    return formatted_text

# 使用示例
segments = segment_and_timestamp_audio("long_meeting.wav")
formatted_result = format_transcription(segments)

with open("transcription_result.md", "w", encoding="utf-8") as f:
    f.write(formatted_result)

print("转录完成，结果已保存到 transcription_result.md")
```

### 2. 说话人识别和分离

```python
from tools.media import ASRService

class SpeakerDiarization:
    """说话人分离和识别"""
    
    def __init__(self):
        self.asr = ASRService()
    
    def recognize_with_speakers(self, file_path):
        """识别语音并尝试分离说话人"""
        # 基础语音识别
        full_text = self.asr.recognize_file(file_path)
        
        if not full_text:
            return None
        
        # 简单的说话人分离（基于静音检测和文本分析）
        segments = self._detect_speech_segments(full_text)
        
        return {
            "full_text": full_text,
            "speaker_segments": segments,
            "estimated_speakers": len(set(s["speaker"] for s in segments))
        }
    
    def _detect_speech_segments(self, text):
        """检测语音段落（简单实现）"""
        # 基于标点符号分割句子
        sentences = re.split(r'[。！？.!?]', text)
        segments = []
        
        current_speaker = "说话人1"
        speaker_count = 1
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 简单的说话人变化检测（基于语言风格变化）
            if self._detect_speaker_change(sentence, segments):
                speaker_count += 1
                current_speaker = f"说话人{speaker_count}"
            
            segments.append({
                "segment_id": i + 1,
                "speaker": current_speaker,
                "text": sentence,
                "estimated_start_time": i * 2,  # 简单估算
                "confidence": 0.8
            })
        
        return segments
    
    def _detect_speaker_change(self, sentence, previous_segments):
        """检测说话人变化（简单实现）"""
        if not previous_segments:
            return False
        
        # 基于一些简单规则检测说话人变化
        change_indicators = ["你好", "请问", "我觉得", "据我了解", "我的观点"]
        
        return any(indicator in sentence for indicator in change_indicators)

# 使用示例
diarization = SpeakerDiarization()
result = diarization.recognize_with_speakers("conversation.wav")

if result:
    print("完整文本:", result["full_text"])
    print(f"\\n检测到 {result['estimated_speakers']} 个说话人:")
    
    for segment in result["speaker_segments"]:
        print(f"{segment['speaker']} ({segment['estimated_start_time']}s): {segment['text']}")
```

### 3. 会议记录生成

```python
from tools.media import ASRService
from datetime import datetime
import json

class MeetingTranscriber:
    """会议记录生成器"""
    
    def __init__(self):
        self.asr = ASRService()
    
    def transcribe_meeting(self, audio_file, meeting_info=None):
        """生成会议记录"""
        print("开始转录会议录音...")
        
        # 基本信息
        if not meeting_info:
            meeting_info = {
                "title": "会议记录",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S")
            }
        
        # 语音识别
        full_text = self.asr.recognize_file(audio_file)
        
        if not full_text:
            return {"error": "语音识别失败"}
        
        # 文本分析和结构化
        structured_content = self._analyze_meeting_content(full_text)
        
        # 生成正式的会议记录
        meeting_record = self._generate_meeting_record(meeting_info, structured_content)
        
        return meeting_record
    
    def _analyze_meeting_content(self, text):
        """分析会议内容"""
        # 提取关键信息
        topics = self._extract_topics(text)
        decisions = self._extract_decisions(text)
        action_items = self._extract_action_items(text)
        
        return {
            "raw_text": text,
            "topics": topics,
            "decisions": decisions,
            "action_items": action_items,
            "word_count": len(text.split()),
            "estimated_duration": len(text.split()) // 150  # 假设每分钟150词
        }
    
    def _extract_topics(self, text):
        """提取讨论主题"""
        topic_keywords = ["讨论", "话题", "议题", "关于", "问题"]
        sentences = text.split('。')
        
        topics = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in topic_keywords):
                topics.append(sentence.strip())
        
        return topics[:5]  # 返回前5个主题
    
    def _extract_decisions(self, text):
        """提取决议和结论"""
        decision_keywords = ["决定", "确定", "同意", "通过", "批准", "结论"]
        sentences = text.split('。')
        
        decisions = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in decision_keywords):
                decisions.append(sentence.strip())
        
        return decisions
    
    def _extract_action_items(self, text):
        """提取行动项"""
        action_keywords = ["负责", "跟进", "执行", "完成", "安排", "准备"]
        sentences = text.split('。')
        
        action_items = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in action_keywords):
                action_items.append(sentence.strip())
        
        return action_items
    
    def _generate_meeting_record(self, meeting_info, content):
        """生成正式会议记录"""
        record = f"""# {meeting_info['title']}

**会议时间**: {meeting_info['date']} {meeting_info['time']}
**会议时长**: 约{content['estimated_duration']}分钟
**记录方式**: AI语音转录
**字数统计**: {content['word_count']}字

## 会议纪要

{content['raw_text']}

## 主要议题

"""
        
        for i, topic in enumerate(content['topics'], 1):
            record += f"{i}. {topic}\\n"
        
        record += "\\n## 决议事项\\n\\n"
        for i, decision in enumerate(content['decisions'], 1):
            record += f"{i}. {decision}\\n"
        
        record += "\\n## 行动项\\n\\n"
        for i, action in enumerate(content['action_items'], 1):
            record += f"- [ ] {action}\\n"
        
        record += f"""

---
*本记录由AI自动生成，生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return {
            "meeting_record": record,
            "structured_data": content,
            "meeting_info": meeting_info
        }

def save_meeting_record(record_data, output_dir="./meeting_records"):
    """保存会议记录"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存Markdown格式
    md_file = os.path.join(output_dir, f"meeting_{timestamp}.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(record_data["meeting_record"])
    
    # 保存JSON格式（结构化数据）
    json_file = os.path.join(output_dir, f"meeting_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(record_data["structured_data"], f, ensure_ascii=False, indent=2)
    
    return md_file, json_file

# 使用示例
transcriber = MeetingTranscriber()

meeting_info = {
    "title": "产品规划会议",
    "date": "2024-01-31", 
    "time": "14:00-15:30",
    "attendees": ["张三", "李四", "王五"]
}

result = transcriber.transcribe_meeting("meeting_audio.mp3", meeting_info)

if "error" not in result:
    md_file, json_file = save_meeting_record(result)
    print(f"会议记录已保存:")
    print(f"  Markdown: {md_file}")
    print(f"  JSON: {json_file}")
    
    print("\\n会议摘要:")
    print(f"  主要议题: {len(result['structured_data']['topics'])}个")
    print(f"  决议事项: {len(result['structured_data']['decisions'])}个") 
    print(f"  行动项: {len(result['structured_data']['action_items'])}个")
else:
    print(f"会议转录失败: {result['error']}")
```

## 错误处理和调试

### 常见错误处理

```python
from tools.media import ASRService
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

def robust_audio_recognition(file_path, max_retries=3):
    """健壮的音频识别"""
    asr = ASRService()
    
    # 文件检查
    if not os.path.exists(file_path):
        return {"error": "文件不存在", "file": file_path}
    
    if not asr.is_supported_format(file_path):
        return {"error": "不支持的文件格式", "file": file_path}
    
    # 重试识别
    for attempt in range(max_retries):
        try:
            logging.info(f"尝试识别 (第{attempt + 1}次): {file_path}")
            
            text = asr.recognize_file(file_path)
            
            if text and len(text.strip()) > 0:
                return {
                    "success": True,
                    "text": text,
                    "file": file_path,
                    "attempts": attempt + 1,
                    "char_count": len(text)
                }
            else:
                logging.warning(f"识别结果为空 (第{attempt + 1}次)")
                
        except Exception as e:
            logging.error(f"识别异常 (第{attempt + 1}次): {e}")
            
            if attempt == max_retries - 1:  # 最后一次尝试
                return {
                    "error": f"识别失败，已重试{max_retries}次",
                    "last_error": str(e),
                    "file": file_path
                }
            
            time.sleep(2 ** attempt)  # 指数退避
    
    return {"error": "识别失败", "file": file_path}

# 使用示例
result = robust_audio_recognition("test_audio.wav")

if result.get("success"):
    print(f"✓ 识别成功: {result['char_count']}个字符，用时{result['attempts']}次尝试")
    print(f"内容: {result['text'][:100]}...")
else:
    print(f"✗ 识别失败: {result['error']}")
```

## 性能优化和监控

### 1. 性能监控

```python
import time
import psutil
from tools.media import ASRService

class ASRPerformanceMonitor:
    """ASR性能监控器"""
    
    def __init__(self):
        self.asr = ASRService()
        self.stats = []
    
    def monitor_recognition(self, file_path):
        """监控单次识别性能"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            text = self.asr.recognize_file(file_path)
            success = text is not None and len(text.strip()) > 0
            
        except Exception as e:
            text = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        stats = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "processing_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "success": success,
            "text_length": len(text) if text else 0,
            "timestamp": datetime.now()
        }
        
        if not success:
            stats["error"] = error
        
        self.stats.append(stats)
        return stats
    
    def get_performance_summary(self):
        """获取性能摘要"""
        if not self.stats:
            return {"error": "无性能数据"}
        
        successful_stats = [s for s in self.stats if s["success"]]
        
        if not successful_stats:
            return {"error": "无成功识别记录"}
        
        return {
            "total_files": len(self.stats),
            "successful_files": len(successful_stats),
            "success_rate": len(successful_stats) / len(self.stats) * 100,
            "avg_processing_time": sum(s["processing_time"] for s in successful_stats) / len(successful_stats),
            "max_processing_time": max(s["processing_time"] for s in successful_stats),
            "min_processing_time": min(s["processing_time"] for s in successful_stats),
            "avg_memory_usage": sum(s["memory_used"] for s in successful_stats) / len(successful_stats),
            "total_text_generated": sum(s["text_length"] for s in successful_stats)
        }
    
    def generate_report(self, output_file="asr_performance_report.txt"):
        """生成性能报告"""
        summary = self.get_performance_summary()
        
        if "error" in summary:
            return summary
        
        report = f"""ASR性能分析报告
===================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
  - 处理文件数: {summary['total_files']}
  - 成功文件数: {summary['successful_files']}
  - 成功率: {summary['success_rate']:.1f}%

处理时间:
  - 平均处理时间: {summary['avg_processing_time']:.2f}秒
  - 最长处理时间: {summary['max_processing_time']:.2f}秒
  - 最短处理时间: {summary['min_processing_time']:.2f}秒

内存使用:
  - 平均内存使用: {summary['avg_memory_usage'] / 1024 / 1024:.1f}MB

输出统计:
  - 总生成文字: {summary['total_text_generated']}字符

详细记录:
"""
        
        for i, stat in enumerate(self.stats, 1):
            status = "✓" if stat["success"] else "✗"
            report += f"  {i}. {status} {os.path.basename(stat['file_path'])} - {stat['processing_time']:.2f}s\\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        return {"report_file": output_file, "summary": summary}

# 使用示例
monitor = ASRPerformanceMonitor()

# 监控多个文件
test_files = ["audio1.wav", "audio2.mp3", "video1.mp4"]
for file_path in test_files:
    print(f"监控处理: {file_path}")
    stats = monitor.monitor_recognition(file_path)
    print(f"  处理时间: {stats['processing_time']:.2f}秒")
    print(f"  成功: {stats['success']}")

# 生成报告
report_result = monitor.generate_report()
if "error" not in report_result:
    print(f"\\n性能报告已保存: {report_result['report_file']}")
    print("摘要:")
    for key, value in report_result['summary'].items():
        print(f"  {key}: {value}")
```

## 最佳实践

### 1. 音频文件优化建议

```python
def get_audio_optimization_suggestions(file_path):
    """获取音频优化建议"""
    suggestions = []
    
    try:
        import librosa
        y, sr = librosa.load(file_path)
        duration = len(y) / sr
        
        # 采样率建议
        if sr < 16000:
            suggestions.append(f"建议提高采样率到16kHz或以上（当前: {sr}Hz）")
        elif sr > 44100:
            suggestions.append(f"采样率过高可能影响处理速度（当前: {sr}Hz）")
        
        # 时长建议
        if duration < 1:
            suggestions.append("音频过短，可能影响识别准确性")
        elif duration > 600:  # 10分钟
            suggestions.append("音频过长，建议分段处理")
        
        # 音频质量检查
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = float(np.mean(rms))
        
        if avg_rms < 0.01:
            suggestions.append("音频信号较弱，建议增强音量")
        elif avg_rms > 0.5:
            suggestions.append("音频可能存在削波失真")
        
        # 静音检查
        silence_ratio = np.sum(np.abs(y) < 0.01) / len(y)
        if silence_ratio > 0.5:
            suggestions.append(f"静音比例过高 ({silence_ratio*100:.1f}%)，建议去除静音段")
        
        return {
            "file_info": {
                "duration": duration,
                "sample_rate": sr,
                "avg_rms": avg_rms,
                "silence_ratio": silence_ratio
            },
            "suggestions": suggestions
        }
        
    except Exception as e:
        return {"error": f"音频分析失败: {e}"}

# 使用示例
analysis = get_audio_optimization_suggestions("test_audio.wav")
if "error" not in analysis:
    print("音频优化建议:")
    for suggestion in analysis["suggestions"]:
        print(f"  - {suggestion}")
else:
    print(f"分析失败: {analysis['error']}")
```

### 2. 最佳使用建议

```python
# 1. 环境配置检查
def check_asr_environment():
    """检查ASR环境配置"""
    issues = []
    
    # 检查API密钥
    if not os.getenv('DASHSCOPE_API_KEY'):
        issues.append("未设置DASHSCOPE_API_KEY环境变量")
    
    # 检查网络连接
    try:
        import requests
        response = requests.get("https://dashscope.aliyuncs.com", timeout=5)
        if response.status_code != 200:
            issues.append("无法连接到阿里百炼服务")
    except:
        issues.append("网络连接异常")
    
    return issues

# 2. 文件格式转换建议
def suggest_format_conversion(file_path):
    """建议文件格式转换"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 推荐格式优先级
    format_priority = {
        '.wav': 1,    # 最佳
        '.flac': 2,   # 很好
        '.m4a': 3,    # 好
        '.mp3': 4,    # 一般
        '.aac': 5,    # 可接受
        '.ogg': 6,    # 可接受
        '.amr': 7,    # 较差
    }
    
    current_priority = format_priority.get(file_ext, 10)
    
    if current_priority > 4:
        return {
            "should_convert": True,
            "recommended_format": ".wav",
            "reason": f"当前格式{file_ext}识别效果较差，建议转换为WAV格式"
        }
    
    return {"should_convert": False}

# 3. 使用模式建议
class ASRUsageGuide:
    """ASR使用指南"""
    
    @staticmethod
    def get_recommendations(use_case):
        """根据使用场景获取建议"""
        recommendations = {
            "会议录音": {
                "format": "WAV",
                "sample_rate": "16kHz",
                "preprocessing": ["降噪", "音量标准化"],
                "post_processing": ["说话人分离", "关键词提取"]
            },
            "电话录音": {
                "format": "WAV", 
                "sample_rate": "8kHz或16kHz",
                "preprocessing": ["电话信道补偿", "降噪"],
                "post_processing": ["情感分析", "关键信息提取"]
            },
            "讲座录音": {
                "format": "FLAC或WAV",
                "sample_rate": "22kHz",
                "preprocessing": ["回声消除", "背景噪音抑制"],
                "post_processing": ["章节分割", "摘要生成"]
            },
            "访谈录音": {
                "format": "WAV",
                "sample_rate": "16kHz", 
                "preprocessing": ["多通道处理", "音量均衡"],
                "post_processing": ["说话人标记", "情感分析"]
            }
        }
        
        return recommendations.get(use_case, {
            "format": "WAV",
            "sample_rate": "16kHz",
            "preprocessing": ["基础降噪"],
            "post_processing": ["文本校对"]
        })

# 使用示例
print("环境检查:")
issues = check_asr_environment()
for issue in issues:
    print(f"  ⚠️  {issue}")

print("\\n使用建议:")
guide = ASRUsageGuide()
recommendations = guide.get_recommendations("会议录音")
print(f"  推荐格式: {recommendations['format']}")
print(f"  采样率: {recommendations['sample_rate']}")
print("  预处理: " + ", ".join(recommendations['preprocessing']))
print("  后处理: " + ", ".join(recommendations['post_processing']))
```

通过这套音视频处理工具，您可以高效地处理各种音频和视频文件的语音识别需求，支持多种格式和语言，适用于会议记录、访谈转录、多媒体内容分析等多种场景。