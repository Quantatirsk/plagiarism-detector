"""
文本处理服务 - 文本分割、清理等基础功能
使用 spaCy 进行中英文句子分割
"""
import re
from typing import List, Tuple, Optional
import spacy
from spacy.lang.zh import Chinese
from spacy.lang.en import English
from spacy.language import Language
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """文本处理服务 - 使用 spaCy 进行智能文本处理"""

    def __init__(self):
        """初始化 spaCy 模型"""
        self.nlp_zh = None
        self.nlp_en = None

        # 尝试加载中文模型
        try:
            self.nlp_zh = spacy.load("zh_core_web_sm", exclude=["parser", "senter"])
            # 添加sentencizer以进行基本句子分割
            self.nlp_zh.add_pipe("sentencizer")
            # 添加自定义句子边界规则
            self._configure_chinese_sentence_boundaries(self.nlp_zh)
            logger.info("成功加载中文 spaCy 模型并配置自定义规则")
        except OSError:
            logger.warning("中文 spaCy 模型未找到，请运行: python -m spacy download zh_core_web_sm")
            # 使用基础中文语言类
            self.nlp_zh = Chinese()
            self.nlp_zh.add_pipe("sentencizer")
            self._configure_chinese_sentence_boundaries(self.nlp_zh)

        # 尝试加载英文模型
        try:
            self.nlp_en = spacy.load("en_core_web_sm", exclude=["parser", "senter"])
            # 添加sentencizer以进行基本句子分割
            self.nlp_en.add_pipe("sentencizer")
            # 添加自定义句子边界规则
            self._configure_english_sentence_boundaries(self.nlp_en)
            logger.info("成功加载英文 spaCy 模型并配置自定义规则")
        except OSError:
            logger.warning("英文 spaCy 模型未找到，请运行: python -m spacy download en_core_web_sm")
            # 使用基础英文语言类
            self.nlp_en = English()
            self.nlp_en.add_pipe("sentencizer")
            self._configure_english_sentence_boundaries(self.nlp_en)

    def _configure_chinese_sentence_boundaries(self, nlp):
        """配置中文句子边界检测规则"""
        # 确保有句子分割器
        if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        @Language.component("chinese_sentence_rules")
        def chinese_sentence_rules(doc):
            """自定义中文句子分割规则 - 在sentencizer之后运行"""
            # 首先确保有基本的句子边界
            if not any(token.is_sent_start for token in doc):
                # 如果没有句子边界，使用基本规则设置
                for i, token in enumerate(doc):
                    if i == 0:
                        token.is_sent_start = True
                    elif token.text in ["。", "！", "？", "；"]:
                        if i + 1 < len(doc):
                            doc[i + 1].is_sent_start = True

            # 应用自定义规则
            for i, token in enumerate(doc):
                # 处理换行符 - 在换行符后开始新句子
                if token.is_space and "\n" in token.text:
                    # 如果下一个token存在且不是空白，则标记为句子开始
                    if i + 1 < len(doc) and not doc[i + 1].is_space:
                        doc[i + 1].is_sent_start = True

                    # 特殊处理：如果是双换行（段落分隔），确保分割
                    if "\n\n" in token.text:
                        # 标记前一个非空白token为句子结束位置
                        j = i - 1
                        while j >= 0 and doc[j].is_space:
                            j -= 1
                        # 如果前面有标点符号，确保其后是句子边界
                        if j >= 0 and doc[j].text in ["。", "！", "？", "；", "）", "】", "】"]:
                            if j + 1 < len(doc):
                                k = j + 1
                                while k < len(doc) and doc[k].is_space:
                                    k += 1
                                if k < len(doc):
                                    doc[k].is_sent_start = True

                # 不要在称谓后分割
                elif token.text in ["先生", "女士", "老师", "医生", "教授", "博士", "同学", "同志"]:
                    if i + 1 < len(doc) and doc[i + 1].is_sent_start:
                        doc[i + 1].is_sent_start = False

                # 不要在省略号后立即分割
                elif token.text in ["...", "……", "…"]:
                    if i + 1 < len(doc) and doc[i + 1].is_sent_start:
                        doc[i + 1].is_sent_start = False

                # 在某些连接词前强制分割
                elif token.text in ["总之", "因此", "所以", "然而", "但是", "可是", "另外", "此外", "首先", "其次", "最后"]:
                    if i > 0 and not token.is_sent_start:
                        token.is_sent_start = True

            return doc

        @Language.component("merge_short_sentences_zh")
        def merge_short_sentences_zh(doc):
            """合并过短的句子以提高语义丰富度"""
            # 最小句子长度（词元数）
            min_tokens = 8  # 中文句子至少8个词

            # 收集当前句子边界
            sent_starts = []
            for i, token in enumerate(doc):
                if token.is_sent_start:
                    sent_starts.append(i)

            # 计算每个句子的长度并决定是否合并
            new_sent_starts = set()
            i = 0
            while i < len(sent_starts):
                start_idx = sent_starts[i]
                end_idx = sent_starts[i + 1] if i + 1 < len(sent_starts) else len(doc)
                sent_length = end_idx - start_idx

                # 当前句子是否应该保留
                should_keep = True

                # 如果句子太短
                if sent_length < min_tokens and i + 1 < len(sent_starts):
                    # 检查是否包含换行符（不跨段落合并）
                    sent_span = doc[start_idx:end_idx]
                    has_newline = any(token.text == "\n" for token in sent_span)

                    # 检查下一句是否以换行开始（标题后的正文不合并）
                    next_start = sent_starts[i + 1]
                    starts_with_newline = False
                    if next_start > 0 and doc[next_start - 1].text == "\n":
                        starts_with_newline = True

                    # 检查是否是独立的短句（感叹句或疑问句）
                    sent_text = sent_span.text
                    is_independent = any(punct in sent_text for punct in ["！", "？"])

                    # 如果没有换行符、不是独立短句、且下一句不是新段落，则可以合并
                    if not has_newline and not starts_with_newline and not is_independent:
                        should_keep = False

                if should_keep or i == 0:  # 第一个句子总是保留
                    new_sent_starts.add(start_idx)

                i += 1

            # 应用新的句子边界
            for i, token in enumerate(doc):
                if i in new_sent_starts:
                    token.is_sent_start = True
                elif i > 0:  # 第一个token总是句子开始
                    token.is_sent_start = False

            return doc

        # 添加自定义组件
        if "chinese_sentence_rules" not in nlp.pipe_names:
            nlp.add_pipe("chinese_sentence_rules", after="sentencizer" if "sentencizer" in nlp.pipe_names else None)
        if "merge_short_sentences_zh" not in nlp.pipe_names:
            nlp.add_pipe("merge_short_sentences_zh", after="chinese_sentence_rules")

    def _configure_english_sentence_boundaries(self, nlp):
        """配置英文句子边界检测规则"""
        # 确保有句子分割器
        if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        @Language.component("english_sentence_rules")
        def english_sentence_rules(doc):
            """自定义英文句子分割规则"""
            for i, token in enumerate(doc):
                # 不要在缩写后分割
                if token.text.lower() in ["dr", "mr", "mrs", "ms", "prof", "sr", "jr"]:
                    if i + 1 < len(doc) and doc[i + 1].text == ".":
                        if i + 2 < len(doc):
                            doc[i + 2].is_sent_start = False

                # 不要在省略号后立即分割
                elif token.text == "...":
                    if i + 1 < len(doc):
                        doc[i + 1].is_sent_start = False

            return doc

        @Language.component("merge_short_english_sentences")
        def merge_short_english_sentences(doc):
            """合并过短的英文句子"""
            # 最小句子长度（词元数）
            min_tokens = 6  # 英文句子至少6个词

            for i, token in enumerate(doc):
                if i == 0:
                    token.is_sent_start = True
                    continue

                if token.is_sent_start or token.is_sent_start is None:
                    sent_length = 0
                    for j in range(i, len(doc)):
                        if j > i and (doc[j].is_sent_start or doc[j].is_sent_start is None):
                            break
                        sent_length += 1

                    # 如果句子太短，合并到下一句
                    if sent_length < min_tokens and i < len(doc) - sent_length:
                        # 检查是否是独立的短句
                        sent_text = doc[i:i+sent_length].text
                        if not any(punct in sent_text for punct in ["!", "?"]):
                            token.is_sent_start = False

            return doc

        # 添加自定义组件
        if "english_sentence_rules" not in nlp.pipe_names:
            nlp.add_pipe("english_sentence_rules", before="parser" if "parser" in nlp.pipe_names else None)
        if "merge_short_english_sentences" not in nlp.pipe_names:
            nlp.add_pipe("merge_short_english_sentences", after="english_sentence_rules")

    def detect_language(self, text: str) -> str:
        """检测文本语言（简单实现）"""
        # 统计中文字符
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        # 统计所有字母字符
        total_alpha = sum(1 for char in text if char.isalpha())

        if total_alpha == 0:
            return "unknown"

        # 如果中文字符占比超过30%，认为是中文
        chinese_ratio = chinese_chars / total_alpha
        return "zh" if chinese_ratio > 0.3 else "en"

    @staticmethod
    def split_paragraphs(text: str, min_length: int = 20) -> List[str]:
        """分割段落 - 简单规则"""
        # 按双换行符分割
        paragraphs = re.split(r'\n\n+', text)
        # 过滤太短的段落 - 降低阈值适配中文文本
        return [p.strip() for p in paragraphs if len(p.strip()) > min_length]

    @staticmethod
    def split_paragraphs_with_spans(text: str, min_length: int = 20) -> List[Tuple[str, int, int]]:
        """分割段落并返回(start,end)偏移（相对全文）"""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r'\n\n+', normalized)
        spans: List[Tuple[str, int, int]] = []
        cursor = 0
        for part in parts:
            if not part:
                continue
            # 找到该部分在全文中的位置（从cursor开始，确保顺序）
            idx = normalized.find(part, cursor)
            if idx == -1:
                # 回退到全文搜索（极端情况下）
                idx = normalized.find(part)
                if idx == -1:
                    continue
            start = idx
            end = idx + len(part)
            # 去除外侧空白，确保高亮贴合实际字符
            trimmed = part.strip()
            if len(trimmed) <= min_length:
                cursor = end
                continue
            # 计算去除空白后的偏移
            leading = len(part) - len(part.lstrip())
            trailing = len(part) - len(part.rstrip())
            start += leading
            end -= trailing
            spans.append((normalized[start:end], start, end))
            cursor = idx + len(part)
        return spans
    
    def split_sentences(self, text: str, min_length: int = 20) -> List[str]:
        """使用 spaCy 分割句子 - 支持中英文"""
        if not text:
            return []

        # 检测语言
        lang = self.detect_language(text)

        # 中文文本使用更短的最小长度阈值
        actual_min_length = 10 if lang == "zh" else min_length

        # 选择合适的模型
        if lang == "zh" and self.nlp_zh:
            nlp = self.nlp_zh
        elif lang == "en" and self.nlp_en:
            nlp = self.nlp_en
        else:
            # 回退到正则表达式方法
            return self._split_sentences_regex(text, actual_min_length)

        try:
            # 使用 spaCy 处理文本
            doc = nlp(text)

            # 提取句子
            sentences = []
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                # 过滤太短的句子
                if sentence_text and len(sentence_text) >= actual_min_length:
                    sentences.append(sentence_text)

            return sentences

        except Exception as e:
            logger.error(f"spaCy 句子分割失败: {e}")
            # 回退到正则表达式方法
            return self._split_sentences_regex(text, actual_min_length)

    def _split_sentences_regex(self, text: str, min_length: int = 20) -> List[str]:
        """正则表达式分割句子 - 作为回退方案"""
        # 将不同换行标准统一
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # 改进的正则表达式：处理中英文标点
        # 使用正向后顾，保留标点符号
        sentences = []
        # 分割位置：句号、问号、感叹号、分号后
        split_pattern = r'(?<=[.!?。！？；])'
        parts = re.split(split_pattern, normalized)

        current_sentence = ""
        for part in parts:
            current_sentence += part
            # 如果当前累积的文本以句子结束符结尾，或包含换行符
            if re.search(r'[.!?。！？；]\s*$', current_sentence) or '\n' in current_sentence:
                # 按换行符进一步分割
                sub_parts = current_sentence.split('\n')
                for sub_part in sub_parts:
                    sub_part = sub_part.strip()
                    if sub_part and len(sub_part) >= min_length:
                        sentences.append(sub_part)
                current_sentence = ""

        # 处理最后剩余的部分
        if current_sentence.strip() and len(current_sentence.strip()) >= min_length:
            sentences.append(current_sentence.strip())

        return sentences

    def split_sentences_with_spans(self, text: str, min_length: int = 20) -> List[Tuple[str, int, int]]:
        """使用 spaCy 分割句子并返回(start,end)偏移（相对全文）"""
        if not text:
            return []

        # 检测语言
        lang = self.detect_language(text)

        # 中文文本使用更短的最小长度阈值
        actual_min_length = 10 if lang == "zh" else min_length

        # 选择合适的模型
        if lang == "zh" and self.nlp_zh:
            nlp = self.nlp_zh
        elif lang == "en" and self.nlp_en:
            nlp = self.nlp_en
        else:
            # 回退到正则表达式方法
            return self._split_sentences_with_spans_regex(text, actual_min_length)

        try:
            # 使用 spaCy 处理文本
            doc = nlp(text)

            # 提取句子和位置
            spans: List[Tuple[str, int, int]] = []
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                # 过滤太短的句子
                if sentence_text and len(sentence_text) >= actual_min_length:
                    # spaCy 提供了字符级别的偏移
                    start = sent.start_char
                    end = sent.end_char
                    # 去除句子两端的空白字符
                    while start < end and text[start].isspace():
                        start += 1
                    while end > start and text[end - 1].isspace():
                        end -= 1

                    if end > start:
                        actual_text = text[start:end]
                        spans.append((actual_text, start, end))

            return spans

        except Exception as e:
            logger.error(f"spaCy 句子分割（带偏移）失败: {e}")
            # 回退到正则表达式方法
            return self._split_sentences_with_spans_regex(text, actual_min_length)

    def _split_sentences_with_spans_regex(self, text: str, min_length: int = 20) -> List[Tuple[str, int, int]]:
        """正则表达式分割句子并返回偏移 - 作为回退方案"""
        # 先使用基础方法获取句子
        sentences = self._split_sentences_regex(text, min_length)

        spans: List[Tuple[str, int, int]] = []
        cursor = 0

        for sentence in sentences:
            # 在原文中查找句子位置
            idx = text.find(sentence, cursor)
            if idx == -1:
                # 容错：回退到全局搜索
                idx = text.find(sentence)
                if idx == -1:
                    continue

            start = idx
            end = idx + len(sentence)
            spans.append((sentence, start, end))
            cursor = end

        return spans
    
    @staticmethod
    def create_sliding_windows(
        text: str,
        window_size: int = 500,
        overlap: int = 100
    ) -> List[Tuple[str, int]]:
        """滑动窗口分块 - 用于长文本"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), window_size - overlap):
            chunk = ' '.join(words[i:i + window_size])
            chunks.append((chunk, i))
        
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本 - 保留各语言字符，避免破坏中文等非拉丁文本"""
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 合并多余的空格与制表符，但保留换行
        text = re.sub(r"[ \t]+", " ", text)
        # 去除首尾空白
        return text.strip()


# 全局实例
_text_processor: Optional[TextProcessor] = None

def get_text_processor() -> TextProcessor:
    """获取文本处理器实例（单例模式）"""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor()
    return _text_processor
