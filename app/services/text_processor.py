"""
文本处理服务 - 文本分割、清理等基础功能
使用 spaCy 进行中英文句子分割，支持混合语言文档
"""
import re
from typing import List, Tuple
import spacy
from spacy.language import Language
from langdetect import detect, LangDetectException
from app.services.base_service import BaseService, singleton


@singleton
class TextProcessor(BaseService):
    """文本处理服务 - 使用 spaCy 进行智能文本处理，支持中英文混合"""

    def _initialize(self):
        """初始化 spaCy 模型"""
        self.nlp_zh = None
        self.nlp_en = None
        self._init_chinese_model()
        self._init_english_model()

    def _init_chinese_model(self):
        """初始化中文模型"""
        try:
            # 加载中文模型，排除parser和senter以提高性能
            self.nlp_zh = spacy.load("zh_core_web_md", exclude=["parser", "senter"])
            # 添加sentencizer进行基本句子分割
            self.nlp_zh.add_pipe("sentencizer")
            # 配置自定义规则
            self._configure_chinese_sentence_boundaries()
            self.logger.info("成功加载中文 spaCy 模型并配置自定义规则")
        except OSError:
            self.logger.error("中文 spaCy 模型未找到，请运行: python -m spacy download zh_core_web_md")
            raise

    def _init_english_model(self):
        """初始化英文模型"""
        try:
            # 加载英文模型（保持默认配置）
            self.nlp_en = spacy.load("en_core_web_md")
            self.logger.info("成功加载英文 spaCy 模型")
        except OSError:
            self.logger.error("英文 spaCy 模型未找到，请运行: python -m spacy download en_core_web_md")
            raise

    def _configure_chinese_sentence_boundaries(self):
        """配置中文句子边界检测规则"""
        # 确保 nlp_zh 不是 None
        if not self.nlp_zh:
            return

        @Language.component("zh_number_protector")
        def zh_number_protector(doc):
            """保护序号不被单独分割成句子"""
            for i, token in enumerate(doc):
                # 如果是数字后的句号
                if (token.text == "." and i > 0 and
                    doc[i-1].text.isdigit() and
                    i + 1 < len(doc)):
                    # 不要在这里分句
                    doc[i + 1].is_sent_start = False

                # 如果整个段落只是一个序号（如 "2."），不作为独立句子
                if len(doc) <= 3 and any(t.text.isdigit() for t in doc) and any(t.text == "." for t in doc):
                    for j in range(len(doc)):
                        if j > 0:
                            doc[j].is_sent_start = False

            return doc

        @Language.component("merge_short_sentences_zh")
        def merge_short_sentences_zh(doc):
            """合并过短的句子以提高语义丰富度"""
            # 最小句子长度（词元数）
            min_tokens = 4  # 中文句子至少4个词

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
                    # 检查是否是独立的短句（感叹句或疑问句）
                    sent_text = doc[start_idx:end_idx].text
                    is_independent = any(punct in sent_text for punct in ["！", "？"])

                    # 如果不是独立短句，则可以合并
                    if not is_independent:
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

        # 添加自定义组件到管道
        if "zh_number_protector" not in self.nlp_zh.pipe_names:
            self.nlp_zh.add_pipe("zh_number_protector", after="sentencizer")
        if "merge_short_sentences_zh" not in self.nlp_zh.pipe_names:
            self.nlp_zh.add_pipe("merge_short_sentences_zh", after="zh_number_protector")

    def detect_language(self, text: str) -> str:
        """
        使用 langdetect 检测文本语言
        返回 'zh' 或 'en'
        """
        # 清理文本
        clean_text = text.strip()

        if not clean_text:
            return 'zh'

        try:
            # 使用langdetect检测语言
            lang = detect(clean_text)

            if lang in ['zh-cn', 'zh-tw']:
                return 'zh'
            elif lang == 'en':
                return 'en'
            else:
                # 其他语言默认使用中文模型
                return 'zh'

        except LangDetectException:
            # 检测失败默认使用中文模型
            self.logger.warning(f"语言检测失败，默认使用中文模型")
            return 'zh'

    def split_paragraphs(self, text: str, min_length: int = 20, max_chars: int = 600) -> List[str]:
        """分割段落 - 使用单换行符作为段落分隔符，并对超长段落进行分割"""
        self._ensure_initialized()
        # 按单个换行符分割
        lines = text.split('\n')
        # 过滤空行和太短的段落
        paragraphs = []
        for line in lines:
            line = line.strip()
            if line and len(line) > min_length:
                # 对超长段落进行分割
                sub_paragraphs = self._split_long_paragraph_internal(line, max_chars)
                paragraphs.extend(sub_paragraphs)
        return paragraphs

    def _split_long_paragraph_internal(self, paragraph: str, max_chars: int = 600) -> List[str]:
        """分割超长段落，确保不超过最大字符数限制"""
        # 使用 langdetect 检测语言类型
        lang = self.detect_language(paragraph)
        is_chinese = (lang == 'zh')

        # 如果段落长度在限制内，直接返回
        if len(paragraph) <= max_chars:
            return [paragraph]

        result = []
        remaining = paragraph

        while len(remaining) > max_chars:
            # 寻找分割点：优先使用句号，其次分号
            if is_chinese:
                # 中文：优先句号，其次分号
                delimiters = ['。', '；']
            else:
                # 英文：优先句号，其次分号
                delimiters = ['. ', '; ']

            # 在段落中间位置附近寻找最佳分割点
            mid_point = len(remaining) // 2
            best_split = -1
            best_distance = float('inf')

            for delimiter in delimiters:
                # 在整个段落中查找所有分隔符位置
                pos = 0
                while True:
                    idx = remaining.find(delimiter, pos)
                    if idx == -1:
                        break
                    # 计算与中点的距离
                    distance = abs(idx + len(delimiter) - mid_point)
                    if distance < best_distance:
                        best_distance = distance
                        best_split = idx + len(delimiter)
                    pos = idx + 1

                # 如果找到合适的分割点，就使用它
                if best_split > 0:
                    break

            # 如果没有找到合适的分隔符，强制在中点分割
            if best_split <= 0:
                # 尝试在空格处分割（避免断词）
                if not is_chinese:
                    # 英文：在中点附近找空格
                    space_idx = remaining.rfind(' ', 0, mid_point)
                    if space_idx > 0:
                        best_split = space_idx + 1
                    else:
                        best_split = mid_point
                else:
                    # 中文：直接在中点分割
                    best_split = mid_point

            # 执行分割
            part = remaining[:best_split].strip()
            if part:
                result.append(part)
            remaining = remaining[best_split:].strip()

        # 添加剩余部分
        if remaining:
            result.append(remaining)

        return result

    def split_paragraphs_with_spans(self, text: str, min_length: int = 20, max_chars: int = 600) -> List[Tuple[str, int, int]]:
        """
        分割段落并返回(text, start, end)偏移
        使用单换行符作为段落分隔符，并对超长段落进行分割
        """
        self._ensure_initialized()
        if not text:
            return []

        spans: List[Tuple[str, int, int]] = []
        lines = text.split('\n')
        current_pos = 0

        for line in lines:
            line_start = current_pos
            line_stripped = line.strip()

            # 只处理非空且足够长的段落
            if line_stripped and len(line_stripped) > min_length:
                # 找到 stripped line 在原始 line 中的位置
                strip_start = line.find(line_stripped)
                if strip_start >= 0:
                    # 对超长段落进行分割
                    sub_paragraphs = self._split_long_paragraph_internal(line_stripped, max_chars)

                    # 计算每个子段落的位置
                    sub_start = 0
                    for sub_para in sub_paragraphs:
                        # 在原始段落中找到子段落的位置
                        sub_idx = line_stripped.find(sub_para, sub_start)
                        if sub_idx >= 0:
                            actual_start = line_start + strip_start + sub_idx
                            actual_end = actual_start + len(sub_para)
                            spans.append((sub_para, actual_start, actual_end))
                            sub_start = sub_idx + len(sub_para)

            # 更新位置（包括换行符）
            current_pos += len(line) + 1  # +1 for newline

        return spans

    def split_sentences(self, text: str, min_length: int = 20) -> List[str]:
        """
        使用双语言模型分割句子
        先按行分割，然后根据每行的语言选择对应的模型
        """
        if not text:
            return []

        # 按单个换行符分割成行
        lines = text.split('\n')
        all_sentences = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测这一行的语言
            lang = self.detect_language(line)

            # 中文文本使用更短的最小长度阈值
            actual_min_length = 10 if lang == "zh" else min_length

            # 选择合适的模型
            if lang == "zh" and self.nlp_zh:
                nlp = self.nlp_zh
            elif lang == "en" and self.nlp_en:
                nlp = self.nlp_en
            else:
                # 回退到正则表达式方法
                sentences = self._split_sentences_regex(line, actual_min_length)
                all_sentences.extend(sentences)
                continue

            try:
                # 使用 spaCy 处理文本
                doc = nlp(line)

                # 提取句子
                for sent in doc.sents:
                    sentence_text = sent.text.strip()
                    # 过滤太短的句子
                    if sentence_text and len(sentence_text) >= actual_min_length:
                        all_sentences.append(sentence_text)

            except Exception as e:
                self.logger.error(f"spaCy 句子分割失败: {e}")
                # 回退到正则表达式方法
                sentences = self._split_sentences_regex(line, actual_min_length)
                all_sentences.extend(sentences)

        return all_sentences

    def split_sentences_with_spans(self, text: str, min_length: int = 20) -> List[Tuple[str, int, int]]:
        """
        使用双语言模型分割句子并返回(text, start, end)偏移
        """
        if not text:
            return []

        # 按单个换行符分割成行
        lines = text.split('\n')
        all_spans: List[Tuple[str, int, int]] = []
        text_cursor = 0  # 在原始文本中的位置

        for line in lines:
            line_start_in_text = text_cursor
            line_stripped = line.strip()

            if line_stripped:
                # 检测这一行的语言
                lang = self.detect_language(line_stripped)

                # 中文文本使用更短的最小长度阈值
                actual_min_length = 10 if lang == "zh" else min_length

                # 选择合适的模型
                if lang == "zh" and self.nlp_zh:
                    nlp = self.nlp_zh
                elif lang == "en" and self.nlp_en:
                    nlp = self.nlp_en
                else:
                    # 回退到正则表达式方法
                    spans = self._split_sentences_with_spans_regex(line_stripped, actual_min_length)
                    # 调整偏移量
                    for sent_text, start, end in spans:
                        actual_start = line_start_in_text + line.find(line_stripped) + start
                        actual_end = line_start_in_text + line.find(line_stripped) + end
                        all_spans.append((sent_text, actual_start, actual_end))
                    text_cursor += len(line) + 1  # +1 for newline
                    continue

                try:
                    # 使用 spaCy 处理文本
                    doc = nlp(line_stripped)

                    # 计算 line_stripped 在原始 line 中的偏移
                    stripped_offset = line.find(line_stripped)

                    # 提取句子和位置
                    for sent in doc.sents:
                        sentence_text = sent.text.strip()
                        # 过滤太短的句子
                        if sentence_text and len(sentence_text) >= actual_min_length:
                            # 计算在整个文本中的偏移
                            start_in_line = sent.start_char
                            end_in_line = sent.end_char

                            # 去除句子两端的空白字符
                            while start_in_line < end_in_line and line_stripped[start_in_line].isspace():
                                start_in_line += 1
                            while end_in_line > start_in_line and line_stripped[end_in_line - 1].isspace():
                                end_in_line -= 1

                            # 转换为在整个文本中的偏移
                            actual_start = line_start_in_text + stripped_offset + start_in_line
                            actual_end = line_start_in_text + stripped_offset + end_in_line

                            if actual_end > actual_start:
                                actual_text = text[actual_start:actual_end]
                                all_spans.append((actual_text, actual_start, actual_end))

                except Exception as e:
                    self.logger.error(f"spaCy 句子分割（带偏移）失败: {e}")
                    # 回退到正则表达式方法
                    spans = self._split_sentences_with_spans_regex(line_stripped, actual_min_length)
                    # 调整偏移量
                    for sent_text, start, end in spans:
                        actual_start = line_start_in_text + line.find(line_stripped) + start
                        actual_end = line_start_in_text + line.find(line_stripped) + end
                        all_spans.append((sent_text, actual_start, actual_end))

            # 更新游标位置（包括换行符）
            text_cursor += len(line) + 1  # +1 for newline

        return all_spans

    def _split_sentences_regex(self, text: str, min_length: int = 20) -> List[str]:
        """正则表达式分割句子 - 作为回退方案"""
        # 改进的正则表达式：处理中英文标点
        sentences = []
        # 分割位置：句号、问号、感叹号、分号后
        split_pattern = r'(?<=[.!?。！？；])'
        parts = re.split(split_pattern, text)

        current_sentence = ""
        for part in parts:
            current_sentence += part
            # 如果当前累积的文本以句子结束符结尾
            if re.search(r'[.!?。！？；]\s*$', current_sentence):
                sentence = current_sentence.strip()
                if sentence and len(sentence) >= min_length:
                    sentences.append(sentence)
                current_sentence = ""

        # 处理最后剩余的部分
        if current_sentence.strip() and len(current_sentence.strip()) >= min_length:
            sentences.append(current_sentence.strip())

        return sentences

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
    def clean_text(text: str) -> str:
        """清理文本 - 保留各语言字符，避免破坏中文等非拉丁文本"""
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 合并多余的空格与制表符，但保留换行
        text = re.sub(r"[ \t]+", " ", text)
        # 去除首尾空白
        return text.strip()
