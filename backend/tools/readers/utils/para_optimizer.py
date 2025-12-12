"""
段落优化工具 - 支持多语言文本处理
"""

from typing import Optional, List, Dict, Any
import re
from langdetect import detect

class ParagraphOptimizer:
    """段落优化器 - 提供文本清理和格式化功能"""
    
    def __init__(self):
        """Initialize the paragraph optimizer."""
        pass
    
    def is_table_row(self, line: str) -> bool:
        """
        检查一行是否为表格行（含|分隔符）
        
        Args:
            line: 要检查的文本行
            
        Returns:
            True if 该行是表格行
        """
        return '|' in line and line.count('|') >= 2
    
    def is_cjk_language(self, text: str) -> bool:
        """
        检测文本是否主要为CJK语言（中文、日文、韩文）
        
        Args:
            text: 待分析的文本
            
        Returns:
            True if 文本为CJK语言
        """
        try:
            # 清理文本，使用前100字符进行检测
            clean_text = text.replace('\n', ' ').strip()
            if len(clean_text) < 3:
                # 文本太短，使用字符检测作为备选
                raise Exception("Text too short")
            
            # 使用前100字符避免"text too long"警告
            sample_text = clean_text[:100]
            detected_lang = detect(sample_text)
            return detected_lang in ['zh-cn', 'zh-tw', 'ja', 'ko']
        except Exception:
            # 备选方案：基于字符的CJK检测
            cjk_chars = 0
            total_chars = 0
            
            for char in text:
                if char.isalpha():
                    total_chars += 1
                    # 检查字符是否在CJK Unicode范围
                    if (0x4e00 <= ord(char) <= 0x9fff or    # 中文
                        0x3400 <= ord(char) <= 0x4dbf or    # 中文扩展A
                        0x3040 <= ord(char) <= 0x309f or    # 平假名
                        0x30a0 <= ord(char) <= 0x30ff or    # 片假名
                        0xac00 <= ord(char) <= 0xd7af):     # 韩文
                        cjk_chars += 1
            
            return total_chars > 0 and cjk_chars / total_chars > 0.3

    def fix_text_line_breaks(self, text: str) -> str:
        """
        修复文本中的断行问题，同时保持句子的完整性。

        此方法使用快速字符串操作来：
        1. 连接属于同一句子的行
        2. 保持有意的段落分隔
        3. 处理中文和英文文本（语言感知）

        Args:
            text: 原始文本内容

        Returns:
            修复断行后的文本
        """
        if not text:
            return text

        # Split into lines and process
        lines = text.split('\n')
        result = []
        current_paragraph: list[str] = []

        for line in lines:
            line = line.strip()

            # Empty line indicates paragraph break
            if not line:
                if current_paragraph:
                    # Join current paragraph and add to result
                    paragraph_text = self.join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        result.append(paragraph_text)
                    current_paragraph = []
                continue

            # Check if this line should start a new paragraph
            if self.should_start_new_paragraph(line, current_paragraph):
                if current_paragraph:
                    paragraph_text = self.join_paragraph_lines(current_paragraph)
                    if paragraph_text:
                        result.append(paragraph_text)
                    current_paragraph = []

            current_paragraph.append(line)

        # Handle remaining paragraph
        if current_paragraph:
            paragraph_text = self.join_paragraph_lines(current_paragraph)
            if paragraph_text:
                result.append(paragraph_text)

        # Join paragraphs with double newlines
        return '\n\n'.join(result)

    def join_paragraph_lines(self, lines: list) -> str:
        """
        使用语言感知的智能行合并。
        
        Args:
            lines: 段落中的行列表

        Returns:
            合并后的段落文本
        """
        if not lines:
            return ""

        if len(lines) == 1:
            return lines[0]

        # 检测段落文本的语言
        paragraph_text = '\n'.join(lines)
        is_cjk = self.is_cjk_language(paragraph_text)
        
        result = []

        for i, line in enumerate(lines):
            if i == 0:
                result.append(line)
                continue

            prev_line = lines[i - 1]

            # 检查是否应该与前一行合并
            if self.should_join_lines(prev_line, line):
                if is_cjk:
                    # CJK语言：直接合并，不添加空格
                    result[-1] += line
                else:
                    # 检查是否为连字符断行
                    if re.search(r'[‐\-—]+\s*$', prev_line.strip()):
                        # 连字符断行：保留连字符并合并
                        # 将末尾的连字符标准化为普通连字符并合并
                        cleaned_prev = re.sub(r'[‐—]+\s*$', '-', result[-1])
                        result[-1] = cleaned_prev + line
                    else:
                        # 普通拉丁语言：合并时添加空格
                        result[-1] += ' ' + line
            else:
                # 保持为独立行
                result.append(line)

        return '\n'.join(result)

    def should_start_new_paragraph(self, line: str, current_paragraph: list) -> bool:
        """
        判断一行是否应该开始新段落。

        Args:
            line: 当前行内容
            current_paragraph: 当前段落的行列表

        Returns:
            True if 应该开始新段落
        """
        if not current_paragraph:
            return False

        # Check for common paragraph starters
        paragraph_starters = [
            r'^\d{1,3}[\.\)]\s+\w',  # 1. word or 1) word (数字后必须有空格和文字)
            r'^[一二三四五六七八九十]+[\.\)、]\s',  # Chinese numerals
            r'^[（\(]\d+[）\)]\s',  # (1)
            r'^[A-Z][a-z]*:\s',  # Title: format
            r'^第[一二三四五六七八九十百千万]+[章节部分]\s',  # Chapter indicators
            r'^[•·]\s',  # Bullet points
            r'^-\s',  # Dash bullets
            r'^\*\s',  # Asterisk bullets
        ]

        for pattern in paragraph_starters:
            if re.match(pattern, line):
                return True

        return False

    def should_join_lines(self, prev_line: str, current_line: str) -> bool:
        """
        判断两行是否应该合并。

        Args:
            prev_line: 前一行内容
            current_line: 当前行内容

        Returns:
            True if 这两行应该合并
        """
        if not prev_line or not current_line:
            return False

        # Don't join if current line starts with special characters
        if re.match(r'^[•·\-\*\d\(\)（）]', current_line):
            return False

        # Don't join if current line looks like a title (all caps, etc.)
        if current_line.isupper() and len(current_line) < 50:
            return False

        # 特殊情况：如果前一行以编号结尾（如"1. "、"2. "等），应该与下一行合并
        # 只匹配简单的整数编号，避免匹配小数或百分比
        if re.search(r'\b\d{1,3}[\.\)]\s*$', prev_line.strip()):
            return True

        # 特殊情况：如果前一行以连字符结尾（单词被断开），应该与下一行合并
        # 支持多种连字符：普通连字符(-)、Unicode连字符(‐)、长破折号(—)等
        if re.search(r'[‐\-—]+\s*$', prev_line.strip()):
            return True

        # Don't join if previous line ends with certain punctuation
        if re.search(r'[。！？：]$', prev_line):
            return False

        # Don't join if previous line ends with English sentence endings
        if re.search(r'[.!?:]$', prev_line) and not re.search(r'\b[A-Z][a-z]*\.$', prev_line):
            return False

        # Join if previous line doesn't end with proper punctuation
        # This is the main case for broken lines
        if not re.search(r'[。！？：.!?:]$', prev_line):
            return True

        # Join if previous line ends with comma or other continuing punctuation
        if re.search(r'[，,、]$', prev_line):
            return True

        return False
    
    def optimize_text_spacing(self, text: str) -> str:
        """
        优化文本间距，统一换行符为双换行符以提高可读性。
        
        Args:
            text: 要优化的文本
            
        Returns:
            优化后的文本
        """
        if not text:
            return text
        
        # 将单个或多个换行符统一替换为双换行符
        optimized_text = re.sub(r'\n+', '\n\n', text)
        return optimized_text
    
    def optimize_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        主要的文本优化入口方法。
        
        Args:
            text: 原始文本
            options: 优化选项字典
                - fix_line_breaks: 是否修复断行 (默认: True)
                - normalize_spacing: 是否优化间距 (默认: True)
                - preserve_tables: 是否保留表格格式 (默认: True)
                
        Returns:
            优化后的文本
        """
        if not text:
            return text
        
        if options is None:
            options = {}
        
        # 默认选项
        fix_line_breaks = options.get('fix_line_breaks', True)
        normalize_spacing = options.get('normalize_spacing', True)
        preserve_tables = options.get('preserve_tables', True)
        
        result = text
        
        # 步骤1: 修复断行问题
        if fix_line_breaks:
            result = self.fix_text_line_breaks(result)
        
        # 步骤2: 优化文本间距
        if normalize_spacing:
            result = self.optimize_text_spacing(result)
            
        # TODO: 未来可以添加更多优化步骤
        # - 表格格式保护
        # - 标题识别和格式化
        # - 列表处理
        
        return result