"""Precise text alignment service for identifying matching text spans."""
from difflib import SequenceMatcher
from typing import List, Tuple


class TextAlignmentService:
    """精确定位匹配文本的具体位置"""

    def find_matching_spans(
        self,
        left_text: str,
        right_text: str,
        min_match_length: int = 20,
        similarity_threshold: float = 0.8
    ) -> List[Tuple[int, int, int, int]]:
        """
        返回匹配的文本片段位置
        Returns: [(left_start, left_end, right_start, right_end), ...]
        """
        matcher = SequenceMatcher(None, left_text, right_text)
        matches = []

        for match in matcher.get_matching_blocks():
            if match.size >= min_match_length:
                # 扩展匹配边界到完整的词
                left_start = self._expand_to_word_boundary(left_text, match.a, -1)
                left_end = self._expand_to_word_boundary(left_text, match.a + match.size, 1)
                right_start = self._expand_to_word_boundary(right_text, match.b, -1)
                right_end = self._expand_to_word_boundary(right_text, match.b + match.size, 1)

                matches.append((left_start, left_end, right_start, right_end))

        return self._merge_overlapping_spans(matches)

    def _expand_to_word_boundary(self, text: str, pos: int, direction: int) -> int:
        """扩展位置到词边界"""
        if direction < 0:  # 向左扩展
            while pos > 0 and not text[pos-1].isspace():
                pos -= 1
        else:  # 向右扩展
            while pos < len(text) and not text[pos].isspace():
                pos += 1
        return pos

    def _merge_overlapping_spans(self, spans: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """合并重叠的span"""
        if not spans:
            return []

        sorted_spans = sorted(spans)
        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # 重叠
                merged[-1] = (
                    last[0],
                    max(last[1], current[1]),
                    last[2],
                    max(last[3], current[3])
                )
            else:
                merged.append(current)

        return merged