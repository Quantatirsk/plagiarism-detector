"""Bidirectional best match filter for paragraph-level matches."""
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class MatchScore:
    """Match score between two chunks."""
    left_id: int
    right_id: int
    score: float
    details: Dict[str, object]


class BidirectionalMatchFilter:
    """
    实现双向最佳匹配过滤器。
    只保留互为最佳匹配的段落对，类似稳定婚姻算法的思想。
    """

    def __init__(self, matches: List[Dict[str, object]]):
        self.matches = matches
        self._left_to_right_best: Dict[int, Tuple[int, float]] = {}
        self._right_to_left_best: Dict[int, Tuple[int, float]] = {}
        self._build_best_matches()

    def _build_best_matches(self) -> None:
        """构建双向最佳匹配映射。"""
        # 找出每个左侧段落的最佳右侧匹配
        for match in self.matches:
            left_id = match["left_chunk_id"]
            right_id = match["right_chunk_id"]
            score = match.get("final_score", 0.0)

            # 更新左侧的最佳匹配
            if left_id not in self._left_to_right_best or score > self._left_to_right_best[left_id][1]:
                self._left_to_right_best[left_id] = (right_id, score)

            # 更新右侧的最佳匹配
            if right_id not in self._right_to_left_best or score > self._right_to_left_best[right_id][1]:
                self._right_to_left_best[right_id] = (left_id, score)

    def get_stable_matches(self) -> List[Dict[str, object]]:
        """
        返回稳定匹配（互为最佳选择的匹配对）。
        """
        stable_pairs: Set[Tuple[int, int]] = set()

        # 找出互为最佳匹配的对
        for left_id, (right_id, _) in self._left_to_right_best.items():
            if (right_id in self._right_to_left_best and
                    self._right_to_left_best[right_id][0] == left_id):
                stable_pairs.add((left_id, right_id))

        # 过滤原始匹配列表，只保留稳定匹配
        return [
            match for match in self.matches
            if (match["left_chunk_id"], match["right_chunk_id"]) in stable_pairs
        ]

    def get_relaxed_matches(self, threshold_ratio: float = 0.95) -> List[Dict[str, object]]:
        """
        返回宽松的匹配（允许次优匹配，如果分数接近最佳匹配）。

        Args:
            threshold_ratio: 相对于最佳匹配的分数比例阈值（默认0.95）
        """
        accepted_matches: Set[Tuple[int, int]] = set()

        # 首先添加所有稳定匹配
        for left_id, (right_id, _) in self._left_to_right_best.items():
            if (right_id in self._right_to_left_best and
                    self._right_to_left_best[right_id][0] == left_id):
                accepted_matches.add((left_id, right_id))

        # 添加接近最佳的次优匹配
        for match in self.matches:
            left_id = match["left_chunk_id"]
            right_id = match["right_chunk_id"]
            score = match.get("final_score", 0.0)

            # 检查是否接近左侧的最佳匹配
            if left_id in self._left_to_right_best:
                _, best_score = self._left_to_right_best[left_id]
                if score >= best_score * threshold_ratio:
                    # 同时检查是否也接近右侧的最佳匹配
                    if right_id in self._right_to_left_best:
                        _, right_best_score = self._right_to_left_best[right_id]
                        if score >= right_best_score * threshold_ratio:
                            accepted_matches.add((left_id, right_id))

        return [
            match for match in self.matches
            if (match["left_chunk_id"], match["right_chunk_id"]) in accepted_matches
        ]

    def get_statistics(self) -> Dict[str, int]:
        """返回匹配统计信息。"""
        stable_matches = self.get_stable_matches()
        return {
            "total_matches": len(self.matches),
            "left_chunks": len(self._left_to_right_best),
            "right_chunks": len(self._right_to_left_best),
            "stable_matches": len(stable_matches),
            "filtered_out": len(self.matches) - len(stable_matches),
        }