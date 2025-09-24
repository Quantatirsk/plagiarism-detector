# 检测系统优化建议 - 快速改进方案

## 立即可执行的优化（不改代码结构）

### 1. 调整Pipeline配置参数

修改 `PipelineConfig` 默认值：

```python
@dataclass
class PipelineConfig:
    # 词汇相似度参数
    lexical_shingle_size: int = 3        # 从8改为3（特别对中文友好）
    lexical_threshold: float = 0.6       # 从0.4提高到0.6（减少噪声）

    # 语义相似度参数
    semantic_threshold: float = 0.72     # 从0.70略微提高
    top_k: int = 20                      # 从5增加到20（避免漏检）

    # 最终阈值
    final_threshold: float = 0.73        # 从0.75略微降低

    # 候选数量限制
    max_candidates: int = 1000           # 从500增加到1000

    # Cross-encoder参数（建议禁用）
    cross_encoder_top_k: int = 50        # 从200减少到50
    cross_encoder_threshold: float = 0.70 # 从0.55提高到0.70
```

### 2. 禁用或弱化CrossEncoderStage

**选项A - 完全禁用**：
```python
# 在 comparison_service.py 中
cross_encoder_service=None  # 传入None禁用
```

**选项B - 提高权重要求**：
仅当Cross-encoder分数很高时才使用：
```python
# 修改 FusionScoringStage
if state.cross_score is not None and state.cross_score > 0.8:
    # 只有高置信度时才融合
    final = 0.2 * lexical + 0.7 * semantic + 0.1 * cross
else:
    # 否则忽略cross_score
    final = 0.2 * lexical + 0.8 * semantic
```

### 3. 优化文本分割参数

针对不同语言优化分割策略：

```python
# 中文文档
paragraph_min_length = 30  # 从20提高到30
paragraph_max_chars = 500  # 从600降低到500
sentence_min_length_zh = 25  # 从20提高到25

# 英文文档
sentence_min_length_en = 50  # 从40提高到50
```

### 4. 环境变量配置

创建 `.env.production` 文件：

```bash
# 语义检测优化配置
DETECTION_MODE=semantic_first
ENABLE_CROSS_ENCODER=false
SEMANTIC_WEIGHT=0.8
LEXICAL_WEIGHT=0.2

# 性能优化
EMBEDDING_BATCH_SIZE=50
TEXT_PROCESSOR_CACHE_SIZE=2000
PARALLEL_COMPARISON_WORKERS=4
```

## 中期优化方案（少量代码修改）

### 1. 添加检测模式选择

```python
class DetectionMode(Enum):
    PURE_SEMANTIC = "pure_semantic"      # 纯语义
    SEMANTIC_FIRST = "semantic_first"    # 语义优先
    BALANCED = "balanced"                # 平衡模式
    STRICT = "strict"                    # 严格模式

def get_pipeline_config(mode: DetectionMode) -> PipelineConfig:
    """根据检测模式返回配置"""
    configs = {
        DetectionMode.PURE_SEMANTIC: PipelineConfig(
            lexical_threshold=1.0,  # 实际上禁用
            semantic_threshold=0.75,
            final_threshold=0.75,
            top_k=30
        ),
        DetectionMode.SEMANTIC_FIRST: PipelineConfig(
            lexical_shingle_size=3,
            lexical_threshold=0.7,
            semantic_threshold=0.72,
            final_threshold=0.73,
            top_k=20
        ),
        # ... 其他模式
    }
    return configs[mode]
```

### 2. 改进分数融合逻辑

```python
class ImprovedFusionScoringStage(PipelineStage):
    """改进的融合评分"""

    def _compute_confidence(self, state: CandidateState) -> float:
        """计算各分数的可信度"""
        confidence = 1.0

        # 如果词汇和语义分数差异很大，降低可信度
        if state.lexical_overlap and state.semantic_score:
            diff = abs(state.lexical_overlap - state.semantic_score)
            if diff > 0.3:
                confidence *= 0.8

        # Cross-encoder降级时降低可信度
        if state.metadata.get("cross_encoder_fallback"):
            confidence *= 0.7

        return confidence

    async def run(self, context: PipelineContext) -> None:
        for state in context.candidate_states.values():
            confidence = self._compute_confidence(state)

            # 基于可信度动态调整权重
            if confidence > 0.9:
                # 高可信度：使用所有信号
                weights = {"lexical": 0.2, "semantic": 0.6, "cross": 0.2}
            else:
                # 低可信度：依赖语义
                weights = {"lexical": 0.1, "semantic": 0.85, "cross": 0.05}

            # 计算加权分数
            final = 0.0
            final += weights["lexical"] * (state.lexical_overlap or 0)
            final += weights["semantic"] * (state.semantic_score or 0)
            final += weights["cross"] * (state.cross_score or 0)

            state.final_score = final
```

### 3. 添加检测质量监控

```python
class DetectionMetrics:
    """检测质量指标收集"""

    def __init__(self):
        self.metrics = {
            "stage_timings": {},
            "candidate_counts": {},
            "score_distributions": {},
            "threshold_effectiveness": {}
        }

    def record_stage_timing(self, stage_name: str, duration: float):
        self.metrics["stage_timings"][stage_name] = duration

    def analyze_effectiveness(self) -> Dict:
        """分析检测效果"""
        return {
            "avg_semantic_score": np.mean(self.semantic_scores),
            "semantic_only_matches": self.semantic_only_count,
            "cross_encoder_impact": self.cross_encoder_changes,
            "detection_confidence": self.calculate_confidence()
        }
```

## 测试建议

### 1. A/B测试不同配置

```python
async def ab_test_detection():
    """对比不同检测模式的效果"""

    test_cases = load_test_dataset()  # 已知的抄袭案例

    modes = [
        DetectionMode.PURE_SEMANTIC,
        DetectionMode.SEMANTIC_FIRST,
        DetectionMode.BALANCED
    ]

    results = {}
    for mode in modes:
        config = get_pipeline_config(mode)
        pipeline = SimilarityPipeline(config)

        metrics = {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "avg_time": 0
        }

        # 运行测试...
        results[mode] = metrics

    return results
```

### 2. 建立基准测试集

收集不同类型的测试样本：
- 完全复制的文本
- 同义词替换的文本
- 句子顺序调整的文本
- 段落重组的文本
- 引用和改写的文本

### 3. 监控线上效果

```python
# 添加日志记录
logger.info("Detection result", {
    "mode": detection_mode,
    "semantic_matches": semantic_count,
    "lexical_matches": lexical_count,
    "cross_encoder_used": cross_encoder_used,
    "avg_confidence": avg_confidence,
    "processing_time": elapsed_time
})
```

## 预期效果

通过以上优化，预期可以达到：

1. **检测准确率提升 15-25%**
   - 减少假阴性（漏检）
   - 保持假阳性率不变

2. **性能提升 20-30%**
   - 减少不必要的计算
   - 更好的缓存利用

3. **可维护性提升**
   - 更清晰的配置
   - 更容易调试
   - 更好的监控

4. **用户体验改善**
   - 更一致的检测结果
   - 更快的响应时间
   - 更透明的检测过程