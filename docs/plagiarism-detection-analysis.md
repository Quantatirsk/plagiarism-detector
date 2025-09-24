# 抄袭检测系统完整逻辑分析报告

## 目录
1. [系统概览](#系统概览)
2. [检测流程详解](#检测流程详解)
3. [核心算法分析](#核心算法分析)
4. [潜在负面影响分析](#潜在负面影响分析)
5. [优化建议](#优化建议)

## 系统概览

当前抄袭检测系统采用了**多阶段混合检测策略**，结合了词汇相似度、语义相似度、交叉编码器重排序等多种技术。

### 技术架构
- **向量嵌入**: OpenAI兼容API (Qwen3-Embedding-8B, 4096维)
- **向量检索**: Milvus向量数据库
- **重排序**: Jina AI Cross-Encoder (降级为TF-IDF)
- **文本处理**: spaCy (中英文分词)

### 检测粒度
- **段落级**: 快速筛选（默认）
- **句子级**: 细粒度检测（用于向量嵌入）

## 检测流程详解

### 1. 文档预处理阶段

```
文档输入 → 文本解析 → 段落分割 → 句子分割 → 向量嵌入 → 存储
```

**段落分割**:
- 最小长度: 20字符
- 最大长度: 600字符
- 超长段落自动分割

**句子分割**:
- 中文最小长度: 20字符
- 英文最小长度: 40字符
- 使用spaCy进行智能分割

### 2. 相似度检测阶段 (SimilarityPipeline)

检测流程包含5个连续阶段：

#### Stage 1: FingerprintRecallStage (词汇召回)
```python
# 使用8-gram shingle + Jaccard相似度
shingle_size = 8
lexical_threshold = 0.4  # 40%相似度阈值
```

**问题**: Shingle size=8对中文文本过大，可能错过相似片段

#### Stage 2: SemanticRecallStage (语义召回)
```python
# 向量余弦相似度计算
semantic_threshold = 0.70  # 70%相似度阈值
top_k = 5  # 每个chunk只保留前5个最相似的
```

**核心逻辑**: 这是最有效的检测阶段

#### Stage 3: CrossEncoderStage (交叉编码器重排序)
```python
cross_encoder_top_k = 200  # 只对前200个候选进行重排序
cross_encoder_threshold = 0.55  # 55%阈值
```

**问题**:
- 依赖Jina API，无API时降级为TF-IDF
- TF-IDF可能与语义向量产生冲突

#### Stage 4: FusionScoringStage (融合评分)
```python
# 有cross_score时:
final_score = 0.1 * lexical + 0.6 * semantic + 0.3 * cross
# 无cross_score时:
final_score = 0.25 * lexical + 0.75 * semantic

# 最终取max(final_score, semantic_score)
```

**问题**: 权重分配可能削弱语义相似度的作用

#### Stage 5: AlignmentStage (对齐阶段)
```python
final_threshold = 0.75  # 只处理75%以上的匹配
# 使用SequenceMatcher进行文本对齐
```

### 3. 结果聚合阶段 (MatchAggregator)

将句子级匹配聚合到段落级：
- 计算覆盖率 (alignment_ratio)
- 合并相邻匹配
- 生成匹配报告

## 核心算法分析

### 词汇相似度 (Fingerprint)

```python
def _jaccard(self, a: set[str], b: set[str]) -> float:
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0
```

**优点**: 快速，对完全相同的文本片段敏感
**缺点**:
- 对同义词替换无效
- 8-gram对中文过长（8个汉字≈一个短句）
- 可能产生大量误报

### 语义相似度 (Semantic)

```python
# 向量归一化后计算余弦相似度
E1n = E1 / np.linalg.norm(E1, axis=1, keepdims=True)
E2n = E2 / np.linalg.norm(E2, axis=1, keepdims=True)
similarity = E1n @ E2n.T
```

**优点**:
- 捕获语义相似性
- 对同义词、改写敏感
- 使用高质量embedding模型

**缺点**:
- Top-K=5可能错过一些匹配
- 依赖embedding质量

### 分数融合算法

当前的融合算法存在多个问题：

1. **权重不合理**: Cross-encoder权重(0.3)过高，而其质量未必好于semantic
2. **Max操作**: `max(final_score, semantic_score)`虽然保护了语义分数，但使融合失去意义
3. **阈值链**: 多个阈值(0.4, 0.7, 0.55, 0.75)增加了复杂性

## 潜在负面影响分析

### 1. 过度复杂的Pipeline

**问题**：
- 5个阶段串行执行，每个阶段都可能引入噪声
- 多个阈值需要精细调优
- 候选剪枝(max_candidates=500)可能丢失真实匹配

**影响**：
- 增加了假阴性（漏检）的可能
- 系统行为难以预测和调试

### 2. Cross-Encoder降级问题

```python
def _tfidf_score(self, left: str, right: str) -> float:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    # ...
```

**问题**：
- TF-IDF与语义向量的分数分布不同
- 1-2 gram对中文效果有限
- 每次调用都重新fit vectorizer，效率低

**影响**：
- 分数不一致，影响融合效果
- 可能产生与语义相似度相反的结果

### 3. Fingerprint阶段的负面影响

**8-gram shingle对中文的问题**：
```
例句: "机器学习是人工智能的重要分支"
8-gram: "机器学习是人工智能的" （几乎是整句）
```

**影响**：
- 中文文本召回率低
- 产生大量无意义的候选
- 浪费计算资源

### 4. 分数融合的问题

当前融合公式：
```python
final = 0.1 * lexical + 0.6 * semantic + 0.3 * cross
final = max(final, semantic)  # 保护语义分数
```

**问题**：
- 如果cross_score质量差，会拉低整体分数
- Max操作使得融合公式前功尽弃
- 权重固定，无法适应不同类型的文本

### 5. Top-K限制

```python
top_k = 5  # SemanticRecallStage
cross_encoder_top_k = 200  # CrossEncoderStage
```

**问题**：
- 语义阶段只保留5个候选太少
- 可能错过排名6-10的真实匹配
- 200个cross-encoder候选可能包含大量噪声

## 优化建议

### 1. 简化为纯语义模式

```python
class SimplifiedSemanticPipeline:
    """纯语义检测模式"""

    async def detect(self, left_chunks, right_chunks):
        # 1. 计算语义相似度
        similarities = compute_cosine_similarity(left_embeddings, right_embeddings)

        # 2. 应用单一阈值
        threshold = 0.75  # 可调整
        matches = similarities >= threshold

        # 3. 直接返回结果
        return matches
```

**优势**：
- 逻辑清晰，易于调试
- 减少假阴性
- 性能更好

### 2. 优化Fingerprint阶段

针对中文优化：
```python
# 中文使用3-gram
shingle_size_zh = 3  # "机器学", "器学习", "学习是"

# 提高词汇阈值，减少噪声
lexical_threshold = 0.6  # 从0.4提高到0.6
```

### 3. 移除或优化Cross-Encoder

**选项1**: 完全移除CrossEncoderStage
```python
self.stages = [
    # FingerprintRecallStage(),  # 可选
    SemanticRecallStage(),
    FusionScoringStage(),
    AlignmentStage(),
]
```

**选项2**: 使用本地模型
- 部署本地cross-encoder模型
- 确保分数分布一致性

### 4. 改进分数融合

```python
def improved_fusion(lexical, semantic, cross=None):
    if cross is None:
        # 纯语义模式
        return semantic

    # 动态权重，基于分数可信度
    if cross > 0.8:  # 高置信度
        return 0.7 * semantic + 0.3 * cross
    else:  # 低置信度，降低cross权重
        return 0.9 * semantic + 0.1 * cross
```

### 5. 增加后处理过滤

```python
def post_process_filter(matches):
    """过滤明显的假阳性"""
    filtered = []
    for match in matches:
        # 过滤过短的匹配
        if match.char_count < 50:
            continue

        # 过滤纯数字/标点的匹配
        if is_meaningless(match.text):
            continue

        filtered.append(match)
    return filtered
```

## 结论

当前系统的复杂性可能确实对检测效果产生了负面影响：

1. **过多的阶段**: 每个阶段都可能引入误差，累积效应导致准确率下降
2. **不合理的参数**: 8-gram对中文过大，top_k=5太小
3. **降级方案问题**: TF-IDF与语义向量不兼容
4. **复杂的融合逻辑**: 多种分数的融合增加了不确定性

**建议采用"纯语义模式"作为基准**：
- 仅使用高质量的语义向量
- 单一阈值控制
- 简单直接的逻辑
- 必要时加入轻量级的词汇过滤

这样可以获得更稳定、可预测的检测效果，同时便于调试和优化。