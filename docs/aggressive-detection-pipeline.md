# 激进版抄袭检测Pipeline设计方案

## 设计理念

### 核心原则
1. **Cross-encoder 是终极真相**：其输出直接作为相似度分数，不再融合
2. **语义向量仅用于召回**：快速筛选候选，不参与最终评分
3. **简化优于复杂**：移除不必要的融合和权重计算

### 为什么要激进？
- Cross-encoder 本身就是设计用来判断两个文本相似度的
- 既然花费 0.6 秒调用它，就应该完全信任它
- 多重融合反而引入噪声和不确定性

## 新Pipeline架构

### 方案A：极简主义（推荐）

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Semantic Recall │ --> │ Cross-Encoder   │ --> │ Final Results    │
│ (Bi-Encoder)    │     │ (Direct Score)  │     │                  │
└─────────────────┘     └─────────────────┘     └──────────────────┘
     宽进(0.65)              严出(0.75)            直接使用分数
```

**实现细节**：

```python
class AggressivePipeline:
    def __init__(self):
        self.stages = [
            SemanticRecallStage(threshold=0.65, top_k=50),  # 宽进
            CrossEncoderRerankStage(threshold=0.75),        # 严出
        ]
```

### 方案B：带MinHash快速过滤的优化版

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Semantic Recall │ --> │ MinHash Filter  │ --> │ Cross-Encoder   │ --> │ Final Results    │
│ (Bi-Encoder)    │     │ (3-gram+LSH)    │     │ (Direct Score)  │     │                  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └──────────────────┘
     宽进(0.60)          快速过滤(≈0.3)             终极裁判              直接输出
```

**实现细节**：

```python
class OptimizedAggressivePipeline:
    def __init__(self):
        self.stages = [
            SemanticRecallStage(threshold=0.60, top_k=50),
            MinHashFilterStage(
                shingle_size=3,      # 中文字符级3-gram
                num_perm=120,        # k=120个哈希函数
                bands=40,            # b=40个band
                rows=3,              # r=3，每band 3位
                threshold=0.3        # Jaccard阈值
            ),
            CrossEncoderRerankStage(threshold=0.75),
        ]
```

## FingerprintRecallStage 的新定位

### 当前问题
- 8-gram 对中文过大（"机器学习是人工智能的" ≈ 整句）
- 作为独立召回源产生太多噪声
- 与语义召回重复度高

### 三种处理方案

#### 1. **完全移除**（最激进）
- 优点：最简单，减少噪声
- 缺点：可能漏检纯复制粘贴
- 适用：学术论文等改写较多的场景

#### 2. **MinHash快速过滤器**（推荐）
```python
class MinHashFilterStage:
    """基于MinHash的快速过滤，高效近似Jaccard相似度"""

    def __init__(self, shingle_size=3, num_perm=120, bands=40, rows=3, threshold=0.3):
        self.shingle_size = shingle_size  # 3-gram 对中文友好
        self.num_perm = num_perm         # k=120个哈希函数
        self.bands = bands               # b=40个band
        self.rows = rows                 # r=3位/band
        self.threshold = threshold       # Jaccard阈值≈0.3

        # 验证参数
        assert num_perm == bands * rows, f"k必须等于b*r: {num_perm} != {bands}*{rows}"

        # 预计算哈希函数参数
        self.hash_params = self._init_hash_functions(num_perm)

        # LSH索引（用于大规模场景）
        self.lsh_index = {}

        # 停用shingle（降低噪声）
        self.stop_shingles = {
            "的是", "是的", "了的", "本文", "如下",
            "以下", "上述", "总结", "其中", "因此"
        }

    def _init_hash_functions(self, k):
        """初始化k个独立哈希函数参数"""
        import random
        random.seed(42)  # 可重现性

        # 使用(a*x+b) mod p形式的哈希函数
        p = 2**31 - 1  # 大质数
        params = []
        for _ in range(k):
            a = random.randint(1, p-1)
            b = random.randint(0, p-1)
            params.append((a, b, p))
        return params

    def _shingle(self, text):
        """生成3-gram字符级shingle集合"""
        if len(text) < self.shingle_size:
            return {text}

        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            # 过滤停用shingle
            if shingle not in self.stop_shingles:
                shingles.add(shingle)

        return shingles

    def _compute_minhash(self, shingles):
        """计算MinHash签名"""
        if not shingles:
            return [0] * self.num_perm

        # 将shingle转换为数值
        shingle_hashes = [hash(s) & 0x7FFFFFFF for s in shingles]

        # 计算每个哈希函数的最小值
        signature = []
        for a, b, p in self.hash_params:
            min_hash = min((a * h + b) % p for h in shingle_hashes)
            signature.append(min_hash)

        return signature

    def _estimate_jaccard(self, sig1, sig2):
        """通过MinHash签名估计Jaccard相似度"""
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)

    def _lsh_signature(self, minhash_sig):
        """将MinHash签名切分为LSH bands"""
        bands = []
        for i in range(self.bands):
            band = tuple(minhash_sig[i * self.rows:(i + 1) * self.rows])
            bands.append((i, band))  # (band_id, band_signature)
        return bands

    async def filter(self, candidates, left_chunks, right_chunks):
        """使用MinHash快速过滤候选对"""
        import asyncio

        # 1. 预计算所有文本的MinHash签名
        left_signatures = {}
        right_signatures = {}

        # 批量计算左侧文档
        for chunk_id in set(c.left_chunk_id for c in candidates):
            text = left_chunks[chunk_id].text
            shingles = self._shingle(text)
            left_signatures[chunk_id] = self._compute_minhash(shingles)

        # 批量计算右侧文档
        for chunk_id in set(c.right_chunk_id for c in candidates):
            text = right_chunks[chunk_id].text
            shingles = self._shingle(text)
            right_signatures[chunk_id] = self._compute_minhash(shingles)

        # 2. LSH索引构建（可选，适用于大规模）
        if len(candidates) > 1000:  # 大规模时使用LSH
            lsh_buckets = self._build_lsh_index(right_signatures)

        # 3. 过滤候选
        filtered = []
        for candidate in candidates:
            left_sig = left_signatures[candidate.left_chunk_id]
            right_sig = right_signatures[candidate.right_chunk_id]

            # 估计Jaccard相似度
            jaccard_est = self._estimate_jaccard(left_sig, right_sig)

            if jaccard_est >= self.threshold:
                # 记录估计的Jaccard值
                candidate.metadata["minhash_jaccard"] = jaccard_est
                filtered.append(candidate)

        return filtered

    def _build_lsh_index(self, signatures):
        """构建LSH索引用于快速查找"""
        buckets = {}
        for chunk_id, sig in signatures.items():
            bands = self._lsh_signature(sig)
            for band_id, band_sig in bands:
                key = (band_id, band_sig)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(chunk_id)
        return buckets

    def compute_candidate_probability(self, jaccard_similarity):
        """计算给定Jaccard相似度下的候选概率（S曲线）"""
        # P = 1 - (1 - s^r)^b
        prob = 1 - (1 - jaccard_similarity ** self.rows) ** self.bands
        return prob
```

#### 3. **并行召回源**（复杂度较高）
```python
# 不推荐：增加复杂性，收益有限
semantic_candidates = semantic_recall()
lexical_candidates = lexical_recall()
all_candidates = merge(semantic_candidates, lexical_candidates)
```

## 具体实现方案

### 1. 简化的 Pipeline 配置

```python
@dataclass
class AggressiveConfig:
    # 语义召回参数
    semantic_threshold: float = 0.65    # 降低阈值，宽进
    semantic_top_k: int = 50           # 增加候选数

    # MinHash过滤参数（可选）
    enable_minhash_filter: bool = True
    minhash_shingle_size: int = 3      # 中文友好的字符级3-gram
    minhash_num_perm: int = 120        # k=120个哈希函数
    minhash_bands: int = 40            # b=40个band
    minhash_rows: int = 3              # r=3位/band
    minhash_threshold: float = 0.3     # Jaccard相似度阈值

    # Cross-encoder 参数
    cross_encoder_threshold: float = 0.75  # 最终阈值
    cross_encoder_batch_size: int = 10     # 批处理大小

    # 降级方案
    fallback_to_semantic: bool = True      # API 不可用时回退
```

### 2. 新的相似度计算流程

```python
class AggressiveSimilarityPipeline:
    """激进版Pipeline - Cross-encoder 直接决定相似度"""

    async def detect_similarity(self, left_chunks, right_chunks):
        # Step 1: 语义召回（仅用于筛选）
        candidates = await self.semantic_recall(
            left_chunks,
            right_chunks,
            threshold=0.65,  # 宽松阈值
            top_k=50         # 多召回一些
        )

        # Step 2: MinHash过滤（可选，减少API调用）
        if self.config.enable_minhash_filter:
            candidates = await self.minhash_filter.filter(
                candidates,
                left_chunks,
                right_chunks
            )

        # Step 3: Cross-encoder 精排（最终得分）
        if self.cross_encoder_available():
            # 批量处理，提高效率
            final_scores = await self.cross_encoder_rerank(candidates)

            # 直接使用 Cross-encoder 分数
            results = [
                (cand, score)
                for cand, score in zip(candidates, final_scores)
                if score >= self.config.cross_encoder_threshold
            ]
        else:
            # 降级方案：使用语义分数
            results = [
                (cand, cand.semantic_score)
                for cand in candidates
                if cand.semantic_score >= 0.75
            ]

        return results
```

### 3. 性能优化策略

```python
class PerformanceOptimizations:
    """性能优化措施"""

    def __init__(self):
        # 1. 结果缓存
        self.cache = LRUCache(maxsize=10000)

        # 2. 批处理队列
        self.batch_queue = []
        self.batch_size = 10

        # 3. 异步处理
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def cached_cross_encoder(self, text_pairs):
        """带缓存的 Cross-encoder 调用"""
        results = []
        uncached_pairs = []
        uncached_indices = []

        # 检查缓存
        for i, (left, right) in enumerate(text_pairs):
            cache_key = f"{hash(left)}_{hash(right)}"
            cached_score = self.cache.get(cache_key)

            if cached_score is not None:
                results.append((i, cached_score))
            else:
                uncached_pairs.append((left, right))
                uncached_indices.append(i)

        # 批量处理未缓存的
        if uncached_pairs:
            new_scores = await self.cross_encoder_batch(uncached_pairs)

            # 更新缓存和结果
            for idx, score, (left, right) in zip(uncached_indices, new_scores, uncached_pairs):
                cache_key = f"{hash(left)}_{hash(right)}"
                self.cache[cache_key] = score
                results.append((idx, score))

        # 按原始顺序返回
        results.sort(key=lambda x: x[0])
        return [score for _, score in results]
```

## 与原方案对比

### 原方案的问题
```python
# 复杂的融合逻辑
final = 0.1 * lexical + 0.6 * semantic + 0.3 * cross
final = max(final, semantic)  # 又保护语义分数

# 问题：
# 1. 权重是猜的，没有理论依据
# 2. max 操作使融合失去意义
# 3. 不同来源的分数分布不一致
```

### 新方案的优势
```python
# 简单直接
if cross_encoder_available:
    final_score = cross_encoder_score  # 完全信任
else:
    final_score = semantic_score       # 降级方案

# 优势：
# 1. 逻辑清晰，易于理解
# 2. 充分利用 Cross-encoder 的能力
# 3. 减少了调参的复杂性
```

## 预期效果

### 准确性提升
- Cross-encoder 能更好地理解语义改写
- 减少了多阶段融合的误差累积
- 最终分数更可解释

### 性能影响
- API 调用次数：通过词汇过滤减少 30-50%
- 缓存命中率：预计 40-60%（重复检测场景）
- 总体延迟：取决于 Cross-encoder 性能

### 适用场景
1. **高精度要求**：学术论文、法律文件
2. **改写检测**：新闻洗稿、内容抄袭
3. **实时性要求不高**：批量检测、定期扫描

## 实施建议

### 第一阶段：试点
1. 保留原 Pipeline，添加激进模式开关
2. 在测试集上对比两种模式
3. 收集性能和准确性数据

### 第二阶段：优化
1. 根据数据调整阈值
2. 实现高效的缓存机制
3. 优化批处理策略

### 第三阶段：全面切换
1. 激进模式作为默认
2. 保留传统模式作为备选
3. 根据场景自动选择模式

## 代码示例

完整的激进版 Pipeline：

```python
class AggressivePlagiarismDetector:
    """激进版抄袭检测器 - Cross-encoder 主导"""

    def __init__(self, config: AggressiveConfig):
        self.config = config
        self.embedding_service = EmbeddingService()
        self.cross_encoder = CrossEncoderService()
        self.cache = LRUCache(maxsize=10000)

    async def detect(self, left_doc: str, right_doc: str) -> List[Match]:
        # 1. 文本预处理
        left_chunks = self.preprocess(left_doc)
        right_chunks = self.preprocess(right_doc)

        # 2. 语义召回（宽松）
        candidates = await self.semantic_recall(
            left_chunks,
            right_chunks,
            threshold=self.config.semantic_threshold,
            top_k=self.config.semantic_top_k
        )

        # 3. 词汇过滤（可选）
        if self.config.enable_lexical_filter:
            candidates = self.lexical_filter(candidates)

        # 4. Cross-encoder 评分
        if await self.cross_encoder.is_available():
            matches = await self.cross_encoder_score(
                candidates,
                threshold=self.config.cross_encoder_threshold
            )
        else:
            # 降级：直接使用语义分数
            matches = [
                Match(c.left, c.right, c.semantic_score)
                for c in candidates
                if c.semantic_score >= 0.75
            ]

        # 5. 后处理
        return self.post_process(matches)
```

## MinHash vs 传统Lexical Filter

### 为什么选择MinHash？

#### 1. **效率优势**
- **传统Jaccard**：需要完整计算两个集合的交并集，复杂度O(n)
- **MinHash**：只需比较固定长度的签名(120维)，复杂度O(k)
- **空间效率**：每个文本只存储120个整数，而非完整shingle集合

#### 2. **可扩展性**
- **LSH支持**：可以快速在海量文档中找到相似候选
- **并行化**：MinHash签名计算可以完全并行
- **预计算**：签名可以离线计算并缓存

#### 3. **参数选择理论依据**
```python
# S曲线计算
def s_curve(s, r=3, b=40):
    """计算给定Jaccard相似度的检出概率"""
    return 1 - (1 - s**r)**b

# 在s=0.3附近的表现
s_curve(0.25) ≈ 0.09  # 25%相似度，9%概率通过
s_curve(0.30) ≈ 0.19  # 30%相似度，19%概率通过
s_curve(0.35) ≈ 0.37  # 35%相似度，37%概率通过
s_curve(0.40) ≈ 0.58  # 40%相似度，58%概率通过
```

这个S曲线在0.3附近有明显转折，符合我们"轻过滤"的需求。

#### 4. **中文优化**
- **字符级3-gram**：适合中文，避免分词问题
- **停用shingle**：过滤常见无意义组合
- **签名稳定性**：同一文本始终产生相同签名

### MinHash参数调优指南

```python
# 不同应用场景的参数建议
scenarios = {
    "轻过滤（推荐）": {
        "k": 120, "b": 40, "r": 3,
        "threshold": 0.3,
        "说明": "快速过滤明显不相关，减少CE负担"
    },
    "中等过滤": {
        "k": 128, "b": 32, "r": 4,
        "threshold": 0.4,
        "说明": "平衡准确性和召回率"
    },
    "严格过滤": {
        "k": 150, "b": 30, "r": 5,
        "threshold": 0.5,
        "说明": "只保留高相似度候选"
    },
    "海量数据": {
        "k": 96, "b": 48, "r": 2,
        "threshold": 0.3,
        "说明": "优化LSH性能，牺牲一定准确性"
    }
}
```

## 总结

这个激进方案的核心是：**完全信任 Cross-encoder 的判断**，配合**MinHash高效过滤**。

优点：
- ✅ 逻辑简单清晰
- ✅ 充分利用最先进的模型
- ✅ MinHash提供高效的预筛选
- ✅ 可扩展到海量数据场景
- ✅ 结果更可解释

缺点：
- ⚠️ 依赖 Cross-encoder 的质量和可用性
- ⚠️ API 成本可能较高
- ⚠️ MinHash需要额外的内存存储签名

建议先在小范围试点，验证效果后再全面推广。特别是MinHash的参数(b,r)可以根据实际效果微调。