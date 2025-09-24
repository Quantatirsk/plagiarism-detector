# 匹配策略指南

本文档介绍了系统中提供的两种最佳匹配选择机制，用于解决"A文档同一段落与B文档多个片段匹配"的问题。

> **重要更新**：从现在起，系统的默认匹配策略已更改为 `bidirectional_stable`（双向稳定匹配），以提供更高质量的匹配结果。

## 背景

在默认的抄袭检测流程中，系统会保留所有超过相似度阈值的匹配对。这可能导致：
- 一个源段落匹配多个目标段落
- 结果中包含大量次优匹配
- 用户难以快速识别最重要的匹配

## 方案对比

### 方案一：基于分数的贪心最佳匹配

**原理**：对每个源段落，只保留分数最高的目标段落匹配。

**实现位置**：`MatchAggregator` 类

**使用方式**：
```python
config = ComparisonConfig(
    mode=DetectionMode.AGGRESSIVE,
    match_strategy="best_per_source",
    semantic_threshold=0.65,
    final_threshold=0.75,
)
```

**优点**：
- 实现简单，计算效率高
- 确保每个源段落最多只有一个匹配
- 结果清晰，易于理解

**缺点**：
- 可能丢失重要的次优匹配
- 不考虑反向匹配（目标到源的最佳选择）
- 可能出现多个源段落匹配同一个目标段落的情况

**适用场景**：
- 需要快速获得最可能的匹配结果
- 文档结构相对简单，不涉及复杂的拆分或重组
- 重视处理速度和结果简洁性

### 方案二：基于双向最佳匹配（稳定匹配）

**原理**：只保留互为最佳选择的匹配对，类似"稳定婚姻算法"。

**实现位置**：`BidirectionalMatchFilter` 类

**使用方式**：

稳定版本（严格）：
```python
config = ComparisonConfig(
    mode=DetectionMode.AGGRESSIVE,
    match_strategy="bidirectional_stable",
    semantic_threshold=0.65,
    final_threshold=0.75,
)
```

宽松版本（推荐）：
```python
config = ComparisonConfig(
    mode=DetectionMode.AGGRESSIVE,
    match_strategy="bidirectional_relaxed",
    relaxed_threshold_ratio=0.95,  # 分数达到最佳的95%即可
    semantic_threshold=0.65,
    final_threshold=0.75,
)
```

**优点**：
- 匹配结果更加稳定和对称
- 减少了"多对一"的匹配情况
- 宽松版本在保持质量的同时提高召回率

**缺点**：
- 计算复杂度稍高（需要双向查找）
- 严格版本可能过于保守，丢失合理匹配
- 需要额外的后处理步骤

**适用场景**：
- 需要高质量、高可信度的匹配结果
- 文档对比用于正式的学术或法律用途
- 希望减少误报，宁缺毋滥

## 性能对比

| 指标 | 默认（全部保留） | 方案一（贪心） | 方案二（稳定） | 方案二（宽松） |
|-----|---------------|--------------|--------------|--------------|
| 处理速度 | 基准 | 略快 | 略慢 | 略慢 |
| 结果数量 | 最多 | 中等 | 最少 | 较少 |
| 准确率 | 中等 | 较高 | 最高 | 高 |
| 召回率 | 最高 | 中等 | 较低 | 较高 |
| 实现复杂度 | 低 | 低 | 中等 | 中等 |

## 选择建议

1. **日常检测**：使用方案一（`best_per_source`），快速获得清晰结果
2. **学术论文检测**：使用方案二宽松版（`bidirectional_relaxed`），平衡准确率和召回率
3. **法律文档对比**：使用方案二严格版（`bidirectional_stable`），确保高可信度
4. **调试和分析**：使用默认策略（`all`），查看所有可能的匹配

## API 示例

```python
from backend.services.comparison_service import ComparisonService, ComparisonConfig
from backend.models.detection_modes import DetectionMode

# 创建比较服务
comparison_service = ComparisonService()

# 使用默认配置（现在是 bidirectional_stable）
default_config = ComparisonConfig()  # 自动使用双向稳定匹配

# 或显式指定其他策略
configs = {
    "保留全部": ComparisonConfig(
        mode=DetectionMode.AGGRESSIVE,
        match_strategy="all"  # 恢复旧的默认行为
    ),
    "快速模式": ComparisonConfig(
        mode=DetectionMode.FAST,
        match_strategy="best_per_source"
    ),
    "平衡模式": ComparisonConfig(
        mode=DetectionMode.AGGRESSIVE,
        match_strategy="bidirectional_relaxed",
        relaxed_threshold_ratio=0.90
    ),
    "严格模式": ComparisonConfig(
        mode=DetectionMode.AGGRESSIVE,
        match_strategy="bidirectional_stable"  # 现在是默认值
    ),
}

# 运行对比
await comparison_service.run_pair(pair_id, default_config)  # 使用新的默认策略
```

## 未来扩展

1. **基于上下文的匹配**：考虑段落在文档中的位置和上下文
2. **机器学习优化**：使用历史数据训练最佳匹配选择模型
3. **用户反馈集成**：根据用户标注的正确/错误匹配优化策略
4. **混合策略**：结合多种策略的优点，动态选择最合适的方法