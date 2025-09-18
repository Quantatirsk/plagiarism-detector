# 多文档抄袭检测重构方案

## 1. 背景与目标
- **现状痛点**：当前服务仅支持一次对两篇文档做贪心匹配，阈值固定、块粒度僵硬；缺乏持久化模型来管理检测过程和历史记录；前端只能展示单次双文档结果。
- **新需求**：支持 N 篇文档两两比对、保留检测过程数据，用于历史追溯与分析，并能在前端按任务维度浏览、过滤、追踪雷同片段。
- **设计原则**：
  1. **Linus 法则** → 数据结构、算法和界面要透明可观察，方便新增测试与审查。
  2. **可扩展性** → 允许后续加入更多相似度策略或模型。
  3. **性能/体验平衡** → 多文档情况下减少重复计算，保证前端交互流畅。

## 2. 架构重构概览
```
┌──────────────────────────────┐
│          Web 前端            │
│  · 任务工作台 + 文档矩阵视图 │
│  · 雷同片段对齐查看器        │
└─────────────┬────────────────┘
              │ REST / WebSocket
┌─────────────▼────────────────┐
│        API/Orchestrator       │
│  · TaskService orchestrates   │
│  · DetectionEngine pipeline   │
│  · StorageGateway (SQL+Milvus)│
└───────┬──────────┬────────────┘
        │          │
┌───────▼───┐  ┌───▼────────────────┐
│SQL 数据库│  │ Milvus / 向量索引  │
│  ·任务/文档│  │  ·嵌入向量         │
│  ·块/候选 │  │  ·Top-k 粗召回      │
└──────────┘  └───────────────────┘
```

### 核心分层
1. **DetectionTask Service**：负责任务生命周期，生成两两比对计划，调度并缓存结果。
2. **SimilarityEngine**：包含指纹粗召回、语义重排、跨编码器精排、字符级对齐四段式流程。
3. **StorageGateway**：
   - 结构化数据库（Postgres/SQLite）用于任务、文档、块、候选匹配结果持久化。
   - Milvus/向量索引只存嵌入向量，用主键关联结构化数据。
4. **Front-end Workbench**：展示任务列表、文档矩阵和块对齐详情，可筛选/跳转。

## 3. 数据模型重构

### 3.1 关系数据模型（建议使用 SQL 数据库）
| 表 | 关键字段 | 说明 |
| --- | --- | --- |
| `detection_task` | `id`, `name`, `status`, `config_json`, `created_at`, `created_by` | 一次 N 文档检测任务；`config_json` 保存阈值、模型、过滤参数版本。|
| `task_document` | `id`, `task_id`, `document_id`, `alias`, `order_index` | 任务内引用的文档，支持同一原文档多版本。|
| `document` | `id`, `hash`, `filename`, `mime_type`, `char_count`, `lang_hint` | 去重管理原始文档，`hash` 防止重复上传。|
| `document_version` | `id`, `document_id`, `content_path`, `processed_text`, `preprocess_meta` | 保存解析后的文本及清洗信息，便于审计。|
| `text_chunk` | `id`, `document_version_id`, `chunk_type`, `chunk_index`, `start_pos`, `end_pos`, `text_hash` | 每个段落/句子/滑窗块；`text_hash` 支持快速指纹比较。|
| `chunk_embedding` | `id`, `text_chunk_id`, `vector_id`, `model`, `norm` | 向量信息，与 Milvus `vector_id` 关联。|
| `pair_plan` | `id`, `task_id`, `left_doc_version_id`, `right_doc_version_id`, `status` | 两两比对计划；用于前端显示矩阵。|
| `candidate_match` | `id`, `plan_id`, `left_chunk_id`, `right_chunk_id`, `rough_score`, `rough_method` | 指纹或 Milvus 粗召回结果。
| `match_evidence` | `id`, `candidate_id`, `semantic_score`, `lexical_overlap`, `alignment_ratio`, `final_score`, `label` | 精排后的证据数据；`label` 可用于人工标注迭代阈值。
| `match_span_alignment` | `id`, `match_evidence_id`, `left_span`, `right_span`, `diff_patch` | 字符级对齐信息，用于前端高亮。

### 3.2 向量存储
- **Collection 设计**：使用 Milvus `documents_multi`，字段包含 `vector`, `chunk_id`, `task_id`, `chunk_type`, `position`。
- **索引策略**：HNSW + Cosine；支持按 `task_id` 或 `document_id` filter。

### 3.3 缓存与消息
- 使用 Redis 记录任务状态、进度条、并发控制（可沿用现有设置）。
- 后续可加入消息队列（Celery/RQ）用于后台批处理。

## 4. 算法流程重构

### 4.1 预处理与切片
1. 文档解析：沿用 `DocumentParser`，增加缓存能力将解析结果与 `document_version` 对齐。
2. 切片策略：
   - 精细化：支持段落、句子、滑动窗口（可配置窗口 40-80 tokens，步长 20）。
   - 多语言阈值自适应：根据字符长度/语言自动调整最短长度与停用词过滤。
   - 引入重叠片段：保证覆盖率，存入 `text_chunk`。

### 4.2 指纹粗召回 (Phase 1)
- **Shingling + MinHash/SimHash**：对每个 `text_chunk` 生成 `n` 份签名，构建 LSH 索引（内存或 SQLite）。
- **跨文档对比**：`pair_plan` 触发时先用指纹过滤，获得高概率雷同的 `candidate_match`。
- 优点：捕获逐字/格式雷同，同时快速剔除完全无关的块。

### 4.3 语义重排 (Phase 2)
- 对每个候选块获取/生成嵌入，查询 Milvus top-k，合并候选集去重。
- 得到 `rough_score`（1 - cosine distance 转相似度），写入 `candidate_match`。

### 4.4 跨编码器精排 (Phase 3)
- 对候选对跑轻量 cross-encoder（如 bge reranker, m3e reranker），出 `semantic_score`。
- 融合指标（语义 + 词面重叠）得到 `final_score`。
- 支持动态阈值：根据任务配置或统计分位数调整，写入 `match_evidence`。

### 4.5 字符/Token 级对齐 (Phase 4)
- 使用 Smith-Waterman 或 difflib/rapidfuzz 生成匹配 span，计算 `alignment_ratio`、`diff_patch`。
- 用于前端高亮与导出报告。

### 4.6 Linus 法则落地
- 每个阶段写入明确信息流，方便测试和 Debug。
- 在 `tests/` 新增仿真数据集：
  - 短句重复、段落复制、模板化、同义改写、多处分布抄袭等用例。
  - 用 fixture 快速跑 end-to-end pipeline。

## 5. API 与服务层调整
- 新增 `TaskController`：
  - `POST /tasks` 创建任务，上传/引用多篇文档。
  - `GET /tasks/{id}` 返回任务状态、矩阵、总体评分。
  - `GET /tasks/{id}/pairs/{pair_id}` 返回单 pair 的匹配证据和对齐片段。
  - `POST /tasks/{id}/rerun` 支持指定参数重新跑。
- `DualDocumentDetectionService` 演进为 `DetectionEngine`：
  - 接受 `pair_plan`，管道式执行四阶段流程。
  - 可多运行实例并发处理。
- `StorageGateway` 提供 `upsert_document`, `record_candidate`, `record_evidence` 等接口。

## 6. 前端重构
1. **任务工作台页**
   - 任务列表（状态、创建时间、配置摘要）。
   - `TaskDetail` 展示矩阵视图：行列分别为文档；矩阵单元显示总体评分、匹配数量。
   - 支持按阈值、匹配数量、标签过滤。
2. **Pair Detail Viewer**
   - 左右文档滚动条联动，块列表按 `final_score` 排序。
   - 点击某条匹配进入字符级高亮，显示指纹/语义分数、对齐差异。
   - 允许标记“确认抄袭/误报”，回写 `match_evidence.label`。
3. **交互增强**
   - 进度跟踪：任务执行时展示阶段进度（粗召回→语义→精排→对齐）。
   - 支持导出报告（PDF/CSV），通过后端聚合 `match_evidence`。

## 7. 迁移与兼容策略
- **阶段化推出**：
  1. 引入新数据模型 + API，保留旧接口兼容单任务。
  2. 新引擎替换旧 `_find_matches`，提供迁移脚本把旧数据写入新结构。
  3. 前端切换到任务工作台视图后再下线旧页面。
- **数据迁移脚本**：从现有结果 JSON 生成 `detection_task`、`pair_plan`、`match_evidence`。
- **性能验证**：在样本集上比较召回率/Precision、平均耗时；根据结果调整窗口与阈值。

## 8. 测试与质量保障
- `tests/integration/test_multi_task_pipeline.py`：构造 3 文档任务，验证矩阵和匹配结果。
- `tests/unit/test_similarity_engine.py`：Mock 各阶段，确保数据在数据库中正确写入。
- `tests/ui/`（如使用 Playwright）确认前端矩阵和高亮交互。
- 监控：API 记录各阶段耗时，接入 Prometheus/Grafana。

## 9. 后续扩展
- 支持增量文档库：对比新文档与历史库。
- 引入作者指纹、引用检测策略，提高抄袭类型覆盖面。
- 引入主动学习：用户标注回灌到阈值/模型调优。

---
该方案在数据模型、算法与前端三方面同步演进，确保多文档检测的准确性和可维护性，并可逐步扩展更多检测能力。
