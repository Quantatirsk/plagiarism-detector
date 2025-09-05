# 开发路线图 - FastAPI向量嵌入文档检测系统

## 🎯 核心原则
> "Talk is cheap. Show me the code." - Linus Torvalds

- **简单性**: 能简单就不要复杂
- **实用性**: 先跑起来，再优化  
- **清晰性**: 代码即文档

## 📋 快速开始检查清单

### Phase 1: 环境准备 (30分钟)
- [ ] 创建项目目录结构
- [ ] 安装Python 3.11+
- [ ] 创建虚拟环境: `python -m venv venv`
- [ ] 激活环境: `source venv/bin/activate` (Mac/Linux)
- [ ] 安装依赖: `pip install -r requirements.txt`
- [ ] 复制环境配置: `cp .env.example .env`
- [ ] 设置OpenAI API Key
- [ ] 验证: `python -c "from pymilvus import MilvusClient; print('Success!')"`

### Phase 2: 最小可运行版本 (MVP) - 第1天

#### 上午：基础架构
```bash
# 1. 创建目录结构
mkdir -p app/{api/v1,core,models,services,repositories,utils} tests

# 2. 创建所有__init__.py文件
find app -type d -exec touch {}/__init__.py \;

# 3. 实现核心配置
# app/core/config.py - Settings类，支持LOCAL模式
```

#### 下午：核心服务
```python
# 优先级顺序：
1. app/core/config.py          # 配置管理
2. app/core/errors.py          # 错误处理
3. app/models/document.py      # 数据模型
4. app/models/detection.py     # 检测模型
5. app/services/text_processor.py  # 文本处理
```

### Phase 3: 向量存储实现 - 第2天

#### 上午：Milvus本地模式
```python
# app/services/storage.py
# 重点：使用 MilvusClient("milvus_demo.db") 本地模式
# 不需要启动Milvus服务器！
```

#### 下午：OpenAI嵌入
```python
# app/services/embedding.py
# 实现批处理和重试机制
# 测试：生成一个测试向量验证
```

### Phase 4: 检测核心逻辑 - 第3天

#### 上午：检测服务
```python
# app/services/detection.py
# 段落级检测 → 句子级检测（可选）
# 相似度阈值：0.75（段落）、0.80（句子）
```

#### 下午：API端点
```python
# app/api/v1/detection.py
# POST /api/v1/detection/check
# app/main.py - FastAPI应用入口
```

### Phase 5: 测试验证 - 第4天

#### 端到端测试流程
```bash
# 1. 启动应用
uvicorn app.main:app --reload

# 2. 检查健康状态
curl http://localhost:8000/api/v1/health

# 3. 查看API文档
open http://localhost:8000/docs

# 4. 测试检测API
curl -X POST http://localhost:8000/api/v1/detection/check \
  -H "Content-Type: application/json" \
  -d '{"content": "测试文本...", "mode": "fast", "threshold": 0.75}'
```

## 🚀 执行优先级

### 必须完成 (Critical Path)
1. **环境配置** → `.env` 文件，OpenAI API Key
2. **Milvus本地模式** → `MilvusClient("milvus_demo.db")`
3. **文本嵌入** → OpenAI embedding API
4. **检测API** → `/api/v1/detection/check`

### 可以延后 (Can Defer)
- Redis缓存 (直接返回结果即可)
- 文档上传API (手动测试即可)
- Prometheus监控
- Docker部署

### 可以简化 (Can Simplify)
- 健康检查：返回 `{"status": "ok"}` 即可
- 日志：使用print()调试，后续再换structlog
- 批处理优化：先实现单个处理

## 📊 关键指标验证

### 功能验证
```python
# tests/test_quick.py
import asyncio
from app.services.embedding import EmbeddingService
from app.services.storage import MilvusStorage

async def quick_test():
    # 1. 测试嵌入
    embedding_service = EmbeddingService()
    vector = await embedding_service.embed_text("Hello World")
    print(f"✅ 嵌入维度: {len(vector)}")  # 应该是3072
    
    # 2. 测试存储
    storage = MilvusStorage()
    print(f"✅ Milvus模式: {storage.mode}")  # 应该是LOCAL
    
    # 3. 测试插入和搜索
    # ... 

asyncio.run(quick_test())
```

### 性能基准
- 响应时间目标: < 100ms (本地模式)
- 准确率目标: > 85%
- 并发支持: 100 requests/second

## 🔧 常见问题解决

### 1. OpenAI API错误
```python
# 检查API Key
import os
print(os.getenv("OPENAI_API_KEY"))

# 测试连接
import openai
client = openai.OpenAI()
response = client.embeddings.create(
    input="test",
    model="text-embedding-3-large",
    dimensions=3072
)
```

### 2. Milvus连接问题
```python
# 本地模式不需要服务器！
from pymilvus import MilvusClient
client = MilvusClient("milvus_demo.db")  # 创建本地文件
print(client.list_collections())
```

### 3. 依赖版本冲突
```bash
# 使用精确版本
pip install pymilvus==2.4.1  # 支持MilvusClient
pip install openai==1.106.1
pip install fastapi==0.116.1
```

## 📈 开发进度跟踪

### Day 1 目标
- [ ] 项目结构创建完成
- [ ] 配置管理可用
- [ ] 基础模型定义完成
- [ ] 文本处理服务可用

### Day 2 目标  
- [ ] OpenAI嵌入服务工作
- [ ] Milvus本地存储工作
- [ ] 向量插入和搜索成功

### Day 3 目标
- [ ] 检测逻辑实现
- [ ] API端点可访问
- [ ] Swagger文档生成

### Day 4 目标
- [ ] 端到端测试通过
- [ ] 性能达到基准
- [ ] 准确率验证

## 💡 开发技巧

### 1. 增量开发
```bash
# 每完成一个模块就测试
python -m pytest tests/test_text_processor.py -v
python -m pytest tests/test_embedding.py -v
```

### 2. 使用Python交互式测试
```python
# ipython 或 python
from app.core.config import get_settings
settings = get_settings()
print(settings.MILVUS_MODE)  # 应该显示 MilvusMode.LOCAL
```

### 3. 日志调试
```python
# 临时调试
print(f"[DEBUG] 向量维度: {len(embedding)}")
print(f"[DEBUG] 搜索结果: {len(matches)}")

# 后续替换为
logger.debug("向量维度", dimension=len(embedding))
```

## 🎓 学习资源

1. **Milvus本地模式**: https://milvus.io/docs/zh/quickstart.md
2. **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
3. **FastAPI**: https://fastapi.tiangolo.com/zh/

## ✅ 完成标准

### MVP完成标志
- [ ] `uvicorn app.main:app` 可以启动
- [ ] `/docs` 页面可访问
- [ ] `/api/v1/detection/check` 返回结果
- [ ] 本地文件 `milvus_demo.db` 创建成功
- [ ] 测试文本相似度检测准确

### 生产就绪标志
- [ ] 所有测试通过
- [ ] Docker容器可运行
- [ ] 切换到SERVER模式成功
- [ ] 性能和准确率达标
- [ ] 部署文档完整

---

**记住：先让它工作，再让它正确，最后让它快速。**