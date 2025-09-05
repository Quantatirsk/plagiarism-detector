#!/usr/bin/env python3
"""
项目初始化脚本 - 遵循Linus原则：先跑起来
使用方法: python scripts/init_project.py
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """创建项目目录结构"""
    base_dir = Path(__file__).parent.parent
    
    # 定义目录结构
    directories = [
        "app",
        "app/api",
        "app/api/v1", 
        "app/core",
        "app/models",
        "app/services",
        "app/repositories",
        "app/utils",
        "tests",
        "tests/fixtures",
        "docs",
        "scripts"
    ]
    
    print("📁 创建目录结构...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建__init__.py文件
        if directory.startswith("app") or directory == "tests":
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Package initialization."""\n')
                print(f"  ✅ {directory}/__init__.py")

def create_env_files():
    """创建环境配置文件"""
    base_dir = Path(__file__).parent.parent
    
    # .env.example 内容
    env_example = """# API Configuration
API_V1_PREFIX="/api/v1"
PROJECT_NAME="Plagiarism Detector"
VERSION="1.0.0"

# OpenAI Configuration
OPENAI_API_KEY="sk-your-api-key-here"
OPENAI_MODEL="text-embedding-3-large"
OPENAI_DIMENSIONS=3072
OPENAI_BATCH_SIZE=100

# Milvus Configuration
MILVUS_MODE="local"                    # 本地开发模式
MILVUS_DB_FILE="milvus_demo.db"        # 本地数据库文件
MILVUS_HOST="localhost"                # 生产服务器地址(暂不使用)
MILVUS_PORT=19530                       # 生产服务器端口(暂不使用)
MILVUS_COLLECTION="documents"          # 集合名称

# Redis Configuration (可选，暂不使用)
REDIS_URL="redis://localhost:6379"
REDIS_TTL=3600

# Performance Configuration
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30

# Detection Configuration
PARAGRAPH_SIMILARITY_THRESHOLD=0.75
SENTENCE_SIMILARITY_THRESHOLD=0.80
TOP_K_PARAGRAPHS=50
"""
    
    env_example_file = base_dir / ".env.example"
    env_example_file.write_text(env_example)
    print("📄 创建 .env.example")
    
    # 如果.env不存在，创建一个
    env_file = base_dir / ".env"
    if not env_file.exists():
        env_file.write_text(env_example)
        print("📄 创建 .env (请记得添加你的OpenAI API Key!)")
    
    # .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Project specific
.env
milvus_demo.db
*.log
*.db
.pytest_cache/
htmlcov/
.coverage
*.egg-info/

# Temporary files
tmp/
temp/
*.tmp
"""
    
    gitignore_file = base_dir / ".gitignore"
    gitignore_file.write_text(gitignore)
    print("📄 创建 .gitignore")

def create_requirements():
    """创建requirements.txt文件"""
    base_dir = Path(__file__).parent.parent
    
    requirements = """# Core Dependencies
fastapi==0.116.1
uvicorn[standard]==0.35.0
pydantic==2.11.7
pydantic-settings==2.10.1

# AI & Vector Database
openai==1.106.1
pymilvus==2.4.1  # 支持MilvusClient本地模式

# Optional (可以后期添加)
redis==6.4.0
httpx==0.25.2
python-multipart==0.0.20

# Utilities
tenacity==9.1.2  # 重试机制
structlog==23.2.0  # 结构化日志
scikit-learn==1.3.2  # 句子相似度计算
numpy==1.24.3

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Optional Monitoring (可以后期添加)
# prometheus-fastapi-instrumentator==6.1.0
"""
    
    req_file = base_dir / "requirements.txt"
    req_file.write_text(requirements)
    print("📄 创建 requirements.txt")

def create_readme():
    """创建README文件"""
    base_dir = Path(__file__).parent.parent
    
    readme = """# FastAPI 向量嵌入文档检测系统

> "Talk is cheap. Show me the code." - Linus Torvalds

## 🚀 快速开始

### 1. 安装依赖
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. 配置环境
```bash
cp .env.example .env
# 编辑 .env 文件，添加你的 OpenAI API Key
```

### 3. 运行应用
```bash
uvicorn app.main:app --reload
```

### 4. 查看文档
打开浏览器访问: http://localhost:8000/docs

## 📁 项目结构
```
plagiarism-detector/
├── app/                # 应用代码
│   ├── api/           # API端点
│   ├── core/          # 核心配置
│   ├── models/        # 数据模型
│   ├── services/      # 业务逻辑
│   └── utils/         # 工具函数
├── tests/             # 测试代码
├── docs/              # 文档
└── scripts/           # 脚本工具
```

## 🎯 核心功能
- 文档相似度检测
- OpenAI文本嵌入
- Milvus向量存储（本地模式）
- FastAPI REST API

## 📊 性能指标
- 响应时间: < 100ms
- 准确率: > 85%
- 并发支持: 100 req/s
"""
    
    readme_file = base_dir / "README.md"
    if not readme_file.exists():
        readme_file.write_text(readme)
        print("📄 创建 README.md")

def create_minimal_app():
    """创建最小可运行的应用代码"""
    base_dir = Path(__file__).parent.parent
    
    # 创建一个简单的main.py来验证环境
    main_py = '''"""
FastAPI应用入口 - 最小可运行版本
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Plagiarism Detector",
    version="1.0.0",
    description="向量嵌入文档检测系统"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Plagiarism Detector",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "mode": "development"}

# 临时测试端点
@app.post("/api/v1/detection/check")
async def check_plagiarism(content: str):
    """检测端点占位符"""
    return {
        "message": "Detection endpoint placeholder",
        "content_length": len(content),
        "status": "Not implemented yet"
    }
'''
    
    main_file = base_dir / "app" / "main.py"
    main_file.write_text(main_py)
    print("📄 创建 app/main.py (最小可运行版本)")

def check_environment():
    """检查Python环境"""
    print("\n🔍 环境检查...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ⚠️  Python版本过低，建议使用3.8+")
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("  ✅ 虚拟环境已激活")
    else:
        print("  ⚠️  建议使用虚拟环境")

def print_next_steps():
    """打印下一步操作指南"""
    print("\n" + "="*50)
    print("🎉 项目初始化完成！")
    print("="*50)
    print("\n📋 下一步操作：")
    print("1. 安装依赖:")
    print("   pip install -r requirements.txt")
    print("\n2. 配置OpenAI API Key:")
    print("   编辑 .env 文件，替换 'sk-your-api-key-here'")
    print("\n3. 测试运行:")
    print("   uvicorn app.main:app --reload")
    print("\n4. 查看API文档:")
    print("   打开 http://localhost:8000/docs")
    print("\n5. 开始开发:")
    print("   查看 docs/development-roadmap.md")
    print("\n提示: 使用本地Milvus模式，无需安装Milvus服务器！")
    print("="*50)

def main():
    """主函数"""
    print("🚀 FastAPI向量嵌入文档检测系统 - 项目初始化")
    print("遵循Linus原则：先让它工作！\n")
    
    # 执行初始化步骤
    check_environment()
    create_directory_structure()
    create_env_files()
    create_requirements()
    create_readme()
    create_minimal_app()
    
    # 打印完成信息
    print_next_steps()

if __name__ == "__main__":
    main()