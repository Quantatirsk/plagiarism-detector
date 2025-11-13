# 类型检查系统配置

## 配置文件

- `pyproject.toml` - mypy、pyright、ruff 配置
- `requirements-dev.txt` - 开发工具依赖
- `.vscode/settings.json` - IDE 集成

## 使用命令

```bash
# 代码质量检查 + 自动修复
ruff check backend/ --fix

# 类型检查
mypy backend/

# 统计信息
ruff check backend/ --statistics
```

## 当前状态

- 检查 66 个文件，排除 `backend/document-skills/`
- mypy: 128 个类型错误
- ruff: 482 个问题（1223 个已自动修复）

## 配置策略

渐进式采用（当前宽松模式）：
- 允许无类型函数定义
- 检查已标注的函数
- 新代码强制类型注解

## 常见修复

```python
# 使用 Any 而非 any
from typing import Any
def to_dict(self) -> dict[str, Any]: ...

# 明确 Optional
def func(value: list[str] | None = None): ...

# 添加类型注解
def process(data: str) -> int:
    return len(data)
```

参考完整配置指南: `docs/Python 后端类型检查配置指南.md`
