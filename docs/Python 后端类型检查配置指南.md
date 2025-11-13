# Python 后端类型检查配置指南

## 为什么需要类型检查

类型检查能在运行前发现：
- 类型不匹配（传错参数类型）
- 属性不存在（拼写错误、API 变更）
- 返回值类型错误
- None 值未处理

**核心价值**：将运行时错误提前到开发阶段。

---

## 工具选择

| 工具 | 特点 | 使用场景 |
|------|------|---------|
| **mypy** | 官方推荐，生态最成熟 | CLI 检查、CI/CD |
| **pyright** | 微软开发，速度快 | VSCode IDE 集成 |
| **ruff** | Rust 编写，极快 | Linter + 格式化 |

**推荐组合**：mypy（类型检查） + ruff（代码质量） + pyright（IDE 支持）

---

## 快速开始

### 1. 安装

```bash
pip install mypy pyright ruff

# 或添加到 requirements-dev.txt
echo "mypy>=1.8.0\npyright>=1.1.0\nruff>=0.3.0" >> requirements-dev.txt
```

### 2. 最小配置

创建 `pyproject.toml`：

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

# 第三方库无类型定义时不报错
[[tool.mypy.overrides]]
module = ["requests.*", "sqlalchemy.*", "redis.*"]
ignore_missing_imports = true

[tool.pyright]
include = ["src", "tests"]
exclude = ["**/__pycache__", "**/.venv"]
typeCheckingMode = "basic"
pythonVersion = "3.11"
reportMissingTypeStubs = false

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
]
```

### 3. 运行检查

```bash
# 类型检查
mypy src/

# 代码质量检查
ruff check src/

# IDE 检查（VSCode 自动运行）
pyright src/
```

---

## 完整配置详解

### mypy 配置选项

```toml
[tool.mypy]
# === 基础设置 ===
python_version = "3.11"
warn_return_any = true          # 警告返回 Any
warn_unused_configs = true      # 警告未使用的配置
warn_redundant_casts = true     # 警告冗余的类型转换
warn_unused_ignores = true      # 警告未使用的 type: ignore

# === 严格模式 ===
disallow_untyped_defs = true    # 禁止无类型注解的函数
disallow_any_generics = true    # 禁止泛型使用 Any
disallow_untyped_calls = true   # 禁止调用无类型的函数
no_implicit_optional = true     # 禁止隐式 Optional
strict_equality = true          # 严格相等性检查

# === 错误输出 ===
show_error_codes = true         # 显示错误代码
pretty = true                   # 美化输出

# === 第三方库 ===
[[tool.mypy.overrides]]
module = [
    "fastapi.*",
    "pydantic.*",
    "sqlalchemy.*",
    "redis.*",
    "celery.*",
]
ignore_missing_imports = true

# === 特定目录放松规则 ===
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### pyright 配置选项

```toml
[tool.pyright]
include = ["src", "tests"]
exclude = [
    "**/__pycache__",
    "**/.venv",
    "**/node_modules",
]

# === 类型检查级别 ===
# "off" | "basic" | "strict"
typeCheckingMode = "basic"

# === Python 版本 ===
pythonVersion = "3.11"
pythonPlatform = "Linux"

# === 报告控制 ===
reportMissingTypeStubs = false
reportUnknownMemberType = false
reportUnknownVariableType = false
reportUnknownArgumentType = false
reportPrivateUsage = "warning"
reportConstantRedefinition = "error"
```

### ruff 配置选项

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

# 排除目录
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "migrations",
]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "RUF",    # Ruff-specific rules
]

ignore = [
    "E501",   # line too long (formatter 处理)
    "B008",   # function calls in defaults (FastAPI Depends)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]        # 允许 __init__ 中未使用的导入
"tests/**/*.py" = ["S101"]      # 允许测试中使用 assert

[tool.ruff.lint.isort]
known-first-party = ["src", "app"]
```

---

## 使用指南

### 1. 基本类型注解

```python
# 函数类型注解
def greet(name: str) -> str:
    return f"Hello, {name}"

# 变量类型注解
count: int = 0
names: list[str] = []
mapping: dict[str, int] = {}

# 可选类型
from typing import Optional
def find_user(user_id: int) -> Optional[str]:
    return None

# Python 3.10+ 新语法
def find_user(user_id: int) -> str | None:
    return None
```

### 2. 高级类型

```python
from typing import Callable, TypeVar, Generic, Protocol

# 泛型
T = TypeVar('T')
def first(items: list[T]) -> T:
    return items[0]

# 回调函数
def apply(func: Callable[[int], str], value: int) -> str:
    return func(value)

# Protocol（结构化类型）
class Closable(Protocol):
    def close(self) -> None: ...

def close_resource(resource: Closable) -> None:
    resource.close()
```

### 3. FastAPI 类型注解

```python
from typing import Annotated
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 使用 Annotated 标注依赖注入
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)]
) -> dict[str, str]:
    return {"id": str(user_id)}
```

---

## 边缘情况处理

### 1. 第三方库无类型

**问题**：导入第三方库时 mypy 报错 "Cannot find implementation or library stub"

**解决方案**：

```toml
# pyproject.toml
[[tool.mypy.overrides]]
module = ["problematic_lib.*"]
ignore_missing_imports = true
```

或安装类型包：
```bash
pip install types-requests types-redis
```

### 2. 动态属性

**问题**：运行时添加的属性无法通过类型检查

```python
class Config:
    pass

config = Config()
config.dynamic = 123  # ❌ Error: "Config" has no attribute "dynamic"
```

**解决方案**：

```python
# 方案 1: 使用 __dict__
config.__dict__["dynamic"] = 123

# 方案 2: 局部禁用检查
config.dynamic = 123  # type: ignore[attr-defined]

# 方案 3: 使用 dataclass（推荐）
from dataclasses import dataclass, field

@dataclass
class Config:
    dynamic: int = field(default=0)
```

### 3. Any 类型传播

**问题**：Any 类型会"传染"到其他变量

```python
from typing import Any

data: Any = get_external_data()
result = process(data)  # result 也是 Any
```

**解决方案**：

```python
from typing import cast

# 显式类型转换
data = get_external_data()
typed_data = cast(dict[str, str], data)
result = process(typed_data)  # result 有正确类型

# 或使用 TypeGuard（Python 3.10+）
from typing import TypeGuard

def is_dict_str_str(val: Any) -> TypeGuard[dict[str, str]]:
    return isinstance(val, dict) and all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in val.items()
    )

data = get_external_data()
if is_dict_str_str(data):
    result = process(data)  # data 已被收窄为 dict[str, str]
```

### 4. 装饰器类型

**问题**：装饰器会丢失函数类型信息

```python
def decorator(func):  # ❌ 返回类型未知
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

**解决方案**：

```python
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
R = TypeVar('R')

def decorator(func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)
    return wrapper
```

### 5. 返回 None 的歧义

**问题**：函数可能返回 None 但未标注

```python
def get_config(key: str):  # ❌ 返回类型不明确
    if key in config:
        return config[key]
    return None
```

**解决方案**：

```python
from typing import Optional

def get_config(key: str) -> Optional[str]:
    if key in config:
        return config[key]
    return None

# Python 3.10+
def get_config(key: str) -> str | None:
    if key in config:
        return config[key]
    return None
```

---

## IDE 集成

### VSCode 配置

创建 `.vscode/settings.json`：

```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.linting.mypyEnabled": true,
  "python.linting.enabled": true,
  "python.formatting.provider": "none",

  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  }
}
```

### PyCharm 配置

1. **启用类型检查**：Settings → Editor → Inspections → Python → Type Checker
2. **配置 mypy**：Settings → Tools → External Tools → Add mypy
3. **格式化工具**：Settings → Tools → Black/Ruff

---

## CI/CD 集成

### GitHub Actions

```yaml
name: Type Check

on: [push, pull_request]

jobs:
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install mypy ruff

      - name: Run mypy
        run: mypy src/

      - name: Run ruff
        run: ruff check src/
```

### pre-commit 钩子

创建 `.pre-commit-config.yaml`：

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

安装：
```bash
pip install pre-commit
pre-commit install
```

---

## 渐进式采用策略

### 阶段 1：宽松模式（初期）

```toml
[tool.mypy]
check_untyped_defs = false
disallow_untyped_defs = false
disallow_untyped_calls = false

[tool.pyright]
typeCheckingMode = "basic"
```

### 阶段 2：基础模式（中期）

```toml
[tool.mypy]
check_untyped_defs = true
disallow_untyped_defs = true  # 新代码必须有类型

[[tool.mypy.overrides]]
module = "legacy.*"  # 旧代码暂时放松
disallow_untyped_defs = false
```

### 阶段 3：严格模式（成熟期）

```toml
[tool.mypy]
strict = true  # 启用所有严格选项

[tool.pyright]
typeCheckingMode = "strict"
```

---

## 常见问题

### Q1: 类型检查很慢怎么办？

```bash
# 使用缓存
mypy --cache-dir=.mypy_cache src/

# 只检查改动的文件
mypy $(git diff --name-only --diff-filter=ACMR | grep '\.py$')

# 使用 pyright（更快）
pyright src/
```

### Q2: 如何处理遗留代码？

```python
# 文件级别禁用
# type: ignore  # 在文件顶部

# 行级别禁用
result = legacy_function()  # type: ignore[no-untyped-call]

# 模块级别放松
# pyproject.toml
[[tool.mypy.overrides]]
module = "legacy.*"
ignore_errors = true
```

### Q3: 第三方库类型不准确怎么办？

创建 `typings/` 目录：

```python
# typings/external_lib.pyi
def problematic_function(x: int) -> str: ...
```

配置：
```toml
[tool.mypy]
mypy_path = "typings"
```

### Q4: 性能影响有多大？

- **开发阶段**：mypy 首次运行 10-30s，后续使用缓存 <1s
- **CI/CD**：增加 10-60s 构建时间
- **运行时**：零影响（类型注解在运行时被忽略）

---

## 最佳实践

1. **从新代码开始**：新写的代码强制类型注解
2. **渐进式迁移**：旧代码逐步添加类型
3. **优先公共 API**：先给对外接口加类型
4. **避免过度使用 Any**：Any 会破坏类型安全
5. **使用类型别名**：提高可读性
   ```python
   UserId = int
   UserMap = dict[UserId, str]
   ```
6. **编写类型测试**：确保类型推断正确
   ```python
   from typing import assert_type

   result = get_user(1)
   assert_type(result, Optional[User])
   ```

---

## 检查清单

部署前验证：

```bash
# 1. 工具已安装
mypy --version
pyright --version
ruff --version

# 2. 配置文件存在
ls pyproject.toml

# 3. 运行检查
mypy src/
pyright src/
ruff check src/

# 4. 查看错误统计
mypy src/ --txt-report reports/
```

---

## 参考资源

- [mypy 文档](https://mypy.readthedocs.io/)
- [pyright 文档](https://microsoft.github.io/pyright/)
- [ruff 文档](https://docs.astral.sh/ruff/)
- [Python typing 文档](https://docs.python.org/3/library/typing.html)
- [Type hints cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

---

**文档版本**: 1.0
**更新日期**: 2025-01-12
**适用 Python 版本**: 3.11+
