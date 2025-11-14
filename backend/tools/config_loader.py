"""
工具模块统一配置加载器

提供与主应用一致的分层配置加载机制，确保工具模块和测试能正确加载环境配置
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 配置日志记录器
logger = logging.getLogger(__name__)
class ToolsConfigLoader:
    """工具模块配置加载器"""

    _initialized = False

    @classmethod
    def load_config(cls, environment: str | None = None, force_reload: bool = False) -> str:
        """
        加载分层配置文件

        Args:
            environment: 指定环境 (local/staging/production)，默认从ENVIRONMENT环境变量读取
            force_reload: 是否强制重新加载配置

        Returns:
            加载的环境名称
        """
        if cls._initialized and not force_reload:
            return os.getenv('ENVIRONMENT', 'local')

        # 获取backend-python目录路径
        backend_dir = cls._get_backend_dir()

        # 1. 先加载 .env 获取 ENVIRONMENT 值
        env_file = backend_dir / '.env'
        if env_file.exists():
            load_dotenv(env_file)

        # 确定环境
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'local')
        else:
            # 如果指定了环境，设置到环境变量中
            os.environ['ENVIRONMENT'] = environment

        logger.info(f"[Tools] Loading configuration for environment: {environment}")

        # 2. 加载基础配置（所有环境共享）
        base_config = backend_dir / '.env.base'
        if base_config.exists():
            logger.info("[Tools] Loading base configuration...")
            load_dotenv(base_config, override=False)

        # 3. 加载环境特定配置（覆盖基础配置，但不覆盖系统环境变量）
        env_specific = backend_dir / f'.env.{environment}'
        if env_specific.exists():
            logger.info(f"[Tools] Loading {environment} environment configuration...")
            load_dotenv(env_specific, override=False)  # 不覆盖已存在的环境变量
        else:
            logger.warning(f"[WARNING] [Tools] {env_specific} not found, using base configuration only")

        # 4. 系统环境变量具有最高优先级（已通过override=False保证）

        cls._initialized = True
        cls._validate_config(environment)

        return environment

    @classmethod
    def _get_backend_dir(cls) -> Path:
        """获取backend-python目录路径"""
        # 获取当前文件所在目录 (tools目录)
        tools_dir = Path(__file__).parent
        # 上级目录就是backend-python
        backend_dir = tools_dir.parent

        # 确保在正确的目录
        if not (backend_dir / 'main.py').exists():
            # 如果main.py不存在，尝试其他方式定位
            # 从sys.path中查找
            for path in sys.path:
                path_obj = Path(path)
                if (path_obj / 'main.py').exists() and 'backend-python' in str(path_obj):
                    backend_dir = path_obj
                    break

        return backend_dir

    @classmethod
    def _validate_config(cls, environment: str) -> None:
        """验证配置加载结果"""
        # 输出一些关键配置的加载状态（不输出敏感信息）
        logger.info(f"[OK] [Tools] Environment: {environment}")

        # 检查一些常用配置是否存在
        configs_to_check = {
            'OCR_BASE_URL': 'OCR服务',
            'MTRANS_BASE_URL': '翻译服务',
            'DB_HOST': '数据库',
            'OPENAI_BASE_URL': 'OpenAI服务'
        }

        for key, name in configs_to_check.items():
            value = os.getenv(key)
            if value:
                # 只显示配置已设置，不显示具体值
                logger.info(f"[OK] [Tools] {name} configuration loaded")
            else:
                logger.warning(f"[WARNING] [Tools] {name} configuration not set ({key})")

    @classmethod
    def reset(cls) -> None:
        """重置配置加载状态（主要用于测试）"""
        cls._initialized = False

def load_tools_config(environment: str | None = None, force_reload: bool = False) -> str:
    """
    便捷函数：加载工具模块配置

    Args:
        environment: 指定环境 (local/staging/production)
        force_reload: 是否强制重新加载

    Returns:
        加载的环境名称
    """
    return ToolsConfigLoader.load_config(environment, force_reload)

# 在模块导入时自动加载配置
def auto_load_config() -> Any:
    """自动加载配置（仅在作为模块导入时执行一次）"""
    # 检查是否已经通过主应用加载了配置
    # 如果ENVIRONMENT已设置且有其他配置，说明主应用已加载
    if not os.getenv('ENVIRONMENT'):
        # 没有设置环境变量，需要加载配置
        load_tools_config()
    elif not os.getenv('OCR_BASE_URL') and not os.getenv('MTRANS_BASE_URL'):
        # 环境变量设置了但没有工具配置，可能需要重新加载
        load_tools_config()

# 模块导入时自动执行
# 但不影响主应用的配置加载
if __name__ != '__main__':
    # 只在作为模块导入时自动加载，不在直接运行时加载
    auto_load_config()
