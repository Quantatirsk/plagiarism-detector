"""
配置管理 - 使用Pydantic Settings实现环境变量管理
支持本地/生产模式切换，遵循简单性和清晰性原则
"""
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MilvusMode(str, Enum):
    """Milvus运行模式枚举"""
    LOCAL = "local"
    SERVER = "server"


class Settings(BaseSettings):
    """应用配置类 - 所有配置项通过环境变量管理"""
    
    # API配置
    api_v1_prefix: str = Field(default="/api/v1", description="API路由前缀")
    project_name: str = Field(default="Plagiarism Detector", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    
    # OpenAI配置 (使用vect.one API)
    openai_api_key: str = Field(..., description="OpenAI-compatible API key")
    openai_base_url: str = Field(default="https://api.vect.one/v1", description="API base URL")
    embedding_model: str = Field(default="dengcao/Qwen3-Embedding-8B:Q5_K_M", description="嵌入模型")
    embedding_dimensions: int = Field(default=4096, description="嵌入向量维度")
    embedding_batch_size: int = Field(default=20, description="批处理大小")

    # Cross-encoder reranker (Jina AI)
    cross_encoder_model: str = Field(
        default="jina-reranker-v2-base-multilingual",
        description="Jina AI reranker model identifier",
    )
    jina_api_key: Optional[str] = Field(
        default=None,
        description="API key for Jina AI reranker integration (required when using Jina provider)",
    )

    # Reranker Configuration (支持多种提供商)
    reranker_provider: str = Field(
        default="openai",
        description="重排序服务提供商: jina 或 openai (默认: openai)",
    )
    reranker_model: str = Field(
        default="Qwen3-Reranker-0.6B",
        description="OpenAI兼容的重排序模型",
    )
    reranker_openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI重排序服务基础URL (如不设置则使用 openai_base_url)",
    )
    reranker_openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI重排序服务API密钥 (如不设置则使用 openai_api_key)",
    )
    
    # Milvus配置
    milvus_mode: MilvusMode = Field(default=MilvusMode.LOCAL, description="运行模式")
    milvus_db_file: str = Field(default="milvus_demo.db", description="本地数据库文件")
    milvus_host: str = Field(default="localhost", description="服务器地址")
    milvus_port: int = Field(default=19530, description="服务器端口")
    milvus_collection: str = Field(default="documents", description="集合名称")

    # 数据库配置
    database_url: str = Field(
        default="sqlite+aiosqlite:///./plagiarism.db",
        description="SQL 数据库连接字符串"
    )
    
    # Redis配置 (可选)
    redis_url: Optional[str] = Field(default="redis://localhost:6379", description="Redis连接URL")
    redis_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    # 性能配置
    max_concurrent_requests: int = Field(default=20, description="最大并发请求数")
    request_timeout: int = Field(default=30, description="请求超时时间(秒)")
    
    # 检测配置
    paragraph_similarity_threshold: float = Field(default=0.65, description="段落相似度阈值")  # Quick win: lowered from 0.75
    sentence_similarity_threshold: float = Field(default=0.70, description="句子相似度阈值")  # Quick win: lowered from 0.80
    top_k_paragraphs: int = Field(default=50, description="返回top-k个相似段落")
    max_total_matches: int = Field(default=2000, description="一次检测返回的最大匹配总数上限")

    # CORS 配置
    # 以逗号分隔的允许来源列表，例如："http://localhost:5173,https://your.app"
    cors_allow_origins: str = Field(default="http://localhost:5173", description="允许的跨域来源，逗号分隔")
    cors_allow_credentials: bool = Field(default=False, description="是否允许携带凭据")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # 忽略未定义的环境变量
    )
    
    @property
    def is_local_mode(self) -> bool:
        """是否为本地开发模式"""
        return self.milvus_mode == MilvusMode.LOCAL
    
    @property
    def is_production(self) -> bool:
        """是否为生产模式"""
        return self.milvus_mode == MilvusMode.SERVER
    
    def get_milvus_connection_params(self) -> dict:
        """获取Milvus连接参数"""
        if self.is_local_mode:
            return {"db_file": self.milvus_db_file}
        else:
            return {
                "host": self.milvus_host,
                "port": self.milvus_port
            }

    def get_cors_origins(self) -> list[str]:
        """返回允许的 CORS 来源列表"""
        raw = (self.cors_allow_origins or "").strip()
        if not raw:
            return ["*"]
        # 拆分并清理空白
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用lru_cache确保全局只有一个Settings实例
    """
    return Settings()


# 导出常用对象
settings = get_settings()
