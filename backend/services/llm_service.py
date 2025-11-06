"""
LLM服务 - OpenAI兼容的语言模型服务
用于生成抄袭检测报告和分析
"""
from typing import List, Optional, Dict, Any, AsyncGenerator
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.errors import LLMError
from backend.services.base_service import BaseService, singleton


@singleton
class LLMService(BaseService):
    """OpenAI兼容的LLM服务 - 支持聊天完成和流式响应"""

    def _initialize(self):
        """初始化OpenAI客户端"""
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=60  # 默认60秒超时
        )
        self.default_model = self.settings.llm_model
        self.max_tokens = 4096  # 默认最大令牌数
        self.temperature = 0.7  # 默认温度

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        创建聊天完成请求

        Args:
            messages: 消息列表
            model: 模型名称 (可选，默认使用配置的模型)
            temperature: 生成温度 (可选)
            max_tokens: 最大令牌数 (可选)
            stream: 是否流式响应
            **kwargs: 其他OpenAI API参数

        Returns:
            聊天完成响应
        """
        self._ensure_initialized()

        try:
            params = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": stream,
                **kwargs
            }

            self.logger.info(
                "Creating chat completion",
                model=params["model"],
                message_count=len(messages),
                stream=stream
            )

            response = await self.client.chat.completions.create(**params)

            if not stream:
                self.logger.info(
                    "Chat completion successful",
                    model=params["model"],
                    usage=response.usage.dict() if response.usage else None
                )

            return response

        except Exception as e:
            self.logger.error(
                "Chat completion failed",
                model=model or self.default_model,
                error=str(e)
            )
            raise LLMError(f"Failed to create chat completion: {e}")

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        创建流式聊天完成请求

        Args:
            messages: 消息列表
            model: 模型名称 (可选)
            temperature: 生成温度 (可选)
            max_tokens: 最大令牌数 (可选)
            **kwargs: 其他OpenAI API参数

        Yields:
            流式响应的文本块
        """
        self._ensure_initialized()

        try:
            response = await self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        except Exception as e:
            self.logger.error("Stream chat completion failed", error=str(e))
            raise LLMError(f"Failed to stream chat completion: {e}")

    async def get_models(self) -> List[Dict[str, Any]]:
        """
        获取可用的模型列表

        Returns:
            模型信息列表
        """
        self._ensure_initialized()

        try:
            self.logger.info("Fetching available models")
            response = await self.client.models.list()

            models = []
            for model in response.data:
                models.append({
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by
                })

            self.logger.info(f"Found {len(models)} available models")
            return models

        except Exception as e:
            self.logger.error("Failed to fetch models", error=str(e))
            raise LLMError(f"Failed to fetch models: {e}")

    def create_system_message(self, content: str) -> Dict[str, str]:
        """创建系统消息"""
        return {"role": "system", "content": content}

    def create_user_message(self, content: str) -> Dict[str, str]:
        """创建用户消息"""
        return {"role": "user", "content": content}

    def create_assistant_message(self, content: str) -> Dict[str, str]:
        """创建助手消息"""
        return {"role": "assistant", "content": content}

    def create_multimodal_message(
        self,
        role: str,
        parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        创建多模态消息 (支持文本和图片)

        Args:
            role: 角色 (system/user/assistant)
            parts: 内容部分列表，每个部分为:
                - {"type": "text", "text": "文本内容"}
                - {"type": "image_url", "image_url": {"url": "图片URL或base64"}}

        Returns:
            多模态消息
        """
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({
                    "type": "text",
                    "text": part["text"]
                })
            elif part["type"] == "image_url":
                content.append({
                    "type": "image_url",
                    "image_url": part["image_url"]
                })

        return {
            "role": role,
            "content": content
        }