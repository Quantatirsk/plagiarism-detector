"""
LLM API endpoints - 支持聊天完成和流式响应
"""
import json
import time as import_time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.deps import get_llm_service
from backend.core.errors import LLMError, create_http_exception
from backend.services.llm_service import LLMService

router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])
logger = structlog.get_logger()


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: system/user/assistant")
    content: str | list[dict[str, Any]] = Field(..., description="消息内容 (文本或多模态)")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求模型"""
    model: str | None = Field(None, description="模型名称")
    messages: list[ChatMessage] = Field(..., description="消息列表")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int | None = Field(None, ge=1, description="最大令牌数")
    stream: bool = Field(False, description="是否流式响应")


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: list[ModelInfo]


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    创建聊天完成

    支持标准和流式响应两种模式
    """
    try:
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        if request.stream:
            # 流式响应
            async def generate():
                try:
                    async for chunk in llm_service.stream_chat_completion(
                        messages=messages,
                        model=request.model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    ):
                        # 构造SSE格式的响应
                        data = {
                            "id": "chatcmpl-stream",
                            "object": "chat.completion.chunk",
                            "created": int(import_time.time()),
                            "model": request.model or llm_service.default_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                    # 发送结束标记
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error("Stream generation error", error=str(e))
                    error_data = {"error": {"message": str(e), "type": "stream_error"}}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # 禁用nginx缓冲
                }
            )
        else:
            # 非流式响应
            response = await llm_service.chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # 转换为标准格式
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            }

    except LLMError as e:
        logger.error("LLM error", error=str(e))
        raise create_http_exception(e) from e
    except Exception as e:
        logger.error("Unexpected error in chat completion", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        ) from e


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    获取可用的模型列表
    """
    try:
        models = await llm_service.get_models()
        return ModelsResponse(
            object="list",
            data=[ModelInfo(**model) for model in models]
        )
    except LLMError as e:
        logger.error("Failed to list models", error=str(e))
        raise create_http_exception(e) from e
    except Exception as e:
        logger.error("Unexpected error listing models", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to list models", "message": str(e)}
        ) from e
