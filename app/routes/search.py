from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import rag_service
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    model: str = "mistral:7b"     # 模型名稱 (不論 local 或 cloud)
    mode: str = "tech"            # Prompt 模式
    custom_prompt: str = ""       # 自定義 Prompt
    provider: str = "local"       # 新增: local | openai | gemini | claude
    api_key: Optional[str] = None           # 新增: 雲端 API Key，僅 provider != local 時需要

@router.post("/search")
def search(request: SearchRequest):
    try:
        result = rag_service.search(
            query=request.query,
            k=request.k,
            model=request.model,
            provider=request.provider,
            api_key=request.api_key,
            prompt_mode=request.mode,
            custom_prompt=request.custom_prompt
        )
        return result
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
