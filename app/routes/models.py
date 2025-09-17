from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from app.services import llm_service
from typing import Optional
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

class ModelDownloadRequest(BaseModel):
    model_name: str

@router.get("/models")
def get_models():
    return {"models": llm_service.get_available_models()}

@router.post("/models/download")
def download_model(request: ModelDownloadRequest):
    try:
        llm_service.pull_model(request.model_name)
        return {"status": "success", "message": f"Model {request.model_name} is being downloaded."}
    except Exception as e:
        logger.error(f"Error downloading model: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@router.get("/list_models")
def list_models(provider: str, authorization: Optional[str] = Header(None)):
    # fetch api key from header Bearer token
    api_key = None

    if authorization:
        try:
            scheme, token = authorization.split(" ")
            if scheme.lower() == "bearer":
                api_key = token
        except ValueError:
            logger.error("Invalid Authorization header format")
            raise HTTPException(status_code=400, detail="Invalid Authorization header format")

    if provider == "Local":
        return {"models": llm_service.list_local_models()}

    if provider == "OpenAI":
        if not api_key:
            logger.error("API key required for OpenAI")
            raise HTTPException(status_code=400, detail="API key required for OpenAI")
        return {"models": llm_service.list_openai_models(api_key)}

    if provider == "Google":
        if not api_key:
            logger.error("API key required for Google Gemini")
            raise HTTPException(status_code=400, detail="API key required for Google Gemini")
        return {"models": llm_service.list_gemini_models(api_key)}

    if provider == "Anthropic":
        if not api_key:
            logger.error("API key required for Anthropic")
            raise HTTPException(status_code=400, detail="API key required for Anthropic")
        return {"models": llm_service.list_anthropic_models(api_key)}

    logger.error(f"Invalid provider: {provider}")
    raise HTTPException(status_code=400, detail="Invalid provider")