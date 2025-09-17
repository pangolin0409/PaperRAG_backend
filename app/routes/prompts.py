from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

PROMPT_MODES = ["summary", "tech", "citation", "custom"]

class PromptSetRequest(BaseModel):
    mode: str
    custom_prompt: str = ""

@router.get("/prompts")
def get_prompts():
    return {"modes": PROMPT_MODES}

@router.post("/prompts/set")
def set_prompt(request: PromptSetRequest):
    # In a stateless API, this would typically be handled client-side.
    # This endpoint is for acknowledging the mode selection.
    if request.mode not in PROMPT_MODES:
        return {"status": "error", "message": "Invalid mode"}
    if request.mode == "custom" and not request.custom_prompt:
        return {"status": "error", "message": "Custom prompt is required for custom mode"}
    return {"status": "success", "mode": request.mode}
