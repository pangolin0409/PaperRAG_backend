import ollama
from typing import List, Dict, Any
import openai
import google.generativeai as genai
import anthropic
from app.utils.logger import get_logger

logger = get_logger(__name__)

def get_available_models() -> List[Dict[str, Any]]:
    """Gets a list of available models from Ollama."""
    try:
        return ollama.list()['models']
    except Exception as e:
        logger.error(f"Error getting available models from Ollama: {e}", exc_info=True)
        return []

def pull_model(model_name: str) -> None:
    """Pulls a model from Ollama."""
    try:
        ollama.pull(model_name)
        logger.info(f"Successfully pulled model {model_name}")
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}", exc_info=True)
        raise

def generate_completion_local(model: str, prompt: str) -> str:
    """Generates a completion using a specified local model."""
    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error generating completion with local model {model}: {e}", exc_info=True)
        raise

def generate_completion_openai(model: str, api_key: str, prompt: str) -> str:
    """Generates a completion using OpenAI."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating completion with OpenAI model {model}: {e}", exc_info=True)
        raise

def generate_completion_gemini(model: str, api_key: str, prompt: str) -> str:
    """Generates a completion using Google Gemini."""
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating completion with Gemini model {model}: {e}", exc_info=True)
        raise

def generate_completion_claude(model: str, api_key: str, prompt: str) -> str:
    """Generates a completion using Anthropic Claude."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error generating completion with Claude model {model}: {e}", exc_info=True)
        raise

# ----------------------
# Local (Ollama)
# ----------------------
def list_local_models():
    """從 Ollama 取得本地可用模型"""
    try:
        return ollama.list()['models']
    except Exception as e:
        logger.error(f"Failed to list local models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list local models: {e}")


# ----------------------
# OpenAI
# ----------------------
def list_openai_models(api_key: str):
    """從 OpenAI 取得可用模型"""
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.models.list()
        # 過濾 GPT 模型 (有些是 embedding / moderation)
        models = [m.id for m in resp.data if "gpt" in m.id]
        return [{"model": mid} for mid in models]
    except Exception as e:
        logger.error(f"Failed to list OpenAI models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list OpenAI models: {e}")


# ----------------------
# Google Gemini
# ----------------------
def list_gemini_models(api_key: str):
    """從 Google Gemini 取得可用模型"""
    try:
        genai.configure(api_key=api_key)
        resp = genai.list_models()
        # 只保留能做 text generation 的
        models = [m.name.split("/")[-1] for m in resp if "generateContent" in m.supported_generation_methods]
        return [{"model": mid} for mid in models]
    except Exception as e:
        logger.error(f"Failed to list Gemini models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list Gemini models: {e}")


# ----------------------
# Anthropic Claude
# ----------------------
def list_anthropic_models(api_key: str):
    """從 Anthropic 取得可用模型"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.models.list()
        models = [m.id for m in resp.data]
        return [{"model": mid} for mid in models]
    except Exception as e:
        logger.error(f"Failed to list Anthropic models: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list Anthropic models: {e}")


# ----------------------
# 統一入口
# ----------------------
def list_models(provider: str, api_key: str = None):
    if provider == "local":
        return list_local_models()
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required")
        return list_openai_models(api_key)
    elif provider == "google":
        if not api_key:
            raise ValueError("Google Gemini API key required")
        return list_gemini_models(api_key)
    elif provider == "anthropic":
        if not api_key:
            raise ValueError("Anthropic API key required")
        return list_anthropic_models(api_key)
    else:
        raise ValueError(f"Invalid provider: {provider}")