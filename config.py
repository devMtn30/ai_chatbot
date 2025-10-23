from typing import Optional

from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
    sg_base_url: str = os.getenv("SG_BASE_URL", "http://10.0.2.62:8084").rstrip("/")
    default_email_domain: str = os.getenv("DEFAULT_EMAIL_DOMAIN", "giantstep.co.kr")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-5-nano-2025-08-07")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))

settings = Settings()
