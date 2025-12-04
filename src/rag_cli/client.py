from openai import OpenAI

from .auth import load_api_key

# Instância única do cliente OpenAI para o projeto
client = OpenAI(api_key=load_api_key())

try:
    # Em versões recentes existe BadRequestError
    from openai import BadRequestError  # type: ignore[attr-defined]
except Exception:  # fallback
    BadRequestError = Exception  # type: ignore[assignment]

__all__ = ["client", "BadRequestError"]
