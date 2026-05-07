"""Cliente GPT-4o compartido por las cuatro arquitecturas."""

from __future__ import annotations

import time
import logging
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def chat(
    messages: list[dict[str, str]],
    system: str | None = None,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    **kwargs: Any,
) -> tuple[str, dict]:
    """
    Llama a la API de OpenAI y devuelve (respuesta_texto, usage_dict).

    Args:
        messages: Lista de mensajes en formato OpenAI.
        system: Prompt de sistema opcional (se antepone al listado).
        model: Identificador del modelo a usar.
        temperature: Temperatura de muestreo.
        max_tokens: Límite de tokens en la respuesta.

    Returns:
        Tupla (texto_respuesta, dict_de_uso) con tokens consumidos.
    """
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    t0 = time.perf_counter()
    response = get_client().chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    latency = time.perf_counter() - t0

    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_s": round(latency, 3),
        "model": response.model,
    }
    logger.debug("LLM call: %d tokens, %.2fs", usage["total_tokens"], latency)
    return text, usage


def simple_chat(prompt: str, system: str | None = None) -> str:
    """Wrapper simplificado para consultas de una sola vuelta."""
    text, _ = chat([{"role": "user", "content": prompt}], system=system)
    return text
