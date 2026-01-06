from __future__ import annotations

import logging

from app.stop_registry import clear_stop, get_stop_event
from app.models.gemma3 import generate_chat_response
from typing import List, Optional, Tuple


ChatHistory = List[Tuple[str, str]]

logger = logging.getLogger("app.ui.callbacks")


def chat_pipeline(
    user_input: str,
    history: Optional[ChatHistory],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token: Optional[str] = None,
) -> Tuple[ChatHistory, ChatHistory, str]:
    """Wrapper for Qwen3-VL model."""
    if history is None:
        history = []

    stop_event = None
    if stop_token:
        clear_stop(stop_token)
        stop_event = get_stop_event(stop_token)

    user_input_stripped = user_input.strip()
    if not user_input_stripped:
        return history, history, user_input

    logger.info(
        "chat_pipeline: start (len_history=%d, max_new_tokens=%d, temp=%.3f, top_p=%.3f)",
        len(history),
        max_new_tokens,
        temperature,
        top_p,
    )

    new_history, _answer = generate_chat_response(
        history=history,
        user_input=user_input_stripped,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_event=stop_event,
    )

    logger.info("chat_pipeline: done (len_history=%d)", len(new_history))
    return new_history, new_history, ""
