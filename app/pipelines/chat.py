from __future__ import annotations

from app.models.gemma3 import generate_chat_response
from typing import List, Optional, Tuple


ChatHistory = List[Tuple[str, str]]


def chat_pipeline(
    user_input: str,
    history: Optional[ChatHistory],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[ChatHistory, ChatHistory, str]:
    """Wrapper for Gemma3 model."""
    if history is None:
        history = []

    user_input_stripped = user_input.strip()
    if not user_input_stripped:
        return history, history, user_input

    new_history, _answer = generate_chat_response(
        history=history,
        user_input=user_input_stripped,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return new_history, new_history, ""
