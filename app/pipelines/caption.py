from __future__ import annotations

import logging

from app.stop_registry import clear_stop, get_stop_event
from app.models.gemma3 import generate_image_caption
from typing import Any, Optional


logger = logging.getLogger("app.ui.caption")


def caption_pipeline(
    image: Optional[Any],
    prompt: str,
    max_new_tokens: int,
    stop_token: Optional[str] = None,
) -> str:
    """Wrapper for image captioning using Qwen3-VL."""
    if image is None:
        return "Please upload an image first."

    stop_event = None
    if stop_token:
        clear_stop(stop_token)
        stop_event = get_stop_event(stop_token)

    prompt_stripped = prompt.strip() or "Describe this image in detail."

    logger.info(
        "caption_pipeline: start (has_image=%s, max_new_tokens=%d)",
        image is not None,
        max_new_tokens,
    )

    caption = generate_image_caption(
        image=image,
        prompt=prompt_stripped,
        max_new_tokens=max_new_tokens,
        stop_event=stop_event,
    )

    logger.info("caption_pipeline: done")
    return caption
