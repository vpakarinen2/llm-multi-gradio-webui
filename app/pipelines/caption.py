from __future__ import annotations

import logging

from app.models.gemma3 import generate_image_caption
from typing import Any, Optional


logger = logging.getLogger("app.ui.caption")


def caption_pipeline(
    image: Optional[Any],
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Wrapper for image captioning using Gemma 3n."""
    if image is None:
        return "Please upload an image first."

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
    )

    logger.info("caption_pipeline: done")
    return caption
