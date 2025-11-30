from __future__ import annotations

from app.models.gemma3 import generate_image_caption
from typing import Any, Optional


def caption_pipeline(
    image: Optional[Any],
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Wrapper for image captioning using Gemma 3n."""
    if image is None:
        return "Please upload an image first."

    prompt_stripped = prompt.strip() or "Describe this image in detail."

    caption = generate_image_caption(
        image=image,
        prompt=prompt_stripped,
        max_new_tokens=max_new_tokens,
    )

    return caption
