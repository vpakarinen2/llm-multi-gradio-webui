from __future__ import annotations

import logging
import torch

from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from typing import Any, Dict, List, Tuple
from functools import lru_cache
from app.config import config


logger = logging.getLogger("app.models.gemma3")


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


@lru_cache(maxsize=1)
def get_model_and_processor() -> Tuple[Gemma3nForConditionalGeneration, AutoProcessor]:
    """Load Gemma 3n model and processor."""
    torch_dtype = _get_torch_dtype(config.model.dtype)

    logger.info(
        "Initializing Gemma 3n backend (model_id=%s, device=%s, dtype=%s)",
        config.model.model_id,
        config.model.device,
        config.model.dtype,
    )

    model = Gemma3nForConditionalGeneration.from_pretrained(
        config.model.model_id,
        torch_dtype=torch_dtype,
    )

    model = model.to(device=config.model.device)
    model = model.eval()

    processor = AutoProcessor.from_pretrained(config.model.model_id)
    logger.info("Gemma 3n model and processor loaded")
    return model, processor


def _build_chat_messages(
    history: List[Tuple[str, str]],
    user_input: str,
    system_prompt: str = "You are a helpful assistant.",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        }
    ]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                    ],
                }
            )
        if assistant_msg:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_msg},
                    ],
                }
            )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
            ],
        }
    )

    return messages


def generate_chat_response(
    history: List[Tuple[str, str]],
    user_input: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[Tuple[str, str]], str]:
    """Run chat with Gemma 3 using the chat template."""
    model, processor = get_model_and_processor()
    messages = _build_chat_messages(history, user_input)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
        )

    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)

    new_history = history + [(user_input, decoded)]
    return new_history, decoded


def generate_image_caption(
    image: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate image caption given a PIL image."""
    model, processor = get_model_and_processor()

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that describes images in detail.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded
