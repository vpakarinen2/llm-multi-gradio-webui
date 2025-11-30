from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    model_id: str = "google/gemma-3n-E2B-it"
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)


config = AppConfig()
