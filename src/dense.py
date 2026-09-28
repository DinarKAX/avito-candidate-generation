"""Эмбеддинги запросов и объявлений.

BM25 находит только совпадающие слова, а тут много синонимов и опечаток,
например "волитель на день" -> "Трезвый водитель", "шарики" -> "Шары воздушные".
Модели open-source с HuggingFace, всё считается локально.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from common import params_useful_text

# у меня python с python.org и без сертификатов скачивание с HF падало с SSL ошибкой,
# поэтому берём сертификаты из certifi
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

# e5 обучались с такими префиксами, без них качество хуже
PREFIX = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "deepvk/USER-base": ("query: ", "passage: "),
}


def item_text(items: pd.DataFrame, desc_chars: int = 300) -> list[str]:
    """Заголовок + вид/тип услуги + начало описания.
    Модель смотрит только ~128 токенов, так что важное ставим в начало"""
    out = []
    for t, p, d in zip(items["item_title_raw"], items["item_infm_params_text"], items["item_description_raw"]):
        p = re.sub(r"\s+", " ", params_useful_text(p))[:200]
        d = re.sub(r"\s+", " ", d if isinstance(d, str) else "")[:desc_chars]
        out.append(f"{t}. {p}. {d}")
    return out


def query_text(queries: pd.DataFrame) -> list[str]:
    """Запрос + фильтры, для коротких запросов фильтр сильно помогает понять что имелось ввиду"""
    out = []
    for q, p in zip(queries["search_query"], queries["search_infm_params_text"]):
        p = p if isinstance(p, str) else ""
        out.append(f"{q}. {p}" if p else q)
    return out


def load_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(name, device=device)
    m.max_seq_length = 128
    return m


def encode(model, texts: list[str], prefix: str = "", batch_size: int = 128) -> np.ndarray:
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=len(texts) > 5000).astype(np.float32)
