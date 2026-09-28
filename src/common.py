"""Общие штуки: загрузка данных, обработка текста, сплиты для валидации и метрика.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = ROOT
CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)

SEED = 42
TOPK = 50

QUERY_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
              "search_infm_params_text", "search_category"]
ITEM_COLS = ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
             "item_category_id", "item_microcat_id", "item_price", "item_rating",
             "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
             "item_is_phone_hidden", "item_is_message_forbidden"]


#  загрузка
def load_train(columns=None) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, "train.parquet"), columns=columns)


def load_items() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, "benchmark_items.parquet"))


def load_queries() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, "benchmark_queries.parquet"))


#  текст
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")


def normalize(text: str) -> str:
    """lower + ё на е, т.к. кто-то пишет ё, кто-то нет"""
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е")


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=2_000_000)
def lemma(word: str) -> str:
    """Начальная форма слова через pymorphy3.
    Кэш обязателен, уникальных слов немного, а вхождений очень много. Без кэша лемматизация корпуса идёт очень долго"""
    if len(word) < 3 or word.isdigit() or not re.search("[а-я]", word):
        return word
    return _morph().parse(word)[0].normal_form.replace("ё", "е")


def tokenize(text: str) -> list[str]:
    """Разбиваем на слова и лемматизируем, иначе "видеодомофонов" в запросе
    не найдёт "видеодомофон" в объявлении"""
    return [lemma(t) for t in _TOKEN_RE.findall(normalize(text))]


_PARAM_KEYS = ["Вид услуги", "Тип услуги автосервиса", "Тип услуги", "Чем вы занимаетесь",
               "Специальность или сфера", "Направление", "Название услуги", "Услуга ",
               "Предмет или специальность", "Техника", "Груз", "Кузов", "Специализация"]


def params_useful_text(params: str) -> str:
    """Чистим параметры объявления, оставляем то что про саму услугу (вид, тип, названия).

    Параметры идут сплошной строкой "ключ значение ключ значение" без разделителей,
    поэтому просто вырезаем регулярками мусорные куски: адрес, расписание, цены.
    Без этого слова типа "понедельник" или "стоимость" матчились с запросами.
    """
    if not isinstance(params, str) or not params:
        return ""
    # адрес: от "Место оказания услуг" до следующего ключа
    text = re.sub(r"Место оказания услуг .*?(?=(Тип|Вид|Где|Начальная|Опыт|Как|Чем|Рабоч|Время|$))",
                  " ", params)
    # расписания, время, цены, дни недели
    text = re.sub(r"(Время для связи|График работы|Время работы)[^А-ЯЁ]*", " ", text)
    text = re.sub(r"Рабочие дни \w+", " ", text)
    text = re.sub(r"(Стоимость|Продолжительность|Выполняю заказы от|Опыт работы)\s*[\d\s\xa0чмин.–-]*",
                  " ", text)
    text = re.sub(r"(Тип стоимости за \w+|Начальная цена|Признак [\w ]+? \d+)", " ", text)
    return text


# метрика
def recall_at_k(pred: dict, truth: dict, k: int = TOPK) -> float:
    """Recall@K как в условии задачи, считаем по каждому запросу и усредняем"""
    scores = []
    for qid, rel in truth.items():
        top = set(pred.get(qid, [])[:k])
        scores.append(len(top & rel) / len(rel))
    return float(np.mean(scores))


def recall_per_query(pred: dict, truth: dict, k: int = TOPK) -> pd.Series:
    return pd.Series({qid: len(set(pred.get(qid, [])[:k]) & rel) / len(rel)
                      for qid, rel in truth.items()})


# сплиты
EVENT_COLS = ["search_query", "search_location_id", "search_infm_params_text",
              "search_is_delivery_search", "search_category"]


def build_splits(n_val: int = 2500, n_rank: int = 20000, seed: int = SEED):
    """Делим train на 3 части по событиям.

    Событие это (запрос, локация, фильтры, доставка, категория), объявления которые
    выбрали в этом событии считаем правильными ответами.

    В EDA заметил что запросы бенчмарка похоже выбирали по уникальным текстам:
    в train встречается только 37% из них. Если для валидации брать случайные события,
    то текст повторяется в 87% случаев и валидация получается слишком оптимистичной.
    Поэтому беру уникальные тексты и по одному событию на каждый, тогда повтор ~39%,
    почти как в бенчмарке.

      V (val)  - 2500 запросов, локальная валидация
      R (rank) - 20000 запросов, на них учится ранкер
      H (hist) - всё остальное, по нему считаются статистики и учатся эмбеддинги

    Признаки для V и R считаются только по H, чтобы не было утечки.
    """
    tr = load_train()
    tr["event"] = tr.groupby(EVENT_COLS, sort=False).ngroup()
    rng = np.random.default_rng(seed)
    texts = tr["search_query"].drop_duplicates().to_numpy()
    chosen = rng.choice(texts, size=n_val + n_rank, replace=False)
    val_texts, rank_texts = set(chosen[:n_val]), set(chosen[n_val:])

    ev = tr[["search_query", "event"]].drop_duplicates()
    ev = ev[ev["search_query"].isin(val_texts | rank_texts)]
    ev = ev.sample(frac=1.0, random_state=seed).drop_duplicates("search_query")
    val_ev = set(ev.loc[ev["search_query"].isin(val_texts), "event"])
    rank_ev = set(ev.loc[ev["search_query"].isin(rank_texts), "event"])

    tr["split"] = np.where(tr["event"].isin(val_ev), "val",
                           np.where(tr["event"].isin(rank_ev), "rank", "hist"))
    return tr


def split_queries(tr: pd.DataFrame, split: str):
    """Запросы сплита в том же формате что benchmark_queries + словарь ответов"""
    rows = tr[tr["split"] == split]
    q = rows.drop_duplicates("event")[["event"] + EVENT_COLS].rename(columns={"event": "query_id"})
    q["query_id"] = split[0] + q["query_id"].astype(str)
    truth = (rows.assign(query_id=split[0] + rows["event"].astype(str))
             .groupby("query_id")["item_id"].apply(set).to_dict())
    return q.reset_index(drop=True), truth
