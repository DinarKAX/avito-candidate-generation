"""Шаг 1. Делим train на H/R/V и готовим тексты всех объявлений.

"Вселенная" объявлений = корпус бенчмарка + правильные ответы из V и R (их в корпусе
бенчмарка может не быть). Дальше корпус для каждого режима задаётся маской:
  бенчмарк - только benchmark_items
  V        - benchmark_items + ответы V
  R        - benchmark_items + ответы R
Так на валидации правильные ответы конкурируют с теми же ~189к объявлений что и на бенчмарке.

Запуск: python src/prepare.py, всё сохраняется в cache/
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import CACHE, ITEM_COLS, SEED, build_splits, load_items, params_useful_text, split_queries, tokenize


def tokenize_items(items: pd.DataFrame) -> pd.DataFrame:
    """Токены для заголовка, параметров и описания.
    Описание режу до 1500 символов, дальше обычно контакты, прайсы и простыни ключевых слов"""
    out = pd.DataFrame({"item_id": items["item_id"].values})
    out["t_title"] = [tokenize(t) for t in tqdm(items["item_title_raw"], desc="title")]
    out["t_params"] = [tokenize(params_useful_text(p)) for p in tqdm(items["item_infm_params_text"], desc="params")]
    out["t_desc"] = [tokenize(d[:1500] if isinstance(d, str) else "")
                     for d in tqdm(items["item_description_raw"], desc="desc")]
    return out


def main():
    t0 = time.time()
    tr = build_splits()
    tr.drop(columns=ITEM_COLS[1:]).to_parquet(os.path.join(CACHE, "train_split.parquet"))
    for split in ["val", "rank"]:
        q, truth = split_queries(tr, split)
        q.to_parquet(os.path.join(CACHE, f"{split}_queries.parquet"))
        with open(os.path.join(CACHE, f"{split}_truth.pkl"), "wb") as f:
            pickle.dump(truth, f)
        print(f"{split}: {len(q)} queries, {sum(map(len, truth.values()))} positives")

    items = load_items()
    bench_ids = set(items["item_id"])
    extra = tr[tr["split"].isin(["val", "rank"])].drop_duplicates("item_id")
    extra = extra[~extra["item_id"].isin(bench_ids)]
    universe = pd.concat([items, extra[items.columns]], ignore_index=True)
    universe["in_bench"] = np.arange(len(universe)) < len(items)
    # обязательно перемешать! без этого добавленные ответы V/R лежат в конце массива,
    # и при равных значениях признака сортировка их выдаёт по индексу. Ранкер выучил
    # "индекс > 189212 значит правильный ответ" и валидация была завышена (0.888 вместо 0.877).
    # Нашёл это когда dense_rank оказался самым важным признаком в прогоне вообще без эмбеддингов
    universe = universe.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # флаги для масок корпуса V и R
    pos = tr[tr["split"].isin(["val", "rank"])].groupby("item_id")["split"].agg(set)
    universe["pos_val"] = universe["item_id"].map(lambda i: "val" in pos.get(i, ()))
    universe["pos_rank"] = universe["item_id"].map(lambda i: "rank" in pos.get(i, ()))
    universe.to_parquet(os.path.join(CACHE, "universe.parquet"))
    print(f"universe: {len(universe)} items ({len(extra)} added to bench corpus)")

    tokenize_items(universe).to_pickle(os.path.join(CACHE, "item_tokens.pkl"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
