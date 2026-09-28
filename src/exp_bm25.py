"""Эксперимент: перебор весов BM25 по полям и веса локации на валидации.
Отсюда веса W_TITLE/W_PARAMS/W_DESC/W_LOC в features.py. Работает минут 6."""
import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, load_train, recall_at_k, tokenize
from retrieval import BM25Field, Vocab, loc_log_prior, location_prior, topk

vq = pd.read_parquet(f"{CACHE}/val_queries.parquet")
truth = pickle.load(open(f"{CACHE}/val_truth.pkl", "rb"))
split = pd.read_parquet(f"{CACHE}/train_split.parquet", columns=["split"])["split"].to_numpy()
tp = load_train(columns=["search_location_id", "item_location_id"])[split == "hist"]
uni = pd.read_parquet(f"{CACHE}/universe.parquet")
tok = pd.read_pickle(f"{CACHE}/item_tokens.pkl")
# корпус для валидации: бенчмарк + ответы V
keep = (uni["in_bench"] | uni["pos_val"]).to_numpy()
tok = tok[keep].reset_index(drop=True)
ids = tok["item_id"].to_numpy()
item_locs = uni.loc[keep, "item_location_id"].to_numpy()

t0 = time.time()
voc = Vocab()
fields = {f: BM25Field(tok[f].tolist(), voc) for f in ["t_title", "t_params", "t_desc"]}
print("index built", time.time() - t0)

qt = [tokenize(q) for q in vq["search_query"]]
prior = location_prior(tp)
qids = vq["query_id"].tolist()
slocs = vq["search_location_id"].tolist()

# все варианты весов считаем за один проход по батчам,
# иначе матрицы 2500 x 191к на каждое поле не влезали в память
configs = [(1.0, 0, 0, 0), (1.0, 0, 0, 1)] + list(itertools.product([1.0], [0, 0.3, 0.6], [0, 0.2, 0.4, 0.7], [0.5, 1, 2, 3]))
hits = np.zeros(len(configs))
B = 250
for i in range(0, len(qt), B):
    Sb = {f: fld.scores(fld.query_matrix(qt[i:i + B], voc)).astype(np.float32) for f, fld in fields.items()}
    Lb = loc_log_prior(slocs[i:i + B], item_locs, prior)
    for c, (wt, wp, wd, wl) in enumerate(configs):
        top = topk(wt * Sb["t_title"] + wp * Sb["t_params"] + wd * Sb["t_desc"] + wl * Lb, 50)
        for r in range(top.shape[0]):
            rel = truth[qids[i + r]]
            hits[c] += len(set(ids[top[r]]) & rel) / len(rel)
    print(f"batch {i} {time.time() - t0:.0f}s", flush=True)

res = pd.Series(hits / len(qt), index=[str(c) for c in configs]).sort_values(ascending=False)
print(res.head(15).to_string())
print("title only, no loc", res[str(configs[0])], "title + loc", res[str(configs[1])])
