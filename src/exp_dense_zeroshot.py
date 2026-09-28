"""Эксперимент: сравнение моделей эмбеддингов без дообучения.

Чтобы было быстрее, корпус урезан: ответы V + 40к случайных объявлений.
Считаю Recall@50 с учётом локации, как для BM25.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, recall_at_k
from dense import PREFIX, encode, item_text, load_model, query_text
from retrieval import loc_log_prior, location_prior, topk

uni = pd.read_parquet(f"{CACHE}/universe.parquet")
tr = pd.read_parquet(f"{CACHE}/train_split.parquet", columns=["search_location_id", "item_id", "split"])
vq = pd.read_parquet(f"{CACHE}/val_queries.parquet")
truth = pickle.load(open(f"{CACHE}/val_truth.pkl", "rb"))

rng = np.random.default_rng(0)
sub = uni[uni["pos_val"] | (uni["in_bench"] & (rng.random(len(uni)) < 40000 / len(uni)))].reset_index(drop=True)
from common import load_train
full = load_train(columns=["item_id", "item_location_id", "search_location_id"])
L = loc_log_prior(vq["search_location_id"].tolist(), sub["item_location_id"].to_numpy(),
                  location_prior(full[tr["split"].values == "hist"]))
ids = sub["item_id"].to_numpy()
it_txt, q_txt = item_text(sub), query_text(vq)

for name in sys.argv[1:]:
    t0 = time.time()
    m = load_model(name)
    qp, dp = PREFIX.get(name, ("", ""))
    E = encode(m, it_txt, dp)
    Q = encode(m, q_txt, qp)
    S = Q @ E.T
    for wl in [0, 0.02, 0.05, 0.1]:
        top = topk(S + wl * L, 50)
        pred = {q: list(ids[top[r]]) for r, q in enumerate(vq["query_id"])}
        print(f"{name} wl={wl}: R@50={recall_at_k(pred, truth):.4f}", flush=True)
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
