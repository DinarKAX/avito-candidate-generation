"""Шаг 2. Дообучаем эмбеддинги и считаем векторы для корпуса и запросов.

  python src/train_dense.py train    - дообучение e5-small на парах из H, сохраняется в models/e5-avito
  python src/train_dense.py encode   - векторы для всех объявлений и запросов R/V/бенчмарка

Взял intfloat/multilingual-e5-small (MIT лицензия). Из того что пробовал без дообучения
она была лучшей (exp_dense_zeroshot.py), rubert-tiny-turbo немного хуже.

Учим только на H. Если учить и на R/V, модель видела бы эти запросы и признак dense
при обучении ранкера был бы лучше чем на бенчмарке.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, SEED, load_queries, load_train
from dense import PREFIX, encode, item_text, load_model, query_text

BASE = "intfloat/multilingual-e5-small"
OUT = os.path.join(ROOT, "models", "e5-avito")
N_PAIRS = int(os.environ.get("N_PAIRS", "100000"))
QP, DP = PREFIX[BASE]


def make_pairs() -> pd.DataFrame:
    tr = load_train(columns=["search_query", "search_infm_params_text", "item_id", "item_title_raw",
                             "item_infm_params_text", "item_description_raw"])
    split = pd.read_parquet(f"{CACHE}/train_split.parquet", columns=["split"])["split"].to_numpy()
    h = tr[split == "hist"].drop_duplicates(["search_query", "item_id"])
    # максимум 8 пар на запрос, иначе "маникюр" и "массаж" забивают весь датасет,
    # а в бенчмарке в основном редкие запросы
    h = h.sample(frac=1.0, random_state=SEED).groupby("search_query").head(8)
    h = h.sample(min(N_PAIRS, len(h)), random_state=SEED)
    return pd.DataFrame({"anchor": [QP + t for t in query_text(h)],
                         "positive": [DP + t for t in item_text(h)]})


def train():
    """Обучение с MultipleNegativesRankingLoss, цикл написан руками.

    Для запроса правильный ответ его объявление, остальные 127 в батче идут как негативы.
    Если в батче попались одинаковые тексты, их маскируем, иначе модель штрафуется
    за правильный ответ.

    Сначала делал через SentenceTransformerTrainer, но на маке (MPS) он шёл ~49 сек на шаг,
    видимо из-за сэмплера NO_DUPLICATES. Свой цикл в разы быстрее.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    pairs = make_pairs()
    print(f"pairs: {len(pairs)}", flush=True)
    model = load_model(BASE)
    dev = model.device
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    bs = 128
    n_steps = len(pairs) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / (0.05 * n_steps)) * max(0.0, (n_steps - st) / n_steps))
    scale = 20.0  # как в MNRL по умолчанию

    def emb(texts, max_len):
        model.max_seq_length = max_len
        f = model.tokenize(texts)
        f = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in f.items()}
        return F.normalize(model(f)["sentence_embedding"], dim=-1)

    model.train()
    t0 = time.time()
    a_all, p_all = pairs["anchor"].to_numpy(), pairs["positive"].to_numpy()
    for st in range(n_steps):
        a, p = list(a_all[st * bs:(st + 1) * bs]), list(p_all[st * bs:(st + 1) * bs])
        qa, dp = emb(a, 32), emb(p, 96)
        logits = scale * qa @ dp.T
        # маска для одинаковых текстов в батче (ложные негативы)
        same = torch.tensor([[a[i] == a[j] or p[i] == p[j] for j in range(bs)] for i in range(bs)], device=dev)
        eye = torch.eye(bs, dtype=torch.bool, device=dev)
        logits = logits.masked_fill(same & ~eye, -1e4)
        labels = torch.arange(bs, device=dev)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if st % 50 == 0:
            el = time.time() - t0
            print(f"step {st}/{n_steps} loss {loss.item():.3f} {el:.0f}s, eta {el / (st + 1) * (n_steps - st - 1) / 60:.0f} min",
                  flush=True)
    model.eval()
    model.max_seq_length = 128
    model.save(OUT)
    print(f"trained in {time.time() - t0:.0f}s, saved {OUT}", flush=True)


def encode_all(name=OUT, tag="e5ft"):
    model = load_model(name)
    uni = pd.read_parquet(f"{CACHE}/universe.parquet")
    for split, q in [("val", pd.read_parquet(f"{CACHE}/val_queries.parquet")),
                     ("rank", pd.read_parquet(f"{CACHE}/rank_queries.parquet")),
                     ("bench", load_queries())]:
        np.save(f"{CACHE}/emb_q_{tag}_{split}.npy", encode(model, query_text(q), QP))
    t0 = time.time()
    E = encode(model, item_text(uni), DP, batch_size=256)
    np.save(f"{CACHE}/emb_items_{tag}.npy", E.astype(np.float16).astype(np.float32))
    print(f"encoded {len(uni)} items in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"train": train, "encode": encode_all}[sys.argv[1]]()
