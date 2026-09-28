"""Шаг 3. Пулы, ранкер, и либо оценка на валидации, либо answer.csv.

  python src/run.py val     - учим ранкер на R, считаем Recall@50 на V
  python src/run.py final   - учим ранкер на R, делаем answer.csv

Код для обоих режимов общий, разные только запросы, корпус и история.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, ROOT, SEED, TOPK, load_queries, load_train, recall_at_k
from features import FEATURES, Corpus, History, build_pool_features

EMB_NAME = os.environ.get("EMB", "")  # какие эмбеддинги брать из cache/, пусто - без них
N_RANK = int(os.environ.get("N_RANK", "20000"))


def load_all():
    uni = pd.read_parquet(f"{CACHE}/universe.parquet")
    tok = pd.read_pickle(f"{CACHE}/item_tokens.pkl")
    emb = np.load(f"{CACHE}/emb_items_{EMB_NAME}.npy") if EMB_NAME else None
    t0 = time.time()
    corpus = Corpus(uni, tok, emb)
    print(f"corpus indexed: {corpus.n} items, {time.time() - t0:.0f}s", flush=True)
    # для истории хватает этих колонок, описания не грузим (много памяти)
    train = load_train(columns=["search_query", "search_location_id", "item_location_id",
                                "item_latitude", "item_longitude", "item_microcat_id"])
    split = pd.read_parquet(f"{CACHE}/train_split.parquet", columns=["split"])["split"].to_numpy()
    return uni, corpus, train, split


def query_emb(name, split_name):
    if not EMB_NAME:
        return None
    return np.load(f"{CACHE}/emb_q_{EMB_NAME}_{split_name}.npy")


def pool_recall(df, truth, name):
    """сколько правильных ответов вообще попало в пул, выше этого ранкер не сделает"""
    hit = df[df["label"]].groupby("query_id").size()
    r = np.mean([hit.get(q, 0) / len(t) for q, t in truth.items()])
    print(f"[{name}] pool recall = {r:.4f}, avg pool size = {len(df) / len(truth):.0f}", flush=True)


def train_ranker(df_train):
    """CatBoost с обычным Logloss (подходит / не подходит).

    Для Recall@50 порядок внутри топа не важен, нужно просто отделить подходящие.
    Пробовал YetiRank, но он очень медленный (~1.6 сек на итерацию против ~0.1).
    Учим только на запросах где ответ есть в пуле, на остальных учить нечему.
    10% запросов откладываем для early stopping.
    """
    from catboost import CatBoostClassifier, Pool
    good = df_train.groupby("query_id")["label"].transform("any")
    d = df_train[good]
    qids = d["query_id"].unique()
    rng = np.random.default_rng(SEED)
    es_q = set(rng.choice(qids, size=max(len(qids) // 10, 1), replace=False))
    is_es = d["query_id"].isin(es_q).to_numpy()
    mk = lambda m: Pool(d.loc[m, FEATURES], label=d.loc[m, "label"].astype(int))
    model = CatBoostClassifier(loss_function="Logloss", iterations=3000, learning_rate=0.08, depth=8,
                               random_seed=SEED, thread_count=-1, verbose=500,
                               od_type="Iter", od_wait=200)
    model.fit(mk(~is_es), eval_set=mk(is_es), use_best_model=True)
    imp = pd.Series(model.get_feature_importance(type="PredictionValuesChange"), index=FEATURES)
    print(imp.sort_values(ascending=False).round(2).to_string())
    return model


def predict_top(model, df, corpus, k=TOPK):
    df = df.copy()
    df["score"] = model.predict_proba(df[FEATURES])[:, 1]
    df = df.sort_values(["query_id", "score"], ascending=[True, False])
    top = df.groupby("query_id").head(k)
    return top.groupby("query_id")["item_idx"].apply(lambda x: list(corpus.ids[x.to_numpy()])).to_dict()


def main(mode):
    uni, corpus, train, split = load_all()
    rank_q = pd.read_parquet(f"{CACHE}/rank_queries.parquet")
    rank_truth = pickle.load(open(f"{CACHE}/rank_truth.pkl", "rb"))
    if N_RANK < len(rank_q):
        rank_q = rank_q.sample(N_RANK, random_state=SEED).reset_index(drop=True)
        rank_truth = {q: rank_truth[q] for q in rank_q["query_id"]}

    # для ранкера и валидации история только H
    t0 = time.time()
    hist_h = History(train[split == "hist"], corpus)
    print(f"history H built {time.time() - t0:.0f}s", flush=True)

    in_bench = uni["in_bench"].to_numpy()
    rq_emb = query_emb(EMB_NAME, "rank")
    if rq_emb is not None and N_RANK < len(pd.read_parquet(f"{CACHE}/rank_queries.parquet")):
        allq = pd.read_parquet(f"{CACHE}/rank_queries.parquet")["query_id"]
        pos = pd.Series(np.arange(len(allq)), index=allq)
        rq_emb = rq_emb[pos[rank_q["query_id"]].to_numpy()]
    df_rank = build_pool_features(rank_q, corpus, hist_h, in_bench | uni["pos_rank"].to_numpy(),
                                  rq_emb, truth=rank_truth, verbose=False)
    pool_recall(df_rank, rank_truth, "R")
    model = train_ranker(df_rank)
    del df_rank

    if mode == "val":
        vq = pd.read_parquet(f"{CACHE}/val_queries.parquet")
        vt = pickle.load(open(f"{CACHE}/val_truth.pkl", "rb"))
        df_val = build_pool_features(vq, corpus, hist_h, in_bench | uni["pos_val"].to_numpy(),
                                     query_emb(EMB_NAME, "val"), truth=vt, verbose=False)
        pool_recall(df_val, vt, "V")
        # для сравнения: просто BM25 + локация без ранкера
        base = (df_val.sort_values(["query_id", "combo_rank"]).groupby("query_id").head(TOPK)
                .groupby("query_id")["item_idx"].apply(lambda x: list(corpus.ids[x.to_numpy()])).to_dict())
        print(f"[V] Recall@50 BM25+loc (no ranker) = {recall_at_k(base, vt):.4f}")
        pred = predict_top(model, df_val, corpus)
        print(f"[V] Recall@50 ranker = {recall_at_k(pred, vt):.4f}")
        df_val.to_parquet(f"{CACHE}/val_pool.parquet")
        with open(f"{CACHE}/val_pred.pkl", "wb") as f:
            pickle.dump(pred, f)
    else:
        # в финале история H + V (данных больше), корпус только бенчмарк
        hist_f = History(train[np.isin(split, ["hist", "val"])], corpus)
        bq = load_queries()
        df_b = build_pool_features(bq, corpus, hist_f, in_bench, query_emb(EMB_NAME, "bench"), verbose=False)
        pred = predict_top(model, df_b, corpus)
        write_answer(bq, pred, uni)


def write_answer(bq, pred, uni):
    """Сохраняем answer.csv и проверяем формат по требованиям из условия"""
    bench_ids = set(uni.loc[uni["in_bench"], "item_id"])
    rows = []
    for q in bq["query_id"]:
        top = list(dict.fromkeys(pred.get(q, [])))[:TOPK]  # убираем повторы и режем до 50
        assert all(i in bench_ids for i in top), "item_id вне корпуса бенчмарка"
        rows.append((q, " ".join(top)))
    ans = pd.DataFrame(rows, columns=["query_id", "answer"])
    assert ans["query_id"].is_unique and set(ans["query_id"]) == set(bq["query_id"])
    assert ans["answer"].str.split().str.len().max() <= TOPK
    assert ans["answer"].str.fullmatch(r"([0-9a-f]{16})( [0-9a-f]{16})*").all()
    path = os.path.join(ROOT, "answer.csv")
    ans.to_csv(path, index=False)
    print(f"saved {path}: {len(ans)} rows, avg {ans['answer'].str.split().str.len().mean():.1f} items")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
