"""BM25, приоритет локации и top-k.

BM25 написал сам на scipy.sparse. Готовые библиотеки отдают сразу топ документов,
а мне нужен скор по всему корпусу, чтобы сначала сложить его с локацией
и эмбеддингами и только потом брать топ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Vocab:
    def __init__(self):
        self.idx: dict[str, int] = {}

    def ids(self, tokens, add=False):
        out = []
        for t in tokens:
            i = self.idx.get(t)
            if i is None and add:
                i = self.idx[t] = len(self.idx)
            if i is not None:
                out.append(i)
        return out


def _csr(token_lists, vocab: Vocab, add: bool) -> sp.csr_matrix:
    rows, cols = [], []
    for r, toks in enumerate(token_lists):
        ids = vocab.ids(toks, add=add)
        rows.extend([r] * len(ids))
        cols.extend(ids)
    m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(token_lists), max(len(vocab.idx), 1)))
    m.sum_duplicates()
    return m


class BM25Field:
    """BM25 по одному полю. Веса idf*tf считаются заранее,
    так что скор для пачки запросов это одно умножение разреженных матриц"""

    def __init__(self, token_lists, vocab: Vocab, k1=1.2, b=0.75):
        tf = _csr(token_lists, vocab, add=True).tocsc()
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocsr()
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean() if dl.mean() > 0 else 1.0
        norm = k1 * (1 - b + b * dl / avg)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.m = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.n_terms = tf.shape[1]

    def query_matrix(self, token_lists, vocab: Vocab) -> sp.csr_matrix:
        q = _csr(token_lists, vocab, add=False)
        q.data[:] = 1.0  # если слово повторилось в запросе, считаем один раз
        # словарь общий для всех полей и мог вырасти после построения этого поля
        if q.shape[1] < self.n_terms:
            q = sp.csr_matrix((q.data, q.indices, q.indptr), shape=(q.shape[0], self.n_terms))
        return q[:, : self.n_terms]

    def scores(self, qm: sp.csr_matrix) -> np.ndarray:
        """скоры (n_queries x n_docs)"""
        return (qm @ self.m.T).toarray()


def location_prior(train: pd.DataFrame, alpha: float = 1.0) -> dict:
    """P(локация объявления | локация поиска), считаем по истории.

    В train 83% выбранных объявлений в той же локации что и поиск. Остальные это соседние
    города, или поиск был по области/всей стране (621540, 107620, 107621 и т.д.).
    Частоты из истории покрывают оба случая, жёсткий фильтр по локации терял 25% ответов.
    """
    cnt = train.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
    tot = cnt.groupby("search_location_id")["n"].transform("sum")
    cnt["p"] = cnt["n"] / (tot + alpha)
    return {s: dict(zip(g["item_location_id"], g["p"])) for s, g in cnt.groupby("search_location_id")}


def loc_log_prior(search_locs, item_locs: np.ndarray, prior: dict, eps=1e-4) -> np.ndarray:
    """log(P + eps) для каждой пары запрос/документ"""
    uniq, inv = np.unique(item_locs, return_inverse=True)
    out = np.empty((len(search_locs), len(item_locs)), np.float32)
    for r, s in enumerate(search_locs):
        pr = prior.get(s, {})
        # если локацию поиска не видели в истории, хотя бы совпадение id считаем нормальным
        pvec = np.array([pr.get(l, 0.0) + (0.3 if l == s and not pr else 0.0) for l in uniq], np.float32)
        out[r] = np.log(pvec + eps)[inv]
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """индексы top-k в каждой строке, от лучшего к худшему"""
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)
