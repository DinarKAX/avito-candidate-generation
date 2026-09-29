"""Основная часть: пул кандидатов и признаки для ранкера.

Как работает для запроса:
  1. по всему корпусу считаем BM25 по полям, косинус эмбеддингов, приоритет локации
     и P(подкатегория | запрос)
  2. пул = объединение топов нескольких генераторов (BM25+локация, BM25 со слабой локацией,
     эмбеддинги+локация, BM25+подкатегория). Они ошибаются на разном, поэтому вместе
     покрывают больше чем любой по отдельности
  3. для пар запрос-кандидат считаем признаки, дальше ранкер выбирает топ-50

Считаем батчами запросов, иначе матрица запросы x корпус не влезает в память.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import tokenize
from retrieval import BM25Field, Vocab, topk

# параметры
# веса BM25 подбирал перебором на валидации (exp_bm25.py)
W_TITLE, W_PARAMS, W_DESC, W_LOC = 1.0, 0.6, 0.7, 3.0
POOL_BM25, POOL_DENSE, POOL_MC, POOL_WEAKLOC = 300, 200, 100, 200
# когда смотрел ошибки, 76% ответов которые не попали в пул были в другой локации,
# хотя текст совпадал нормально. Добавил генератор где локация весит меньше,
# полнота пула на V выросла с 0.923 до 0.940
W_LOC_WEAK = 1.0
LOC_EPS = 1e-4


def _to_float(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").astype(np.float32).to_numpy()


_VID_RE = re.compile(r"Вид услуги (.+?)(?= Место оказания услуг| Тип услуги|$)")


def item_vid(params: str) -> str:
    """достаём "Вид услуги" из параметров объявления"""
    m = _VID_RE.search(params) if isinstance(params, str) else None
    return m.group(1).strip() if m else ""


# корпус
class Corpus:
    """Индексы по всем объявлениям (бенчмарк + ответы V/R).
    Строим один раз, а какой корпус использовать задаём маской"""

    def __init__(self, uni: pd.DataFrame, tok: pd.DataFrame, emb: np.ndarray | None):
        assert (uni["item_id"].values == tok["item_id"].values).all()
        self.ids = uni["item_id"].to_numpy()
        self.n = len(uni)
        self.loc = uni["item_location_id"].to_numpy()
        self.mc = uni["item_microcat_id"].to_numpy()
        self.lat, self.lon = _to_float(uni["item_latitude"]), _to_float(uni["item_longitude"])
        self.rating = uni["item_rating"].to_numpy(np.float32)
        self.reviews = np.log1p(uni["item_rating_reviews_count"].fillna(0).to_numpy(np.float32))
        price = np.nan_to_num(_to_float(uni["item_price"]), nan=-1)
        self.price = np.where(price >= 0, np.log1p(np.clip(price, 0, 1e9)), -1).astype(np.float32)
        self.phone_hidden = uni["item_is_phone_hidden"].to_numpy(np.float32)
        self.msg_forbidden = uni["item_is_message_forbidden"].to_numpy(np.float32)
        self.desc_len = np.log1p(uni["item_description_raw"].fillna("").str.len().to_numpy(np.float32))
        self.title_len = tok["t_title"].str.len().to_numpy(np.float32)
        self.vid = uni["item_infm_params_text"].map(item_vid).to_numpy()
        self.vid_values = sorted({v for v in self.vid if v}, key=len, reverse=True)

        self.vocab = Vocab()
        self.title = BM25Field(tok["t_title"].tolist(), self.vocab)
        self.params = BM25Field(tok["t_params"].tolist(), self.vocab)
        self.desc = BM25Field(tok["t_desc"].tolist(), self.vocab)
        # есть ли слово в документе, для признаков покрытия
        self.title_bin = (self.title.m > 0).astype(np.float32).T.tocsr()
        # словарь общий и рос по ходу, поэтому у title и params матрицы уже чем у desc,
        # дополняем до одного размера
        V = self.desc.n_terms

        def pad(m):
            m = m.tocsr()
            return sp.csr_matrix((m.data, m.indices, m.indptr), shape=(m.shape[0], V))
        allb = (pad(self.title.m) > 0) + (pad(self.params.m) > 0) + (self.desc.m > 0)
        self.all_bin = (allb > 0).astype(np.float32).T.tocsr()
        self.params_bin = (pad(self.params.m) > 0).astype(np.float32).T.tocsr()
        self.emb = emb


# история
class History:
    """Статистики по истории (часть H).

    Сами пары запрос-объявление почти не повторяются (только 3.7% случаев), поэтому
    их не запоминаем. Берём более общие вещи: в какие локации ходят из данной локации
    и какие подкатегории выбирают по каким словам.
    """

    def __init__(self, hist: pd.DataFrame, corpus: Corpus):
        # локации
        cnt = hist.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        tot = cnt.groupby("search_location_id")["n"].transform("sum")
        cnt["p"] = cnt["n"] / (tot + 1.0)
        self.loc_prior = {s: dict(zip(g["item_location_id"], g["p"])) for s, g in cnt.groupby("search_location_id")}
        # как часто из этой локации выбирают объявления в другой (поиск по области/стране)
        same = (hist["search_location_id"] == hist["item_location_id"])
        self.loc_same_rate = same.groupby(hist["search_location_id"]).mean().to_dict()
        # центр локации поиска - медиана координат объявлений которые оттуда выбирали
        h = hist.assign(lat=_to_float(hist["item_latitude"]), lon=_to_float(hist["item_longitude"]))
        self.loc_center = h.groupby("search_location_id")[["lat", "lon"]].median().to_dict("index")
        # считаем по уникальным локациям, потом разворачиваем на весь корпус (так быстрее)
        self.uniq_loc, self.loc_inv = np.unique(corpus.loc, return_inverse=True)

        # подкатегории по словам и по целому тексту запроса
        self.mc_index = {m: i for i, m in enumerate(np.unique(corpus.mc))}
        self.item_mc_idx = np.array([self.mc_index[m] for m in corpus.mc])
        n_mc = len(self.mc_index)
        tok_mc: dict[str, Counter] = defaultdict(Counter)
        text_mc: dict[str, Counter] = defaultdict(Counter)
        for q, m in zip(hist["search_query"], hist["item_microcat_id"]):
            if m not in self.mc_index:
                continue
            text_mc[q][m] += 1
            for t in set(tokenize(q)):
                tok_mc[t][m] += 1
        self.n_mc = n_mc
        self.tok_mc = {t: self._dist(c) for t, c in tok_mc.items() if sum(c.values()) >= 2}
        self.tok_cnt = {t: sum(c.values()) for t, c in tok_mc.items()}
        self.text_mc = {q: (self._dist(c), sum(c.values())) for q, c in text_mc.items()}
        self.text_cnt = hist["search_query"].value_counts().to_dict()

    def _dist(self, c: Counter) -> np.ndarray:
        v = np.zeros(self.n_mc, np.float32)
        for m, k in c.items():
            v[self.mc_index[m]] = k
        return v / v.sum()

    def mc_dist(self, query: str, qtok: list[str]) -> np.ndarray:
        """P(подкатегория | запрос). Если такой запрос уже был, берём его распределение,
        плюс смешиваем со средним по словам (слова с чётким распределением весят больше)"""
        acc, wsum = np.zeros(self.n_mc, np.float32), 0.0
        # sorted а не set: порядок обхода set зависит от хэш-сида питона, float суммируется
        # в разном порядке и результат плавает между запусками (answer.csv не воспроизводился)
        for t in sorted(set(qtok)):
            d = self.tok_mc.get(t)
            if d is None:
                continue
            w = np.log1p(self.tok_cnt[t]) * (d.max() ** 2)  # частые и однозначные слова весят больше
            acc += w * d
            wsum += w
        tok_part = acc / wsum if wsum > 0 else None
        exact = self.text_mc.get(query)
        if exact is not None:
            d, n = exact
            lam = n / (n + 3.0)
            return lam * d + (1 - lam) * (tok_part if tok_part is not None else d)
        if tok_part is not None:
            return tok_part
        return np.full(self.n_mc, 1.0 / self.n_mc, np.float32)

    def loc_logp(self, search_loc) -> np.ndarray:
        pr = self.loc_prior.get(search_loc, {})
        pv = np.array([pr.get(l, 0.0) for l in self.uniq_loc], np.float32)
        if not pr:  # локацию не видели, тогда хоть совпадение id
            pv[self.uniq_loc == search_loc] = 0.5
        return np.log(pv + LOC_EPS)[self.loc_inv]


# запросы
def parse_search_params(params: str, vid_values: list[str]):
    """Разбираем фильтры поиска: вид услуги, минимальный рейтинг и просто слова фильтра"""
    params = params if isinstance(params, str) else ""
    vid = ""
    pos = params.find("Вид услуги ")
    if pos >= 0:
        rest = params[pos + len("Вид услуги "):]
        vid = next((v for v in vid_values if rest.startswith(v)), "")
    m = re.search(r"Рейтинг пользователя (\d)", params)
    rating_min = float(m.group(1)) if m else 0.0
    stop = {"вид", "услуга", "тип", "рейтинг", "пользователь", "звезда", "и", "выше", "для"}
    ptok = [t for t in tokenize(params) if t not in stop and not t.isdigit()]
    return vid, rating_min, ptok


FEATURES = [
    # текстовая релевантность
    "bm_title", "bm_params", "bm_desc", "bm_combo", "bm_title_rel", "bm_combo_rel",
    "cov_title", "cov_all", "cov_params_filter", "dense", "dense_rel", "dense_rank", "combo_rank",
    # локация
    "loc_logp", "same_loc", "dist_km", "loc_same_rate",
    # рубрикатор
    "mc_logp", "mc_rank", "vid_match", "rating_ok",
    # объявление
    "rating", "reviews", "price", "phone_hidden", "msg_forbidden", "desc_len", "title_len",
    # запрос
    "q_ntok", "q_has_params", "q_text_seen", "q_mc_entropy",
]


_RNG = np.random.default_rng(0)


def _rank_desc(x: np.ndarray) -> np.ndarray:
    """Ранг в строке, 0 это лучший.
    Одинаковые значения разбиваем случайно, а не по индексу (см. утечку в prepare.py)"""
    tie = _RNG.random(x.shape)
    order = np.lexsort((tie, -x), axis=1)
    return np.argsort(order, axis=1).astype(np.float32)


def build_pool_features(queries: pd.DataFrame, corpus: Corpus, hist: History, mask: np.ndarray,
                        q_emb: np.ndarray | None, batch: int = 200, truth: dict | None = None,
                        verbose: bool = True) -> pd.DataFrame:
    """Пул + признаки для всех запросов, если передать truth то ещё и label.
    mask - какие объявления входят в корпус
    """
    out = []
    qids = queries["query_id"].tolist()
    texts = queries["search_query"].tolist()
    qtoks = [tokenize(t) for t in texts]
    params = queries["search_infm_params_text"].fillna("").tolist()
    slocs = queries["search_location_id"].tolist()
    neg_inf = np.where(mask, 0.0, -1e9).astype(np.float32)

    for s in range(0, len(qids), batch):
        e = min(s + batch, len(qids))
        B = e - s
        qt = qtoks[s:e]
        qm = corpus.title.query_matrix(qt, corpus.vocab)
        bt = corpus.title.scores(qm).astype(np.float32)
        bp = corpus.params.scores(corpus.params.query_matrix(qt, corpus.vocab)).astype(np.float32)
        bd = corpus.desc.scores(corpus.desc.query_matrix(qt, corpus.vocab)).astype(np.float32)
        L = np.vstack([hist.loc_logp(l) for l in slocs[s:e]])
        mcd = np.vstack([hist.mc_dist(texts[i], qtoks[i]) for i in range(s, e)])
        MC = np.log(mcd[:, hist.item_mc_idx] + 1e-4)

        combo = W_TITLE * bt + W_PARAMS * bp + W_DESC * bd
        g_bm = combo + W_LOC * L + neg_inf
        cand = [topk(g_bm, POOL_BM25)]
        cand.append(topk(combo + W_LOC * L + 1.0 * MC + neg_inf, POOL_MC))
        cand.append(topk(combo + W_LOC_WEAK * L + neg_inf, POOL_WEAKLOC))
        if q_emb is not None and corpus.emb is not None:
            D = q_emb[s:e] @ corpus.emb.T
            # косинусы отличаются на ~0.1, поэтому умножаем на 20 и локации даём меньше веса
            cand.append(topk(D * 20 + W_LOC * 0.5 * L + neg_inf, POOL_DENSE))
        else:
            D = np.zeros_like(bt)
        pool = np.concatenate(cand, axis=1)
        pool = np.sort(pool, axis=1)
        # генераторы пересекаются, убираем дубли
        dup = np.zeros_like(pool, dtype=bool)
        dup[:, 1:] = pool[:, 1:] == pool[:, :-1]
        K = pool.shape[1]

        # какая доля слов запроса есть в объявлении
        qb = corpus.title.query_matrix(qt, corpus.vocab)
        qb = sp.csr_matrix((qb.data, qb.indices, qb.indptr), shape=(B, corpus.title_bin.shape[0]))
        qlen = np.maximum(np.array([len(set(x)) for x in qt], np.float32), 1)[:, None]
        cov_t = (qb @ corpus.title_bin).toarray() / qlen
        qb2 = corpus.desc.query_matrix(qt, corpus.vocab)
        cov_a = (qb2 @ corpus.all_bin).toarray() / qlen
        # сколько слов из фильтра есть в параметрах объявления
        parsed = [parse_search_params(p, corpus.vid_values) for p in params[s:e]]
        fm = corpus.desc.query_matrix([p[2] for p in parsed], corpus.vocab)
        flen = np.array([len(set(p[2])) for p in parsed], np.float32)[:, None]
        cov_f = np.where(flen > 0, (fm @ corpus.all_bin).toarray() / np.maximum(flen, 1), -1)

        g = lambda M: np.take_along_axis(M, pool, 1)
        f = {}
        f["bm_title"], f["bm_params"], f["bm_desc"] = g(bt), g(bp), g(bd)
        f["bm_combo"] = g(combo)
        f["bm_title_rel"] = f["bm_title"] / (bt.max(1, keepdims=True) + 1e-6)
        f["bm_combo_rel"] = f["bm_combo"] / (combo.max(1, keepdims=True) + 1e-6)
        f["cov_title"], f["cov_all"], f["cov_params_filter"] = g(cov_t), g(cov_a), g(cov_f)
        f["dense"] = g(D)
        f["dense_rel"] = f["dense"] - D.max(1, keepdims=True)
        f["loc_logp"] = g(L)
        f["mc_logp"] = g(MC)
        f["combo_rank"] = _rank_desc(np.where(dup, -1e9, g(g_bm)))
        f["dense_rank"] = _rank_desc(np.where(dup, -1e9, f["dense"]))
        f["mc_rank"] = _rank_desc(np.where(dup, -1e9, f["mc_logp"]))

        sl = np.array(slocs[s:e])[:, None]
        f["same_loc"] = (corpus.loc[pool] == sl).astype(np.float32)
        centers = [hist.loc_center.get(l, {"lat": np.nan, "lon": np.nan}) for l in slocs[s:e]]
        clat = np.array([c["lat"] for c in centers], np.float32)[:, None]
        clon = np.array([c["lon"] for c in centers], np.float32)[:, None]
        dlat = corpus.lat[pool] - clat
        dlon = (corpus.lon[pool] - clon) * np.cos(np.radians(clat))
        f["dist_km"] = np.nan_to_num(np.sqrt(dlat ** 2 + dlon ** 2) * 111.0, nan=-1)
        f["loc_same_rate"] = np.repeat(np.array([hist.loc_same_rate.get(l, -1) for l in slocs[s:e]],
                                                np.float32)[:, None], K, 1)
        qvid = np.array([p[0] for p in parsed], dtype=object)[:, None]
        f["vid_match"] = np.where(qvid == "", -1, (corpus.vid[pool] == qvid).astype(np.float32)).astype(np.float32)
        rmin = np.array([p[1] for p in parsed], np.float32)[:, None]
        f["rating_ok"] = np.where(rmin > 0, (np.nan_to_num(corpus.rating[pool]) >= rmin), -1).astype(np.float32)

        for name, arr in [("rating", corpus.rating), ("reviews", corpus.reviews), ("price", corpus.price),
                          ("phone_hidden", corpus.phone_hidden), ("msg_forbidden", corpus.msg_forbidden),
                          ("desc_len", corpus.desc_len), ("title_len", corpus.title_len)]:
            f[name] = arr[pool]
        rep = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], K, 1)
        f["q_ntok"] = rep([len(x) for x in qt])
        f["q_has_params"] = rep([p != "" for p in params[s:e]])
        f["q_text_seen"] = rep([np.log1p(hist.text_cnt.get(t, 0)) for t in texts[s:e]])
        f["q_mc_entropy"] = rep(-(mcd * np.log(mcd + 1e-9)).sum(1))

        keep = ~dup
        df = pd.DataFrame({k: f[k][keep] for k in FEATURES})
        df.insert(0, "item_idx", pool[keep].astype(np.int32))
        df.insert(0, "query_id", np.repeat(np.array(qids[s:e], dtype=object)[:, None], K, 1)[keep])
        if truth is not None:
            ids = corpus.ids[df["item_idx"].to_numpy()]
            df["label"] = [i in truth[q] for q, i in zip(df["query_id"], ids)]
        out.append(df)
        if verbose:
            print(f"  pool batch {s}-{e}: avg pool {keep.sum() / B:.0f}", flush=True)
    res = pd.concat(out, ignore_index=True)
    for c in FEATURES:
        res[c] = res[c].astype(np.float32)
    return res
