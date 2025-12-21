from gensim.models import FastText
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from collections import defaultdict
from pathlib import Path
import json
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any

class _DVShim:
    """Mimic gensim.Doc2Vec .dv interface with a minimal .most_similar(vectors, topn)."""
    def __init__(self, outer: "FastTextDocVec"):
        self.outer = outer

    def most_similar(self, vectors: List[np.ndarray], topn: int = 10):
        if isinstance(vectors, (list, tuple)):
            if len(vectors) != 1:
                raise ValueError("DVShim expects a single vector in a list like [vector].")
            inferred = np.asarray(vectors[0], dtype=np.float32)
        else:
            inferred = np.asarray(vectors, dtype=np.float32)
        return self.outer.most_similar(inferred, topn=topn)

class FastTextDocVec:
    """
    Minimal Doc2Vec-like API over FastText:
      - fit(df, text_col, label_col) trains FastText and caches doc vectors (L2-normalized means of word vectors)
      - infer_vector(text) -> np.ndarray (L2-normalized)
      - most_similar(vec, topn) -> [(tag, cosine), ...]
      - predict_classes(text, topn, k_neighbors) -> neighbor-vote by label

    Compatibility for your Doc2Vec evaluation:
      - .dv is a shim exposing .most_similar([vec], topn)
      - infer_vector(text) available
    """
    def __init__(self, dim: int = 254, window: int = 5, min_count: int = 3,
                 epochs: int = 20, sg: int = 1, min_n: int = 3, max_n: int = 6):
        
        self.dim = dim # dim of the feature vectors
        self.window = window # context window size +/-
        self.min_count = min_count # 0 = not ignore any words
        self.epochs = epochs
        self.sg = sg # 1 = skip-gram for similarity with doc2vec
        self.min_n = min_n
        self.max_n = max_n

        self.model: Optional[FastText] = None
        self.docvecs: Optional[np.ndarray] = None   # (n_docs, dim), L2-normalized
        self.tags: Optional[list[str]] = None       # list of doc ids aligned with docvecs
        self.labels: Optional[np.ndarray] = None    # array[str] aligned with docvecs

        # doc2vec-compat shim
        self.dv = _DVShim(self)

        # metadata (filled in fit)
        self._text_col = None
        self._label_col = None

    @staticmethod
    def _tokens(s: str) -> list:
        return s.split() if isinstance(s, str) and s.strip() else []
    
    def _docvec(self, tokens: list) -> np.ndarray:
        vecs = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.mean(vecs, axis=0)
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def fit(self, df, text_col: str = "GOODS_DESCRIPTION", label_col: str = "HS04"):
        """
        Train FastText robustly (explicit build_vocab/train) and cache:
        - self.docvecs: L2-normalized mean of word vectors per document
        - self.labels:  label per document (as str)
        - self.tags:    row index (as str), doc2vec-like
        """
        self._text_col, self._label_col = text_col, label_col

        # 1) Tokenize
        sentences = [self._tokens(x) for x in df[text_col].astype(str).tolist()]

        # Defensive: drop totally empty docs (avoid zero-only corpora)
        nonempty = [i for i, toks in enumerate(sentences) if len(toks) > 0]
        if not nonempty:
            raise ValueError("All documents are empty after tokenization/preprocessing.")
        if len(nonempty) < len(sentences):
            # Filter df and sentences together to avoid misalignment
            df = df.iloc[nonempty].copy()
            sentences = [sentences[i] for i in nonempty]

        # 2) Build model
        import os
        # - workers: set to max(1, os.cpu_count()-1) for portability (avoid -1)
        workers = max(1, (os.cpu_count() or 2) - 1)
        # - min_count: enforce >=1
        min_count = max(1, self.min_count)

        self.model = FastText(
            vector_size=self.dim,
            window=self.window,
            min_count=min_count,
            sg=self.sg,
            min_n=self.min_n,
            max_n=self.max_n,
            workers=workers
        )
        self.model.build_vocab(corpus_iterable=sentences)
        self.model.train(
            corpus_iterable=sentences,
            total_examples=len(sentences),
            epochs=self.epochs
        )

        # 3) Cache doc vectors
        self.docvecs = np.vstack([self._docvec(toks) for toks in sentences]).astype(np.float32)
        self.tags   = list(df.index.astype(str))
        self.labels = df[label_col].astype(str).to_numpy()

        # Diagnostics: % of zero vectors
        norms = np.linalg.norm(self.docvecs, axis=1)
        zero_ratio = float((norms < 1e-9).mean())
        if zero_ratio > 0.25:
            print(f"[WARN] {zero_ratio:.1%} of doc vectors are ~zero; check preprocessing/min_count.")

    def infer_vector(self, text: str) -> np.ndarray:
        toks = self._tokens(text)
        return self._docvec(toks)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: (d,), b: (n,d) -> (n,)
        a = a.reshape(1, -1).astype(np.float32, copy=False)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        return (a @ b.T).ravel() / (a_norm.ravel() * b_norm.ravel())

    def most_similar(self, inferred: np.ndarray, topn: int = 10) -> List[Tuple[str, float]]:
        sims = self._cosine(inferred, self.docvecs)
        k = min(topn, len(sims))
        idx = np.argpartition(-sims, k-1)[:k]
        ranked = sorted(((self.tags[i], float(sims[i])) for i in idx), key=lambda x: x[1], reverse=True)
        return ranked[:topn]

    def predict_classes(self, text: str, topn: int = 5, k_neighbors: int = 50) -> List[Tuple[str, float]]:
        inferred = self.infer_vector(text)
        sims = self._cosine(inferred, self.docvecs)
        k = min(k_neighbors, sims.shape[0])
        nn_idx = np.argpartition(-sims, k-1)[:k]
        agg = defaultdict(float)
        for i in nn_idx:
            agg[self.labels[i]] += float(sims[i])
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        return ranked[:topn]
    
    def save(self, directory: Union[str, Path]) -> None:
        """
        Save model + cached matrices and metadata to a folder:
           - fasttext.model (gensim)
           - docvecs.npy, labels.npy, tags.npy
           - meta.json  (dims, training params, columns)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        assert self.model is not None, "Model is empty. Train before saving."
        assert self.docvecs is not None and self.labels is not None and self.tags is not None, \
            "Missing cached arrays. Train before saving."

        self.model.save(str(directory / "fasttext.model"))
        np.save(directory / "docvecs.npy", self.docvecs)
        np.save(directory / "labels.npy", self.labels)
        np.save(directory / "tags.npy", np.array(self.tags, dtype=object))

        meta = {
            "dim": self.dim, "window": self.window, "min_count": self.min_count,
            "epochs": self.epochs, "sg": self.sg, "min_n": self.min_n, "max_n": self.max_n,
            "text_col": self._text_col, "label_col": self._label_col
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FastTextDocVec":
        """Load a previously saved wrapper."""
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        obj = cls(dim=meta["dim"], window=meta["window"], min_count=meta["min_count"],
                  epochs=meta["epochs"], sg=meta["sg"], min_n=meta["min_n"], max_n=meta["max_n"])
        obj._text_col = meta.get("text_col")
        obj._label_col = meta.get("label_col")

        obj.model = FastText.load(str(directory / "fasttext.model"))
        obj.docvecs = np.load(directory / "docvecs.npy")
        obj.labels = np.load(directory / "labels.npy", allow_pickle=True)
        obj.tags = np.load(directory / "tags.npy", allow_pickle=True).tolist()
        obj.dv = _DVShim(obj)   # reattach shim
        return obj
    

def _to_tokens(x: Union[str, Iterable[str]]) -> List[str]:
    """
    Accepts either a raw string or an iterable of tokens and returns a token list.
    """
    if isinstance(x, str):
        x = x.strip()
        return x.split() if x else []
    if isinstance(x, Iterable):
        return [str(t) for t in x]
    raise TypeError(f"input_text must be str or iterable of str, not {type(x)}")


def predict_ft(
    input_text: Union[str, Iterable[str]],
    model,                     # FastTextDocVec instance OR dict-like wrapper
    top_n: int = 5,
    *,
    epochs: int = 30,          # ignored (for signature parity)
    alpha: Optional[float] = None,      # ignored
    min_alpha: Optional[float] = None,  # ignored
) -> List[Tuple[str, float]]:
    """
    Infer a vector and return top_n (HS04_label, score) pairs.
    Uses model.predict_classes if available; otherwise maps most-similar tags to labels.
    """
    tokens = _to_tokens(input_text)
    if not tokens:
        return []

    text = " ".join(tokens)

    if hasattr(model, "predict_classes"):
        # many wrappers default k_neighbors internally (often 50)
        preds = model.predict_classes(text, topn=top_n)
        # Ensure (label, score) as (str, float)
        return [(str(lbl), float(scr)) for (lbl, scr) in preds]
    

def evaluate_df_ft(
    val_df: pd.DataFrame,
    model: FastTextDocVec = None,
    model_name: str = "",
    model_path: str = "",     # directory produced by FTDoc2VecLike.save()
    *,
    text_col: str = "GOODS_DESCRIPTION",
    target_col: str = "HS04",
    top_n: int = 5,
    epochs: int = 30,                 # accepted for signature parity (ignored)
    alpha: Optional[float] = None,    # ignored
    min_alpha: Optional[float] = None,# ignored
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Same API/outputs as before.
    Minimal fixes:
      - call predict_ft (not the model object)
      - ensure predict_ft returns HS04 labels, not tags
      - trim memory: evaluate on a slim copy; downcast scores to float32
    """
    print(f"Model: {model_name}")
    print(f"Text column: {text_col}")
    print(f"Target column: {target_col}")
    print(f"Top-N: {top_n}")

    if model is None:
        ft = FastTextDocVec.load(model_path)  # avoid shadowing 'model' name
        print(f"Model loaded from: {model_path}")
    else:
        ft = model

    # work on a slim copy to save RAM
    base_cols = [c for c in (text_col, target_col) if c in val_df.columns]
    df_ = val_df[base_cols].copy()

    if target_col in df_.columns:
        df_[target_col] = df_[target_col].astype(str)

    preds_tags = [[] for _ in range(top_n)]
    preds_scores = [[] for _ in range(top_n)]

    iterator = df_.itertuples(index=False, name=None)
    if show_progress:
        iterator = tqdm(iterator, total=df_.shape[0])

    cols = list(df_.columns)
    text_idx = cols.index(text_col)

    for row in iterator:
        text_val = row[text_idx]
        try:
            sims = predict_ft(text_val, ft, top_n=top_n) 
        except Exception:
            sims = []

        if len(sims) < top_n:
            sims = sims + [("", float("nan"))] * (top_n - len(sims))

        for k in range(top_n):
            tag, score = sims[k]
            preds_tags[k].append("" if tag is None else str(tag))
            # downcast to float32 to cut memory
            try:
                preds_scores[k].append(np.float32(score))
            except Exception:
                preds_scores[k].append(np.float32(np.nan))

    del ft

    # attach predictions (keep dtypes lean)
    for k in range(top_n):
        df_[f"top_{k+1}"] = pd.Series(preds_tags[k], index=df_.index, dtype="string")
        df_[f"top_{k+1}_SCORE"] = pd.Series(preds_scores[k], index=df_.index, dtype="float32")

    # display(df_)

    metrics: Dict[str, float] = {}
    if target_col in df_.columns:
        y_true = df_[target_col].astype(str)
        hit_cum = None
        for k in range(1, top_n + 1):
            eq_k = y_true.eq(df_[f"top_{k}"].astype(str))
            hit_cum = eq_k if hit_cum is None else (hit_cum | eq_k)
            acc_k = float(hit_cum.mean())
            metrics[f"top_{k}_acc"] = round(acc_k, 6)

        total = df_.shape[0]
        print(f"Total samples: {total}")
        for k in range(1, top_n + 1):
            correct = int(metrics[f"top_{k}_acc"] * total)
            print(f"Top-{k} Accuracy: {metrics[f'top_{k}_acc']:.4f} ({correct}/{total})")
    else:
        print("(No target column present; metrics skipped.)")

    return df_, metrics
