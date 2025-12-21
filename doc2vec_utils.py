from gensim.models import Doc2Vec
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any

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


def predict_d2v(
    input_text: Union[str, Iterable[str]],
    model: Doc2Vec,
    top_n: int = 5,
    *,
    epochs: int = 30,
    alpha: Optional[float] = None,
    min_alpha: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """
    Infer a vector for input_text and return top_n most similar documents (tag, score).
    Works with gensim>=4 (uses model.dv).
    """
    tokens = _to_tokens(input_text)
    # Safe guard: empty token list -> return empty results
    if not tokens:
        return []

    # infer_vector in gensim 4: epochs replaces steps; alpha/min_alpha optional
    inferred = model.infer_vector(tokens, epochs=epochs, alpha=alpha, min_alpha=min_alpha)

    # dv is the doc vectors (alias to docvecs in older gensim)
    dv = getattr(model, "dv", getattr(model, "docvecs", None))
    if dv is None:
        raise AttributeError("Doc2Vec model missing .dv/.docvecs")

    sims = dv.most_similar([inferred], topn=top_n)
    # Round ONLY for display; keep numeric float
    return [(str(tag), float(score)) for tag, score in sims]


def evaluate_df_d2v(
    val_df: pd.DataFrame,
    model: Doc2Vec = None,
    model_name: str = "",
    model_path: str = "",
    *,
    text_col: str = "GOODS_DESCRIPTION",
    target_col: str = "HS04",
    top_n: int = 5,
    epochs: int = 30,
    alpha: Optional[float] = None,
    min_alpha: Optional[float] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    For each row, run predict_d2v and create top_k prediction columns: top_1..top_k and top_1_SCORE..top_k_SCORE.
    Returns (scored_df, metrics) with top-k accuracies.
    """
    print(f"Model: {model_name}")
    print(f"Text column: {text_col}")
    print(f"Target column: {target_col}")
    print(f"Top-N: {top_n}")

    # Load model
    if model is None:
        model = Doc2Vec.load(model_path)
        print(f"Model loaded from: {model_path}")

    df_ = val_df.copy()

    # Ensure target is string for fair comparison (HS codes often have leading zeros)
    if target_col in df_.columns:
        df_[target_col] = df_[target_col].astype(str)

    # Containers for predictions
    preds_tags = [[] for _ in range(top_n)]
    preds_scores = [[] for _ in range(top_n)]

    iterator = df_.itertuples(index=False, name=None)
    if show_progress:
        iterator = tqdm(iterator, total=df_.shape[0])

    # Map column index for speed in itertuples
    cols = list(df_.columns)
    text_idx = cols.index(text_col)

    for row in iterator:
        text_val = row[text_idx]
        try:
            sims = predict_d2v(
                text_val, model, top_n=top_n, epochs=epochs, alpha=alpha, min_alpha=min_alpha
            )
        except Exception as e:
            # In case of any unexpected row-level issue, fill with nulls and continue
            sims = []

        # Normalize to length top_n with blanks if fewer results (e.g., empty text)
        if len(sims) < top_n:
            sims = sims + [("", float("nan"))] * (top_n - len(sims))

        for k in range(top_n):
            tag, score = sims[k]
            preds_tags[k].append(tag)
            preds_scores[k].append(score)

    # Write columns programmatically
    for k in range(top_n):
        df_[f"top_{k+1}"] = preds_tags[k]
        df_[f"top_{k+1}_SCORE"] = preds_scores[k]

    # ----- Metrics (top-k accuracy) -----
    metrics: Dict[str, float] = {}
    if target_col in df_.columns:
        y_true = df_[target_col].astype(str)
        # We compute cumulative "any correct up to k"
        hit_cumulative = None
        for k in range(1, top_n + 1):
            eq_k = y_true.eq(df_[f"top_{k}"].astype(str))
            hit_cumulative = eq_k if hit_cumulative is None else (hit_cumulative | eq_k)
            acc_k = float(hit_cumulative.mean())
            metrics[f"top_{k}_acc"] = round(acc_k, 6)

        total = df_.shape[0]
        print(f"Total samples: {total}")
        for k in range(1, top_n + 1):
            correct = int(metrics[f'top_{k}_acc'] * total)
            print(f"Top-{k} Accuracy: {metrics[f'top_{k}_acc']:.4f} ({correct}/{total})")
    else:
        print("(No target column present; metrics skipped.)")
        

    return df_, metrics
