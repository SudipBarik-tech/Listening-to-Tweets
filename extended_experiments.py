"""
extended_experiments.py -- the fair, full comparison added after review.

It answers three questions a reviewer will ask:
  1. Does TF-IDF beat Word2Vec with ALL six classifiers, not only with
     Logistic Regression?
  2. Is our Word2Vec weak only because it learnt from fewer than 1,000 tweets?
     (We add pre-trained GloVe vectors, learnt from 2 billion tweets.)
  3. Do the results change if we do not balance the classes and use all
     1,777 tweets instead of the 1,216-tweet balanced set?

Usage (after prepare_data.py):

    python extended_experiments.py --data data/tweets_balanced.csv \
        --full data/tweets_full.csv --glove glove-twitter-100.gz --outdir results/extended

glove-twitter-100.gz comes from the gensim-data project:
https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/glove-twitter-100.gz
(if --glove is not given, the GloVe rows are skipped).

Outputs (in --outdir):
  test_results.csv     one row per (features, model) on the same 244 test tweets
  cv_results.csv       stratified 5-fold CV, mean and SD, features learnt inside each fold
  mcnemar_vs_w2v.txt   for each classifier: TF-IDF vs Word2Vec and GloVe vs Word2Vec
  full_data_cv.csv     the same 5-fold CV on all 1,777 tweets (no balancing)
  table_rows.tex       rows for the results table in Copy.tex
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from run_experiments import (get_nlp_tools, clean, train_w2v, tweet_vectors, make_models,
                             metrics, mcnemar_exact)

TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2)


def load_glove(path):
    from gensim.models import KeyedVectors
    print(f"Loading GloVe vectors from {path} (takes a minute)...")
    return KeyedVectors.load_word2vec_format(path, binary=False)


def features(kind, tokens, tr, seed, glove=None):
    """Return the feature matrix for all tweets; anything learnt uses training tweets only."""
    if kind == "Word2Vec (own)":
        return tweet_vectors(tokens, train_w2v([tokens[i] for i in tr], "own", seed))
    if kind == "GloVe (pre-trained)":
        return tweet_vectors(tokens, glove)
    if kind == "TF-IDF":
        vec = TfidfVectorizer(**TFIDF_PARAMS)
        docs = [" ".join(t) for t in tokens]
        vec.fit([docs[i] for i in tr])
        return vec.transform(docs).toarray().astype(np.float32)
    raise ValueError(kind)


def read(path, tok, stops, lem):
    df = pd.read_csv(path)[["text", "label"]].dropna()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    y = df["label"].astype(int).values
    tokens = [clean(t, tok, stops, lem) for t in df["text"].astype(str)]
    return tokens, y


def cross_validate(tokens, y, kinds, seed, glove):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    store = {}
    for ftr, fte in skf.split(np.arange(len(y)), y):
        for kind in kinds:
            X = features(kind, tokens, ftr, seed, glove)
            for name, clf in make_models(seed).items():
                clf.fit(X[ftr], y[ftr])
                store.setdefault((kind, name), []).append(metrics(y[fte], clf.predict(X[fte])))
    rows = []
    for (kind, name), ms in store.items():
        r = {"features": kind, "model": name}
        for key in ["acc", "prec", "rec", "f1"]:
            v = [m[key] for m in ms]
            r[f"{key}_mean"], r[f"{key}_sd"] = np.mean(v), np.std(v, ddof=1)
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", required=True, help="balanced CSV from prepare_data.py (text,label)")
    ap.add_argument("--full", help="optional CSV with all 1,777 de-duplicated tweets (text,label)")
    ap.add_argument("--glove", help="path to glove-twitter-100.gz (optional)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results/extended")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    tok, stops, lem = get_nlp_tools()
    glove = load_glove(a.glove) if a.glove else None
    kinds = ["Word2Vec (own)"] + (["GloVe (pre-trained)"] if glove is not None else []) + ["TF-IDF"]

    tokens, y = read(a.data, tok, stops, lem)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=a.seed)
    print(f"Balanced set: {len(y)} tweets, train {len(tr)}, test {len(te)}")
    if glove is not None:
        vocab = {w for i in tr for w in tokens[i]}
        cover = np.mean([w in glove for w in vocab])
        empty = sum(1 for t in tokens if not any(w in glove for w in t))
        print(f"GloVe covers {100*cover:.1f}% of the training vocabulary; {empty} tweets have no known word")

    preds, rows = {}, []
    for kind in kinds:
        X = features(kind, tokens, tr, a.seed, glove)
        for name, clf in make_models(a.seed).items():
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te]); preds[(kind, name)] = p
            m = metrics(y[te], p)
            rows.append({"features": kind, "model": name, **{k: m[k] for k in
                         ["acc", "lo", "hi", "prec", "rec", "f1", "tp", "fn", "fp", "tn"]}})
    test = pd.DataFrame(rows).round(4)
    test.to_csv(os.path.join(a.outdir, "test_results.csv"), index=False)

    with open(os.path.join(a.outdir, "mcnemar_vs_w2v.txt"), "w") as f:
        f.write("classifier | other features | other right & W2V wrong | other wrong & W2V right | exact p\n")
        for name in make_models(a.seed):
            for kind in kinds[1:]:
                n01, n10, p = mcnemar_exact(y[te], preds[(kind, name)], preds[("Word2Vec (own)", name)])
                f.write(f"{name} | {kind} | {n01} | {n10} | {p:.4f}\n")
        if "GloVe (pre-trained)" in kinds:
            f.write("\nclassifier | GloVe right & TF-IDF wrong | GloVe wrong & TF-IDF right | exact p\n")
            for name in make_models(a.seed):
                n01, n10, p = mcnemar_exact(y[te], preds[("GloVe (pre-trained)", name)], preds[("TF-IDF", name)])
                f.write(f"{name} | {n01} | {n10} | {p:.4f}\n")

    cv = cross_validate(tokens, y, kinds, a.seed, glove)
    cv.to_csv(os.path.join(a.outdir, "cv_results.csv"), index=False)

    # LaTeX rows: test accuracy (CI), P, R, F1, CV accuracy
    lines = []
    for kind in kinds:
        lines.append(f"\\multicolumn{{6}}{{|l|}}{{\\textit{{{kind} features}}}} \\\\\n\\hline")
        for name in make_models(a.seed):
            t = test[(test.features == kind) & (test.model == name)].iloc[0]
            c = cv[(cv.features == kind) & (cv.model == name)].iloc[0]
            lines.append(f"{name} & {t.acc:.3f} ({t.lo:.3f}--{t.hi:.3f}) & {t.prec:.3f} & {t.rec:.3f} & "
                         f"{t.f1:.3f} & {c.acc_mean:.3f} $\\pm$ {c.acc_sd:.3f} \\\\\n\\hline")
    with open(os.path.join(a.outdir, "table_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    if a.full:
        ftok, fy = read(a.full, tok, stops, lem)
        print(f"Full set: {len(fy)} tweets, suicidal {fy.sum()} ({100*fy.mean():.1f}%)")
        cross_validate(ftok, fy, kinds, a.seed, glove).to_csv(
            os.path.join(a.outdir, "full_data_cv.csv"), index=False)

    print(test[["features", "model", "acc", "prec", "rec", "f1"]].to_string(index=False))
    print(cv[["features", "model", "acc_mean", "acc_sd"]].to_string(index=False))
    print(f"All outputs written to {a.outdir}/")


if __name__ == "__main__":
    main()
