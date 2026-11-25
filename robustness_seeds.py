"""
robustness_seeds.py -- repeats the held-out evaluation over several random splits.

The main results in the paper come from one 80:20 split (seed 42). This script
repeats the whole procedure (split -> Word2Vec on training tweets -> six models,
plus the TF-IDF + Logistic Regression baseline) for seeds 0..N-1, to show how
much the scores move from one split to another (paper Section V-B).

Usage:
    python robustness_seeds.py --data data/tweets_balanced.csv --n-seeds 10 --out results/robustness_seeds.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import run_experiments as R

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--data", default="data/tweets_balanced.csv", help="CSV made by prepare_data.py")
ap.add_argument("--n-seeds", type=int, default=10, help="number of random splits")
ap.add_argument("--out", default="results/robustness_seeds.csv", help="where to save the summary")
a = ap.parse_args()

tok, stops, lem = R.get_nlp_tools()
df = pd.read_csv(a.data).drop_duplicates("text")
y = df["label"].values
tokens = [R.clean(t, tok, stops, lem) for t in df["text"].astype(str)]
idx = np.arange(len(df))

acc = {}
for seed in range(a.n_seeds):
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    wv = R.train_w2v([tokens[i] for i in tr], "own", seed)          # training tweets only
    X = R.tweet_vectors(tokens, wv)
    for name, clf in R.make_models(seed).items():
        clf.fit(X[tr], y[tr])
        acc.setdefault(name, []).append(R.metrics(y[te], clf.predict(X[te]))["acc"])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    A = tfidf.fit_transform([" ".join(tokens[i]) for i in tr])
    B = tfidf.transform([" ".join(tokens[i]) for i in te])
    pred = LogisticRegression(max_iter=2000).fit(A, y[tr]).predict(B)
    acc.setdefault("TF-IDF + LR", []).append(R.metrics(y[te], pred)["acc"])

rows = [{"model": n, "mean_acc": np.mean(v), "sd_acc": np.std(v, ddof=1),
         "min_acc": min(v), "max_acc": max(v)} for n, v in acc.items()]
out = pd.DataFrame(rows).round(4)
out.to_csv(a.out, index=False)
print(out.to_string(index=False))
