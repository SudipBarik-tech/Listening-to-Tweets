"""
run_experiments.py  --  one script that produces every number in Copy.tex.

Usage (after prepare_data.py):

    python run_experiments.py --data labelled_tweets.csv [--text-col text] [--label-col label]
                              [--seed 42] [--w2v own|google] [--outdir results]

The CSV must contain one row per tweet: the tweet text and a 0/1 label
(1 = potential suicide post, 0 = not suicide post).

Outputs (in --outdir):
  table1_rows.tex       rows to paste into Table I of Copy.tex (6 models + 2 baselines)
  cv_results.csv        stratified 5-fold cross-validation, mean and SD per metric
  mcnemar.txt           McNemar exact p-values for every pair of models on the test set
  cm_*.jpg              confusion matrices of the six Word2Vec models (paper Fig. 3)
  settings.txt          seed, library versions and all model settings (for Section IV)
  preprocess_example.txt  pipeline output for the example sentence in Section III-D
"""
import argparse, os, re, sys, math, platform
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from scipy.stats import binomtest
import sklearn

# ---------------------------------------------------------------- settings
# Collection keywords from Section III-B. Edit this list if your real list differs.
KEYWORDS = ["anxiety disorder", "depression", "help me out", "trapped", "kill myself",
            "suffering", "sleep forever", "my sad life", "struggle"]
EXAMPLE = "can't take this anymore... nobody even cares lol"
W2V_PARAMS = dict(vector_size=100, window=5, min_count=2, epochs=20, sg=1)

def make_models(seed):
    from xgboost import XGBClassifier
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "XGBoost":             XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=seed),
        "Decision Tree":       DecisionTreeClassifier(random_state=seed),
        "Gaussian NB":         GaussianNB(),
        "k-NN":                KNeighborsClassifier(n_neighbors=5),
    }
# Confusion-matrix images used in the paper (Fig. 3, panels a-f)
FIG_NAMES = {"Logistic Regression": "cm_lr.jpg", "Gaussian NB": "cm_nb.jpg", "Random Forest": "cm_rf.jpg",
             "Decision Tree": "cm_dt.jpg", "XGBoost": "cm_xgb.jpg", "k-NN": "cm_knn.jpg"}

# ---------------------------------------------------------------- preprocessing
def get_nlp_tools():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        sys.exit("NLTK stopwords not available; run nltk.download('stopwords') first.")
    lem = WordNetLemmatizer()
    try:
        lem.lemmatize("tears")
    except LookupError:
        sys.exit("NLTK WordNet not available; run nltk.download('wordnet') first.")
    return word_tokenize, stops, lem

EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf]", flags=re.UNICODE)
def clean(text, tok, stops, lem):
    t = str(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)      # URLs
    t = re.sub(r"@\w+", " ", t)                   # user mentions
    t = EMOJI.sub(" ", t)                         # emojis
    t = re.sub(r"\d+", " ", t)                    # digits
    t = re.sub(r"[^\w\s']", " ", t)               # punctuation (keep apostrophes for tokenizer)
    words = [w for w in tok(t) if w.isalpha() and w not in stops]
    return [lem.lemmatize(w) for w in words]

# ---------------------------------------------------------------- features
def train_w2v(token_lists, mode, seed):
    if mode == "google":
        import gensim.downloader as api
        return api.load("word2vec-google-news-300")      # pre-trained, 300-d
    from gensim.models import Word2Vec
    m = Word2Vec(sentences=token_lists, seed=seed, workers=1, **W2V_PARAMS)
    return m.wv

def tweet_vectors(token_lists, wv):
    dim = wv.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [wv[w] for w in toks if w in wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)               # average of word vectors
    return out

# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def metrics(y, yhat):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    lo, hi = wilson(tp + tn, n)
    return dict(acc=acc, lo=lo, hi=hi, prec=prec, rec=rec, f1=f1, bal=(rec + spec) / 2,
                tn=tn, fp=fp, fn=fn, tp=tp)

def mcnemar_exact(y, a, b):
    """Exact McNemar test on the tweets where the two models disagree."""
    a_right, b_right = (a == y), (b == y)
    n01 = int(np.sum(a_right & ~b_right)); n10 = int(np.sum(~a_right & b_right))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, binomtest(min(n01, n10), n01 + n10, 0.5).pvalue

def keyword_predict(texts):
    low = [str(t).lower() for t in texts]
    return np.array([int(any(k in t for k in KEYWORDS)) for t in low])

def save_cm(m, name, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=200)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Predicted Negative", "Predicted Positive"], fontsize=7)
    ax.set_yticks([0, 1], ["Actual Negative", "Actual Positive"], fontsize=7, rotation=90, va="center")
    ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(path, format="jpg"); plt.close(fig)

def row(name, m):
    return (f"{name} & {m['acc']:.3f} ({m['lo']:.3f}--{m['hi']:.3f}) & {m['prec']:.3f} & "
            f"{m['rec']:.3f} & {m['f1']:.3f} & {m['bal']:.3f} \\\\\n\\hline")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Reproduce every number in the Listening to Tweets paper.")
    ap.add_argument("--data", required=True, help="CSV with one tweet per row (made by prepare_data.py)")
    ap.add_argument("--text-col", default="text", help="name of the tweet-text column")
    ap.add_argument("--label-col", default="label", help="name of the 0/1 label column (1 = potential suicide post)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for the split, Word2Vec and the models")
    ap.add_argument("--w2v", choices=["own", "google"], default="own", help="own = train Word2Vec on training tweets (paper); google = pre-trained GoogleNews vectors (1.6 GB download)")
    ap.add_argument("--outdir", default="results", help="folder for all outputs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    np.random.seed(a.seed)

    df = pd.read_csv(a.data)
    df = df[[a.text_col, a.label_col]].dropna()
    before = len(df)
    df = df.drop_duplicates(subset=[a.text_col]).reset_index(drop=True)
    texts, y = df[a.text_col].astype(str).values, df[a.label_col].astype(int).values
    print(f"Tweets: {before} read, {len(df)} after removing exact duplicates; "
          f"suicidal = {y.sum()} ({100*y.mean():.1f}%)")

    tok, stops, lem = get_nlp_tools()
    tokens = [clean(t, tok, stops, lem) for t in texts]
    with open(os.path.join(a.outdir, "preprocess_example.txt"), "w") as f:
        f.write(f"IN : {EXAMPLE}\nOUT: {' '.join(clean(EXAMPLE, tok, stops, lem))}\n")

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=a.seed)
    print(f"Train {len(tr)}, test {len(te)} (test suicidal = {y[te].sum()})")

    # Word2Vec fitted on TRAINING tweets only (no test leakage)
    wv = train_w2v([tokens[i] for i in tr], a.w2v, a.seed)
    X = tweet_vectors(tokens, wv)

    models = make_models(a.seed)
    preds, results, lines = {}, {}, []
    for name, clf in models.items():
        clf.fit(X[tr], y[tr])
        preds[name] = clf.predict(X[te])
        results[name] = m = metrics(y[te], preds[name])
        save_cm(m, name, os.path.join(a.outdir, FIG_NAMES[name]))
        lines.append(row(name, m))

    # Baselines
    kw = keyword_predict(texts[te])
    results["Keyword rule"] = metrics(y[te], kw); preds["Keyword rule"] = kw
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xt_tr = tfidf.fit_transform([" ".join(tokens[i]) for i in tr])
    Xt_te = tfidf.transform([" ".join(tokens[i]) for i in te])
    lr_t = LogisticRegression(max_iter=2000).fit(Xt_tr, y[tr])
    preds["TF-IDF + LR"] = lr_t.predict(Xt_te)
    results["TF-IDF + LR"] = metrics(y[te], preds["TF-IDF + LR"])
    lines.append("\\hline\n\\multicolumn{6}{|l|}{\\textit{Baselines}} \\\\\n\\hline")
    lines.append(row("Keyword rule", results["Keyword rule"]))
    lines.append(row("TF-IDF + LR", results["TF-IDF + LR"]))
    with open(os.path.join(a.outdir, "table1_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # McNemar for every pair
    names = list(preds)
    with open(os.path.join(a.outdir, "mcnemar.txt"), "w") as f:
        f.write("model A | model B | A right & B wrong | A wrong & B right | exact p\n")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n01, n10, p = mcnemar_exact(y[te], preds[names[i]], preds[names[j]])
                f.write(f"{names[i]} | {names[j]} | {n01} | {n10} | {p:.4f}\n")

    # Stratified 5-fold CV (Word2Vec refitted inside each fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    cv = {n: [] for n in models}
    for k, (ftr, fte) in enumerate(skf.split(idx, y)):
        wv_k = train_w2v([tokens[i] for i in ftr], a.w2v, a.seed)
        Xk = tweet_vectors(tokens, wv_k)
        for name, clf in make_models(a.seed).items():
            clf.fit(Xk[ftr], y[ftr])
            cv[name].append(metrics(y[fte], clf.predict(Xk[fte])))
    rows = []
    for name, ms in cv.items():
        r = {"model": name}
        for key in ["acc", "prec", "rec", "f1", "bal"]:
            vals = [m[key] for m in ms]
            r[f"{key}_mean"], r[f"{key}_sd"] = np.mean(vals), np.std(vals, ddof=1)
        rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.outdir, "cv_results.csv"), index=False)

    # Settings for Section IV
    import gensim, xgboost
    with open(os.path.join(a.outdir, "settings.txt"), "w") as f:
        f.write(f"seed={a.seed}\nsplit=stratified 80:20\nword2vec={a.w2v} {W2V_PARAMS if a.w2v=='own' else 'GoogleNews 300d'}"
                f", fitted on training tweets only; tweet vector = mean of word vectors\n")
        f.write(f"python={platform.python_version()} sklearn={sklearn.__version__} "
                f"gensim={gensim.__version__} xgboost={xgboost.__version__}\n")
        for name, clf in make_models(a.seed).items():
            f.write(f"{name}: {clf.get_params()}\n")

    print("\nTest-set results:")
    for name, m in results.items():
        print(f"  {name:20s} acc={m['acc']:.3f}  P={m['prec']:.3f}  R={m['rec']:.3f}  F1={m['f1']:.3f}")
    print(f"\nAll outputs written to {a.outdir}/")

if __name__ == "__main__":
    main()
