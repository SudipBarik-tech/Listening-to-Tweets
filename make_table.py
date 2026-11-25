"""
make_table.py -- builds the wide results table (Table III of Copy.tex) from the
CSV files written by extended_experiments.py.

    python make_table.py --indir results/extended
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default="results/extended")
a = ap.parse_args()
t = pd.read_csv(os.path.join(a.indir, "test_results.csv"))
c = pd.read_csv(os.path.join(a.indir, "cv_results.csv"))

kinds = ["Word2Vec (own)", "GloVe (pre-trained)", "TF-IDF"]
models = ["Logistic Regression", "Gaussian NB", "Decision Tree", "Random Forest", "XGBoost", "k-NN"]
best = {k: t[t.features == k].acc.max() for k in kinds}

lines = []
for m in models:
    cells = [m]
    for k in kinds:
        r = t[(t.features == k) & (t.model == m)].iloc[0]
        v = c[(c.features == k) & (c.model == m)].iloc[0]
        # recompute from the confusion counts so that rounding is exact
        acc_v = (r.tp + r.tn) / (r.tp + r.tn + r.fp + r.fn)
        prec = r.tp / (r.tp + r.fp) if r.tp + r.fp else 0.0
        rec = r.tp / (r.tp + r.fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = f"{acc_v:.3f}"
        if r.acc == best[k]:
            acc = f"\\textbf{{{acc}}}"
        cells += [acc, f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{v.acc_mean:.3f}"]
    lines.append(" & ".join(cells) + " \\\\")
out = os.path.join(a.indir, "table_wide.tex")
open(out, "w").write("\n".join(lines) + "\n")
print(open(out).read())
