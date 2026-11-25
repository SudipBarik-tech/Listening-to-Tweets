"""Methodology figure for the Sravana paper (single IEEE column, 3.5 in wide).

Run from the repository top folder: python figures/methodology.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "pdf.fonttype": 42})

W, H = 3.5, 4.75                    # inches
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 136); ax.axis("off")

C = {"data": "#DCE6F2", "prep": "#E8EEF6", "split": "#F2F2F2",
     "w2v": "#DDEBDD", "glove": "#D6ECEC", "tfidf": "#FBE5CC", "kw": "#EDE4F2", "eval": "#F7DCDC", "edge": "#3A3A3A"}

def box(x, y, w, h, title, body=None, color="#FFFFFF", bold=True, size=7, bsize=6.1, gap=4.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.6",
                                fc=color, ec=C["edge"], lw=0.6))
    if body:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=size,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h - 2.2 - gap, body, ha="center", va="top", fontsize=bsize, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", linespacing=1.2)

def arrow(pts):
    """Polyline with an arrow head on the last segment."""
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=C["edge"], lw=0.6, solid_capstyle="butt")
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.6, color=C["edge"], shrinkA=0, shrinkB=0))

FULL_X, FULL_W = 3, 94
# 1. data
box(FULL_X, 123, FULL_W, 11, "Suicidal Tweet Detection Dataset (Kaggle)",
    "1,787 English tweets  |  labels: potential suicide post / not suicide post", C["data"])
arrow([(50, 123), (50, 119.5)])
# 2. data preparation
box(FULL_X, 104.5, FULL_W, 15, "Data preparation",
    "drop 2 empty rows (1,785)  →  drop 8 duplicates (1,777)\n"
    "keep all 653 suicidal + random 563 not suicidal (seed 42)  →  1,216 tweets", C["data"])
arrow([(50, 104.5), (50, 101)])
# 3. text cleaning
box(FULL_X, 86, FULL_W, 15, "Text cleaning",
    "lower-case; remove URLs, @mentions, emojis, digits, punctuation\n"
    "NLTK tokenise  →  remove stop words  →  WordNet lemmatise", C["prep"])
arrow([(50, 86), (50, 82.5)])
# 4. split
box(FULL_X, 76, FULL_W, 6.5, "Stratified 80 : 20 split (seed 42)", color=C["split"])
box(8, 66.5, 34, 6, "Train: 972 tweets", bold=False, size=6.5)
box(58, 66.5, 30, 6, "Test: 244 tweets", bold=False, size=6.5)
arrow([(25, 76), (25, 72.5)])
arrow([(73, 76), (73, 72.5)])

# 5. fitting region: four feature boxes, then the classifiers
cols = [4.5, 26, 47.5, 69]; cw = 19.5
ax.add_patch(FancyBboxPatch((3, 29), 87.5, 35, boxstyle="round,pad=0.2,rounding_size=1.6",
                            fc="none", ec="#777777", lw=0.6, ls=(0, (3, 2))))
ax.text(88.5, 62.2, "fitted on training tweets only", ha="right", va="center",
        fontsize=5.8, style="italic", color="#444444")
arrow([(25, 66.5), (25, 60.5)])
feat = [("Word2Vec", "own, learnt\nfrom 972 tweets", C["w2v"]),
        ("GloVe", "ready-made,\nfrom 2B tweets", C["glove"]),
        ("TF-IDF", "word 1–2-grams\nmin. freq. 2", C["tfidf"]),
        ("Keyword", "9 distress\nphrases", C["kw"])]
for x, (t, b, c) in zip(cols, feat):
    box(x, 45.5, cw, 12, t, b, c, bsize=5.6, size=6.6)
box(cols[0], 31, cols[2] + cw - cols[0], 11.5, "Six classifiers on each feature type",
    "LR · Naive Bayes · Decision Tree\nRandom Forest · XGBoost · k-NN", "#F4F4E8", bsize=5.8, gap=4.2)
box(cols[3], 31, cw, 11.5, "Rule-based", "baseline,\nno training", C["kw"], bsize=5.6, gap=4.2, size=6.6)
for x in cols:
    arrow([(x + cw / 2, 45.5), (x + cw / 2, 42.5)])
ax.plot([cols[0] + cw / 2, cols[3] + cw / 2], [60.5, 60.5], color=C["edge"], lw=0.6)
for x in cols:
    arrow([(x + cw / 2, 60.5), (x + cw / 2, 57.5)])

# 6. evaluation: models feed down; test tweets come in on the right lane
for x in (cols[0] + (cols[2] + cw - cols[0]) / 2 - 18, cols[0] + (cols[2] + cw - cols[0]) / 2 + 18, cols[3]):
    xx = x + cw / 2 if x == cols[3] else x
    arrow([(xx, 31), (xx, 23)])
arrow([(88, 69.5), (95, 69.5), (95, 23)])
ax.text(96.4, 46, "test tweets", rotation=-90, ha="center", va="center", fontsize=5.6,
        style="italic", color="#444444")
box(FULL_X, 2, FULL_W, 21, "Evaluation on the 244 test tweets",
    "accuracy with 95% Wilson interval\n"
    "precision, recall, F1 (suicidal class)\n"
    "exact McNemar test between models\n"
    "stratified 5-fold cross-validation (features learnt inside each fold)\n"
    "repeat on all 1,777 tweets; precision at realistic prevalence", C["eval"], bsize=6.1)

for ext in ("pdf", "png"):
    fig.savefig(f"figures/methodology.{ext}", dpi=600)
print("saved")
