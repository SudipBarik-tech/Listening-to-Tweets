"""Two reader-friendly figures for the Sravana paper (single IEEE column).

Run from the repository top folder: python figures/explain_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "pdf.fonttype": 42})
EDGE = "#3A3A3A"

# ---------------------------------------------------------------- Fig: tweet -> numbers
fig = plt.figure(figsize=(3.5, 2.15))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 61); ax.axis("off")

def box(x, y, w, h, text, fc, size=6.4, bold=False, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.5",
                                fc=fc, ec=EDGE, lw=0.6))
    tx = x + w / 2 if ha == "center" else x + 1.5
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, linespacing=1.3,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.6, color=EDGE, shrinkA=0, shrinkB=0))

box(3, 52, 94, 7, "Made-up tweet:  “can't take this anymore... nobody even cares lol”", "#F2F2F2", 6.6)
arrow(50, 52, 50, 48.5)
box(3, 41, 94, 7.5, "After cleaning:  ca · take · anymore · nobody · even · care · lol", "#E8EEF6", 6.6)
arrow(26, 41, 26, 37.5); arrow(74, 41, 74, 37.5)

ax.text(26, 35.3, "TF-IDF: every word keeps its own place", ha="center", fontsize=6.6, fontweight="bold")
box(3, 10.5, 46, 22,
    "1,737 slots, one for each word or\nword pair; almost all are 0.\n"
    "Non-zero slots for this tweet:\n"
    "take anymore  0.46\nnobody  0.43     ca  0.36\ncare  0.33     lol  0.33 …",
    "#FBE5CC", 6.2)
ax.text(74, 35.3, "Word2Vec: all words are mixed", ha="center", fontsize=6.6, fontweight="bold")
box(51, 10.5, 46, 22,
    "each of the 7 words gets\n100 numbers, then they are\naveraged into one list:\n"
    "[0.12, 0.04, −0.17, 0.01, −0.17, …]\n(100 numbers in total)",
    "#DDEBDD", 6.2)
ax.text(26, 4.8, "a strong word or pair such as\n“take anymore” stays visible", ha="center",
        fontsize=6.0, style="italic", linespacing=1.2)
ax.text(74, 4.8, "a strong word is blurred into\nthe average of all words", ha="center",
        fontsize=6.0, style="italic", linespacing=1.2)
fig.savefig("figures/tweet_to_numbers.pdf"); fig.savefig("figures/tweet_to_numbers.png", dpi=600)
plt.close(fig)

# ---------------------------------------------------------------- Fig: caught / missed / false alarms
rows = [  # name, caught (of 131), false alarms (of 113); from results/extended/test_results.csv
    ("GloVe + Random Forest", 119, 7),
    ("GloVe + XGBoost", 122, 13),
    ("TF-IDF + Random Forest", 119, 10),
    ("TF-IDF + Log. Regression", 121, 19),
    ("Own Word2Vec + XGBoost", 119, 35),
    ("Own Word2Vec + Random Forest", 118, 38),
    ("Keyword rule (baseline)", 12, 0),
]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
fig.subplots_adjust(left=0.37, right=0.97, top=0.86, bottom=0.16)
import numpy as np
y = np.arange(len(rows))[::-1]
caught = [r[1] for r in rows]; fa = [r[2] for r in rows]
h = 0.38
ax.barh(y + h / 2, caught, height=h, color="#4C7FB5", label="suicidal tweets caught (out of 131)")
ax.barh(y - h / 2, fa, height=h, color="#E39A4C", label="false alarms (out of 113 others)")
for yi, c, f in zip(y, caught, fa):
    ax.text(c + 1.5, yi + h / 2, str(c), va="center", fontsize=5.8)
    ax.text(f + 1.5, yi - h / 2, str(f), va="center", fontsize=5.8)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.3)
ax.set_xlim(0, 140); ax.set_xlabel("number of test tweets", fontsize=6.5)
ax.tick_params(axis="x", labelsize=6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.axvline(131, color="#4C7FB5", lw=0.5, ls=(0, (2, 2)))
ax.text(132, len(rows) - 0.55, "all 131", ha="left", va="center", fontsize=5.6, color="#4C7FB5")
ax.legend(loc="lower center", bbox_to_anchor=(0.32, 1.0), ncol=1, fontsize=6, frameon=False)
fig.savefig("figures/caught_vs_false.pdf"); fig.savefig("figures/caught_vs_false.png", dpi=600)
print("ok")
