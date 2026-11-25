"""
prepare_data.py -- builds the 1,216-tweet balanced dataset used in the paper.

Input : the public "Suicidal Tweet Detection Dataset" (Mahmud, 2023), file
        "Suicide_Ideation_Dataset(Twitter-based).csv" with columns Tweet, Suicide.
        Kaggle: https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset
Output: data/tweets_balanced.csv with columns text, label (1 = potential suicide post)

Steps (paper Section III-C):
  1,787 rows -> drop 2 rows with no text      -> 1,785 (659 suicidal)
             -> drop 8 exact duplicate tweets -> 1,777 (653 suicidal)
             -> keep all suicidal, sample 563 non-suicidal (seed 42) -> 1,216
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/Suicide_Ideation_Dataset(Twitter-based).csv")
ap.add_argument("--out", default="data/tweets_balanced.csv")
ap.add_argument("--n-neg", type=int, default=563)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--full-out", default="data/tweets_full.csv",
                help="also save all de-duplicated tweets (no balancing), used by extended_experiments.py")
a = ap.parse_args()

df = pd.read_csv(a.raw)
print(f"raw rows: {len(df)}")
df = df.dropna(subset=["Tweet", "Suicide"])
df["label"] = (df["Suicide"].str.strip() == "Potential Suicide post").astype(int)
print(f"after dropping empty rows: {len(df)} (suicidal {df.label.sum()})")
df = df.drop_duplicates(subset=["Tweet"])
print(f"after removing exact duplicates: {len(df)} (suicidal {df.label.sum()})")
full = df.rename(columns={"Tweet": "text"})[["text", "label"]]
os.makedirs(os.path.dirname(a.full_out) or ".", exist_ok=True)
full.to_csv(a.full_out, index=False)
print(f"all de-duplicated tweets: {len(full)} -> {a.full_out}")
pos = df[df.label == 1]
neg = df[df.label == 0].sample(n=a.n_neg, random_state=a.seed)
out = (pd.concat([pos, neg]).sample(frac=1, random_state=a.seed)
         .rename(columns={"Tweet": "text"})[["text", "label"]])
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
out.to_csv(a.out, index=False)
print(f"final: {len(out)} tweets, {out.label.sum()} suicidal ({100*out.label.mean():.1f}%) -> {a.out}")
