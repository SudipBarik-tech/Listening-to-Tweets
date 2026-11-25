# Reproducing the results

This guide takes you from nothing to every number in the paper. Choose one of the three options below.

## Option A: your own computer

**Requirements:** Python 3.9 or newer, about 500 MB of free disk space, and an internet connection for the first setup.

```bash
git clone https://github.com/SudipBarik-tech/Listening-to-Tweets.git
cd Listening-to-Tweets

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

**Get the data:**

1. Open <https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset> (a free Kaggle account is needed).
2. Click **Download** and unzip it.
3. Create a folder `data/` inside the repository and put `Suicide_Ideation_Dataset(Twitter-based).csv` in it.

**Run:**

```bash
python prepare_data.py
python run_experiments.py --data data/tweets_balanced.csv --outdir results
python robustness_seeds.py --data data/tweets_balanced.csv
```

## Option B: Kaggle Notebook (no download needed)

1. Open the dataset page on Kaggle and click **New Notebook**. The dataset is then available under `/kaggle/input/`.
2. In the first cell:
   ```python
   !git clone https://github.com/SudipBarik-tech/Listening-to-Tweets.git
   %cd Listening-to-Tweets
   !pip install -q gensim
   import glob; raw = glob.glob('/kaggle/input/**/*.csv', recursive=True)[0]; print(raw)
   ```
3. In the next cell:
   ```python
   !python prepare_data.py --raw "{raw}"
   !python run_experiments.py --data data/tweets_balanced.csv --outdir results
   ```

## Option C: Google Colab

Upload the CSV to Colab (folder icon → upload), then run:

```python
!git clone https://github.com/SudipBarik-tech/Listening-to-Tweets.git
%cd Listening-to-Tweets
!pip install -q -r requirements.txt
!mkdir -p data && mv "/content/Suicide_Ideation_Dataset(Twitter-based).csv" data/
!python prepare_data.py
!python run_experiments.py --data data/tweets_balanced.csv --outdir results
```

## What you should see

`prepare_data.py` prints:

```
raw rows: 1787
after dropping empty rows: 1785 (suicidal 659)
after removing exact duplicates: 1777 (suicidal 653)
final: 1216 tweets, 653 suicidal (53.7%) -> data/tweets_balanced.csv
```

`run_experiments.py` prints (about 30 seconds):

```
Tweets: 1216 read, 1216 after removing exact duplicates; suicidal = 653 (53.7%)
Train 972, test 244 (test suicidal = 131)

Test-set results:
  Random Forest        acc=0.791  P=0.756  R=0.901  F1=0.822
  Logistic Regression  acc=0.770  P=0.752  R=0.855  F1=0.800
  XGBoost              acc=0.807  P=0.773  R=0.908  F1=0.835
  Decision Tree        acc=0.758  P=0.731  R=0.870  F1=0.794
  Gaussian NB          acc=0.754  P=0.780  R=0.756  F1=0.767
  k-NN                 acc=0.754  P=0.713  R=0.908  F1=0.799
  Keyword rule         acc=0.512  P=1.000  R=0.092  F1=0.168
  TF-IDF + LR          acc=0.881  P=0.864  R=0.924  F1=0.893
```

Outputs appear in `results/` (see the table in the README). Your numbers should match ours. With different library versions, the third decimal place may differ slightly.

## Command-line options

### `prepare_data.py`

| Option | Default | Meaning |
|---|---|---|
| `--raw` | `data/Suicide_Ideation_Dataset(Twitter-based).csv` | Path to the downloaded CSV |
| `--out` | `data/tweets_balanced.csv` | Where to write the balanced dataset |
| `--n-neg` | `563` | Number of non-suicidal tweets to keep |
| `--seed` | `42` | Random seed for sampling |

### `run_experiments.py`

| Option | Default | Meaning |
|---|---|---|
| `--data` | (required) | CSV with `text` and `label` columns |
| `--text-col`, `--label-col` | `text`, `label` | Column names, if your CSV differs |
| `--seed` | `42` | Seed for the split, Word2Vec and models |
| `--w2v` | `own` | `own` trains Word2Vec on training tweets (as in the paper); `google` uses pre-trained GoogleNews vectors |
| `--outdir` | `results` | Output folder |

### `robustness_seeds.py`

| Option | Default | Meaning |
|---|---|---|
| `--data` | `data/tweets_balanced.csv` | Balanced dataset |
| `--n-seeds` | `10` | Number of random splits |
| `--out` | `results/robustness_seeds.csv` | Summary file |

## Troubleshooting

| Problem | Fix |
|---|---|
| `LookupError: Resource stopwords/wordnet/punkt_tab not found` | Run the `nltk.download(...)` line from the setup again |
| `ModuleNotFoundError: gensim` or `xgboost` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: run_experiments` (from `robustness_seeds.py`) | Run the script from the repository's top folder |
| `FileNotFoundError: data/Suicide_Ideation_Dataset(Twitter-based).csv` | Download the CSV (see "Get the data") or pass `--raw path/to/file.csv` |
| Numbers differ in the 3rd decimal | Different library versions; compare with `results/settings.txt` |
