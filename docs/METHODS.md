# Methods

This page explains, step by step, exactly what the code does and why. Section numbers refer to the paper.

## 1. Data preparation (`prepare_data.py`, paper §III-C)

| Step | Tweets | Suicidal | Not suicidal | Why |
|---|---|---|---|---|
| Raw Kaggle file | 1,787 | 660 | 1,127 | – |
| Drop rows with no tweet text | 1,785 | 659 | 1,126 | Nothing to classify |
| Drop exact duplicate tweets | 1,777 | 653 | 1,124 | A duplicate in both the training and test sets would give the model an unfair peek |
| Keep all suicidal, sample 563 non-suicidal (seed 42) | **1,216** | **653 (53.7%)** | **563** | Balance the two classes |

* Labels come from the dataset's `Suicide` column. `"Potential Suicide post"` becomes 1 and `"Not Suicide post"` becomes 0. The raw file has a trailing space in some labels, so labels are stripped first.
* The output, `data/tweets_balanced.csv`, has two columns: `text` and `label`.

## 2. Text cleaning (`clean()` in `run_experiments.py`, paper §III-D)

For every tweet, in this order:

1. Lower-case the text.
2. Remove URLs (`http…`, `www.…`), user mentions (`@name`), emojis, digits and punctuation, using regular expressions. Apostrophes are kept until tokenisation.
3. Tokenise with NLTK's `word_tokenize`.
4. Keep only alphabetic tokens that are **not** in NLTK's English stop-word list.
5. Lemmatise each token with NLTK's `WordNetLemmatizer` (default part of speech: noun).

Example (from `results/preprocess_example.txt`):

```
IN : can't take this anymore... nobody even cares lol
OUT: ca take anymore nobody even care lol
```

The leftover `ca` comes from the tokenizer splitting *can't* into *ca* + *n't*. We keep it as it is, rather than hide it.

## 3. Train/test split (paper §IV-D)

* `train_test_split(test_size=0.2, stratify=label, random_state=42)`
* Training set: 972 tweets. Test set: **244 tweets** (131 suicidal, 113 not).
* Every model is trained on the same training tweets and tested on the same test tweets.

## 4. Features (paper §IV-A)

### Word2Vec (main features)

* gensim `Word2Vec`: skip-gram (`sg=1`), `vector_size=100`, `window=5`, `min_count=2`, `epochs=20`, `seed=42`, `workers=1` (single worker for reproducibility).
* **Trained on training tweets only**, so no information from the test tweets leaks into the features.
* A tweet vector is the **mean** of the vectors of its words that are in the vocabulary. A tweet with no known word becomes the zero vector (17 tweets in the main run).
* Vocabulary size in the main run: 1,178 words.
* Option: `--w2v google` uses the pre-trained 300-d GoogleNews vectors instead (1.6 GB download; not used in the paper).

### TF-IDF (baseline)

* scikit-learn `TfidfVectorizer(ngram_range=(1, 2), min_df=2)` on the cleaned, joined tokens.
* Fitted on training tweets only.

## 5. Classifiers (paper §IV-B)

| Model | Library | Settings (everything else is the library default) |
|---|---|---|
| Logistic Regression | scikit-learn | `max_iter=2000` |
| Gaussian Naive Bayes | scikit-learn | defaults |
| Decision Tree | scikit-learn | `random_state=42` |
| Random Forest | scikit-learn | `n_estimators=300, random_state=42` |
| XGBoost | xgboost | `n_estimators=300, eval_metric="logloss", random_state=42` |
| k-Nearest Neighbours | scikit-learn | `n_neighbors=5` |

The full parameter dump of every model is written to `results/settings.txt`. No hyperparameter tuning was done; with only 972 training tweets, we preferred fixed, sensible settings over tuning on a tiny validation set.

## 6. Baselines (paper §IV-C)

1. **Keyword rule.** A tweet is predicted *suicidal* if its raw, lower-cased text contains any of: `anxiety disorder, depression, help me out, trapped, kill myself, suffering, sleep forever, my sad life, struggle`. It shows how far one gets with obvious phrases alone.
2. **TF-IDF + Logistic Regression.** The standard reference point for text classification (`LogisticRegression(max_iter=2000)`).

## 7. Evaluation (paper §IV-D)

| Measure | How it is computed |
|---|---|
| Accuracy | (TP + TN) / n |
| 95% CI for accuracy | Wilson score interval |
| Precision, recall, F1 | For the **suicidal** class (label 1) |
| Balanced accuracy | Mean of recall on both classes |
| Pairwise comparison | **Exact McNemar test**: a two-sided binomial test on the tweets where the two models disagree |
| Cross-validation | Stratified 5-fold (`shuffle=True, random_state=42`). Word2Vec is **retrained inside every fold** on that fold's training tweets |
| Robustness | `robustness_seeds.py` repeats the 80:20 evaluation for seeds 0–9 |

## 8. Reproducibility notes

* All randomness is seeded (split, sampling, Word2Vec, tree models).
* Word2Vec uses one worker thread; multiple threads would make results vary slightly between runs.
* Exact library versions from the paper run are in `results/settings.txt` (Python 3.13, scikit-learn 1.9.1, gensim 4.4.0, xgboost 3.4.1). Other versions may change the last decimal place.
