# Results

> **Update (after review): the full, fair comparison.** The paper's main table (Table III) now gives every one of the six classifiers each of three kinds of features: our own Word2Vec, ready-made GloVe Twitter vectors, and TF-IDF. Those numbers come from `extended_experiments.py` and are in `results/extended/`. The own-Word2Vec numbers below are unchanged and appear again in Table III.

## Table III: three kinds of features × six classifiers

Same 244 test tweets for every row. `results/extended/test_results.csv` has precision, recall, F1, Wilson intervals and confusion counts; `cv_results.csv` has 5-fold CV means and SDs.

* Best results: **GloVe + Random Forest 92.2%** (95% CI 88.2–95.0%; 119/131 caught, 7 false alarms) and **TF-IDF + Random Forest 91.0%** (119/131 caught, 10 false alarms). McNemar between them: *p* = 0.69, so no clear winner.
* GloVe vs own Word2Vec (same classifier): significantly better for 5 of 6 classifiers (*p* ≤ 0.008; k-NN *p* = 0.23).
* TF-IDF vs own Word2Vec: significantly better for RF, LR, XGBoost, Decision Tree (*p* ≤ 0.007); no difference for Naive Bayes (*p* = 0.37); much worse for k-NN (k-NN on sparse TF-IDF vectors finds few truly similar tweets).
* GloVe vs TF-IDF: no significant difference for RF, LR, XGBoost, Decision Tree (*p* ≥ 0.49). All pairs: `mcnemar_vs_w2v.txt`.
* GloVe knows 91% of the training vocabulary; only 3 tweets have no known word (own Word2Vec: 1,178-word vocabulary, 17 empty tweets).
* **No balancing:** 5-fold CV on all 1,777 tweets (36.7% suicidal), `full_data_cv.csv`: TF-IDF + RF 91.2%, GloVe + XGBoost 91.0%, own Word2Vec + RF 87.1%. Same order as on the balanced set.
* **Realistic prevalence** (best model, recall 0.908, false-alarm rate 7/113 = 0.062): about 43% of alerts correct at 5% prevalence, about 13% at 1%.


All numbers below are produced by the scripts in this repository (seed 42 unless stated). Files are in [`../results/`](../results/).

## 1. Held-out test set (244 tweets: 131 suicidal, 113 not)

| Model | Correct | Accuracy (95% Wilson CI) | Precision | Recall | F1 | Balanced acc. |
|---|---|---|---|---|---|---|
| XGBoost + Word2Vec | 197 | 0.807 (0.753–0.852) | 0.773 | 0.908 | 0.835 | 0.799 |
| Random Forest + Word2Vec | 193 | 0.791 (0.736–0.837) | 0.756 | 0.901 | 0.822 | 0.782 |
| Logistic Regression + Word2Vec | 188 | 0.770 (0.714–0.819) | 0.752 | 0.855 | 0.800 | 0.764 |
| Decision Tree + Word2Vec | 185 | 0.758 (0.701–0.808) | 0.731 | 0.870 | 0.794 | 0.749 |
| Gaussian NB + Word2Vec | 184 | 0.754 (0.696–0.804) | 0.780 | 0.756 | 0.767 | 0.754 |
| k-NN + Word2Vec | 184 | 0.754 (0.696–0.804) | 0.713 | 0.908 | 0.799 | 0.742 |
| Keyword rule | 125 | 0.512 (0.450–0.574) | 1.000 | 0.092 | 0.168 | 0.546 |
| **TF-IDF + Logistic Regression** | **215** | **0.881 (0.835–0.916)** | **0.864** | **0.924** | **0.893** | **0.878** |

Precision, recall and F1 are for the **suicidal** class.

### Confusion matrices (TN, FP, FN, TP)

| Model | TN | FP (false alarms) | FN (missed suicidal) | TP |
|---|---|---|---|---|
| XGBoost | 78 | 35 | 12 | 119 |
| Random Forest | 75 | 38 | 13 | 118 |
| Logistic Regression | 76 | 37 | 19 | 112 |
| Decision Tree | 71 | 42 | 17 | 114 |
| Gaussian NB | 85 | 28 | 32 | 99 |
| k-NN | 65 | 48 | 12 | 119 |
| Keyword rule | 113 | 0 | 119 | 12 |
| TF-IDF + LR | 94 | 19 | 10 | 121 |

Images: `results/cm_*.jpg` (paper Fig. 3).

## 2. Stratified 5-fold cross-validation (Word2Vec retrained in each fold)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.849 ± 0.025 | 0.825 ± 0.020 | 0.911 ± 0.036 | 0.866 ± 0.023 |
| Random Forest | 0.849 ± 0.019 | 0.822 ± 0.020 | 0.917 ± 0.028 | 0.867 ± 0.018 |
| Logistic Regression | 0.800 ± 0.031 | 0.794 ± 0.028 | 0.849 ± 0.044 | 0.820 ± 0.029 |
| k-NN | 0.800 ± 0.028 | 0.758 ± 0.029 | 0.925 ± 0.026 | 0.833 ± 0.023 |
| Decision Tree | 0.782 ± 0.018 | 0.776 ± 0.015 | 0.836 ± 0.027 | 0.805 ± 0.017 |
| Gaussian NB | 0.754 ± 0.027 | 0.823 ± 0.030 | 0.691 ± 0.041 | 0.751 ± 0.031 |

(mean ± standard deviation over 5 folds; source: `results/cv_results.csv`)

## 3. Is one model really better than another? (exact McNemar test)

Selected pairs on the 244 test tweets (all pairs are in `results/mcnemar.txt`). "A only" counts tweets that A got right and B got wrong.

| Model A | Model B | A only | B only | p-value | Meaning |
|---|---|---|---|---|---|
| TF-IDF + LR | XGBoost | 29 | 11 | 0.006 | TF-IDF clearly better |
| TF-IDF + LR | Random Forest | 34 | 12 | 0.002 | TF-IDF clearly better |
| TF-IDF + LR | Logistic Regression (W2V) | 32 | 5 | < 0.001 | TF-IDF clearly better |
| XGBoost | Random Forest | 9 | 5 | 0.42 | No real difference |
| Random Forest | Logistic Regression (W2V) | 13 | 8 | 0.38 | No real difference |

TF-IDF + LR beats **every** Word2Vec model with p ≤ 0.007. Among the six Word2Vec models, most differences are not significant.

## 4. Robustness over 10 random splits (`robustness_seeds.py`)

| Model | Mean accuracy | SD | Min | Max |
|---|---|---|---|---|
| TF-IDF + LR | **0.887** | 0.015 | 0.861 | 0.910 |
| XGBoost | 0.851 | 0.025 | 0.820 | 0.898 |
| Random Forest | 0.838 | 0.023 | 0.799 | 0.885 |
| k-NN | 0.803 | 0.016 | 0.779 | 0.824 |
| Logistic Regression | 0.803 | 0.018 | 0.779 | 0.828 |
| Decision Tree | 0.784 | 0.022 | 0.750 | 0.824 |
| Gaussian NB | 0.768 | 0.023 | 0.734 | 0.803 |

The ranking is stable: TF-IDF + LR is best on every split, and no Word2Vec model goes above 0.90 on any split.

## 5. How to read these results

* **Word2Vec models catch most suicidal tweets but raise many false alarms.** XGBoost misses 12 of 131 suicidal tweets but wrongly flags 35 of 113 others.
* **Gaussian NB fails the other way:** fewer false alarms (28) but more missed suicidal tweets (32), the more dangerous mistake here.
* **Why TF-IDF wins:** Word2Vec is learnt from fewer than 1,000 short tweets (vocabulary of 1,178 words), and averaging word vectors blurs strong single words such as *die* or *alone*. TF-IDF keeps each word and word pair separately.
* **The keyword rule is weak** (12 of 131 suicidal tweets caught), so the dataset is not just a keyword-matching task.

## 6. Note on an earlier draft

An earlier draft of the paper reported about 97% accuracy for Random Forest and Logistic Regression with Word2Vec. We could **not** reproduce that figure. Across clean and more favourable settings (Word2Vec trained on all tweets, duplicates kept, the paper's original 1,222-tweet split) and 10 random splits, no Word2Vec model exceeded about 90%. The numbers on this page replace the earlier ones and match the submitted paper.
