# Listening to Tweets

**Detecting suicidal ideation on Twitter with Word2Vec, TF-IDF and six classical classifiers**

This repository contains the complete, reproducible code for the paper:

> S. Barik and A. Ghosh, *"Listening to Tweets: Detecting Suicidal Ideation on Twitter with Word2Vec, TF-IDF and Six Classical Classifiers,"* submitted to the IEEE International Conference on Research in Computational Intelligence and Communication Networks (ICRCICN 2027), Kalyani, India.

Every number, table and confusion matrix in the paper is produced by the scripts here. A full run takes about 30 seconds on an ordinary laptop.

> [!IMPORTANT]
> This is **research code for studying text**. It is **not** a diagnostic, screening or crisis tool, and it must not be used to make decisions about real people. If you or someone you know is struggling, please contact a local helpline or a mental health professional. See [docs/ETHICS.md](docs/ETHICS.md).

---

## The story in one minute

People who are struggling often write about it online before they speak to anyone. We asked a simple question: **can a computer read short tweets and gently flag the ones that show signs of suicidal thoughts?**

Using 1,216 annotated English tweets, we compared six classical classifiers on **Word2Vec** features, and checked them against two simple yardsticks: a **keyword rule** and **TF-IDF + Logistic Regression**.

What we found:

* With Word2Vec features, **XGBoost and Random Forest** did best, at about **85% accuracy** in 5-fold cross-validation. They caught about **9 in 10 suicidal tweets**, but with many false alarms.
* The simple **TF-IDF + Logistic Regression** baseline did **significantly better**: **88.1% accuracy** on the test set, catching 121 of 131 suicidal tweets (exact McNemar test vs. every Word2Vec model, *p* ≤ 0.007).
* A **keyword rule** caught only 12 of 131 suicidal tweets, so the models learn much more than a list of obvious phrases.

**Lesson:** on small datasets, averaging Word2Vec vectors blurs strong single words; plain word counts keep them. Always test a simple baseline first.

---

## Results at a glance

Held-out test set: 244 tweets (131 suicidal, 113 not). P, R, F1 are for the suicidal class.

| Model | Test accuracy (95% CI) | Precision | Recall | F1 | 5-fold CV accuracy |
|---|---|---|---|---|---|
| XGBoost + Word2Vec | 0.807 (0.753–0.852) | 0.773 | 0.908 | 0.835 | 0.849 ± 0.025 |
| Random Forest + Word2Vec | 0.791 (0.736–0.837) | 0.756 | 0.901 | 0.822 | 0.849 ± 0.019 |
| Logistic Regression + Word2Vec | 0.770 (0.714–0.819) | 0.752 | 0.855 | 0.800 | 0.800 ± 0.031 |
| Decision Tree + Word2Vec | 0.758 (0.701–0.808) | 0.731 | 0.870 | 0.794 | 0.782 ± 0.018 |
| Gaussian Naive Bayes + Word2Vec | 0.754 (0.696–0.804) | 0.780 | 0.756 | 0.767 | 0.754 ± 0.027 |
| k-NN + Word2Vec | 0.754 (0.696–0.804) | 0.713 | 0.908 | 0.799 | 0.800 ± 0.028 |
| Keyword rule (baseline) | 0.512 (0.450–0.574) | 1.000 | 0.092 | 0.168 | – |
| **TF-IDF + Logistic Regression (baseline)** | **0.881 (0.835–0.916)** | **0.864** | **0.924** | **0.893** | – |

More detail, including all pairwise significance tests and a 10-split robustness check, is in [docs/RESULTS.md](docs/RESULTS.md).

---

## Quick start

```bash
# 1. Get the code
git clone https://github.com/SudipBarik-tech/Listening-to-Tweets.git
cd Listening-to-Tweets

# 2. Install the Python packages (Python 3.9 or newer)
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

# 3. Put the dataset in data/ (see "Data" below), then:
python prepare_data.py                                        # builds data/tweets_balanced.csv
python run_experiments.py --data data/tweets_balanced.csv     # all paper numbers -> results/
python robustness_seeds.py                                    # optional: 10 random splits
```

Step-by-step instructions, including Google Colab and Kaggle Notebooks and a troubleshooting section, are in [docs/REPRODUCING.md](docs/REPRODUCING.md).

---

## Data

We use the public **Suicidal Tweet Detection Dataset** by Syeda Aunanya Mahmud (2023):

* Kaggle: <https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset>
* File: `Suicide_Ideation_Dataset(Twitter-based).csv` (1,787 tweets; columns `Tweet`, `Suicide`)
* Licence: **CC BY-NC-SA 4.0** (non-commercial use; please cite the dataset)

The tweets are **not** included in this repository. Download the CSV and place it at `data/Suicide_Ideation_Dataset(Twitter-based).csv`. The dataset card, with counts at every step, is in [docs/DATA.md](docs/DATA.md).

---

## Repository layout

```
Listening-to-Tweets/
├── README.md                 ← you are here
├── requirements.txt          ← Python packages
├── CITATION.cff              ← how to cite this work
├── prepare_data.py           ← raw CSV  → balanced 1,216-tweet dataset
├── run_experiments.py        ← all models, baselines, statistics and figures
├── robustness_seeds.py       ← repeats the evaluation over 10 random splits
├── data/                     ← put the downloaded CSV here (ignored by git)
├── results/                  ← every output used in the paper
│   ├── table1_rows.tex       ← rows of Table II in the paper (LaTeX)
│   ├── cv_results.csv        ← 5-fold cross-validation, mean and SD
│   ├── mcnemar.txt           ← exact McNemar test for every pair of models
│   ├── robustness_seeds.csv  ← 10-split robustness summary
│   ├── settings.txt          ← seed, library versions, all hyperparameters
│   ├── preprocess_example.txt← cleaning example shown in the paper
│   └── cm_*.jpg              ← confusion matrices (paper Fig. 3)
└── docs/
    ├── METHODS.md            ← every step of the pipeline, with parameters
    ├── REPRODUCING.md        ← detailed run guide and troubleshooting
    ├── RESULTS.md            ← full results and how to read them
    ├── DATA.md               ← dataset card
    └── ETHICS.md             ← responsible-use statement
```

---

## How it works (short version)

1. **Prepare data** (`prepare_data.py`): drop 2 empty rows → remove 8 exact duplicates → keep all 653 suicidal tweets and a random 563 non-suicidal tweets (seed 42).
2. **Clean text**: lower-case; remove URLs, @mentions, emojis, digits, punctuation; NLTK tokenise; remove stop words; WordNet lemmatise.
3. **Split**: stratified 80:20 → 972 training and 244 test tweets (seed 42).
4. **Features**: skip-gram Word2Vec (100-d, window 5, min count 2, 20 epochs) trained on **training tweets only**; a tweet = the average of its word vectors.
5. **Models**: Logistic Regression, Gaussian NB, Decision Tree, Random Forest (300 trees), XGBoost (300 trees), k-NN (k = 5).
6. **Baselines**: a 9-phrase keyword rule, and TF-IDF (1–2-grams, min df 2) + Logistic Regression.
7. **Evaluation**: accuracy with 95% Wilson CIs; suicidal-class precision, recall, F1; balanced accuracy; exact McNemar tests; stratified 5-fold CV with Word2Vec retrained inside each fold.

The full, detailed description is in [docs/METHODS.md](docs/METHODS.md).

---

## Citation

If you use this code, please cite the paper and the dataset (see [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{barik2027listening,
  title     = {Listening to Tweets: Detecting Suicidal Ideation on Twitter with Word2Vec, TF-IDF and Six Classical Classifiers},
  author    = {Barik, Sudip and Ghosh, Ammlan},
  booktitle = {IEEE International Conference on Research in Computational Intelligence and Communication Networks (ICRCICN)},
  year      = {2027},
  note      = {Submitted}
}

@misc{mahmud2023suicidal,
  title        = {Suicidal Tweet Detection Dataset},
  author       = {Mahmud, Syeda Aunanya},
  year         = {2023},
  howpublished = {\url{https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset}},
  note         = {CC BY-NC-SA 4.0}
}
```

## Authors

* **Sudip Barik**, Department of Master of Computer Application, Techno International New Town, Kolkata, India (sudip.barik@tint.edu.in)
* **Ammlan Ghosh**, Department of Artificial Intelligence and Data Science, Saveetha School of Engineering, Chennai, India (ammlanghosh.sse@saveetha.com)

## Licence

The authors have not yet chosen a licence for this code. Until a `LICENSE` file is added, all rights are reserved by the authors. The dataset keeps its own CC BY-NC-SA 4.0 licence.
