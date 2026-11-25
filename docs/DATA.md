# Dataset card

## Source

| | |
|---|---|
| Name | Suicidal Tweet Detection Dataset |
| Creator | Syeda Aunanya Mahmud (Kaggle user `aunanya875`) |
| Link | <https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset> |
| Last updated on Kaggle | 7 August 2023 |
| Licence | CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike) |
| File | `Suicide_Ideation_Dataset(Twitter-based).csv` (227,918 bytes) |
| Columns | `Tweet` (text), `Suicide` (label) |
| Labels | `Potential Suicide post`, `Not Suicide post` |
| Language | English |

The labels were assigned by the dataset's creators. We did not re-label the tweets.

## Counts

| Stage | Total | Potential suicide post | Not suicide post |
|---|---|---|---|
| Raw file | 1,787 | 660 | 1,127 |
| After dropping empty rows | 1,785 | 659 | 1,126 |
| After dropping exact duplicates | 1,777 | 653 | 1,124 |
| **Balanced dataset used in the paper** | **1,216** | **653 (53.7%)** | **563 (46.3%)** |
| Training split (80%) | 972 | 522 | 450 |
| Test split (20%) | 244 | 131 | 113 |

## Known limitations

* **Small:** 1,216 tweets, so every test-set score has a confidence interval of about ±5 points.
* **Labels are text judgements, not diagnoses.** A "potential suicide post" label describes the text, not the clinical state of the writer.
* **Balanced by sampling.** About half of the non-suicidal tweets were left out. In real life, suicidal posts are much rarer, so false alarms would matter more.
* **English only, Twitter only.** Results may not carry over to other languages (including Indian languages and code-mixed text) or other platforms.
* **Collection period and method are not documented** by the dataset's creators.

## What is not in this repository

The tweets themselves are **not** redistributed here. Please download them from Kaggle and respect the CC BY-NC-SA 4.0 licence.
