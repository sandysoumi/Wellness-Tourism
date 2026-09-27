# 🧳 Wellness Tourism Package — Purchase Prediction (MLOps Pipeline)

[![MLOps Pipeline](https://github.com/benitabenony/wellness-tourism-mlops/actions/workflows/pipeline.yml/badge.svg)](https://github.com/benitabenony/wellness-tourism-mlops/actions/workflows/pipeline.yml)

Predicts whether a customer will purchase **Visit with Us**'s new **Wellness Tourism
Package**, so the sales team can prioritise the right customers. A fully automated
GitHub Actions pipeline validates the data, retrains and tracks six models in MLflow,
commits the champion to `main`, and Streamlit Community Cloud redeploys the app.

🔗 **Live app:** https://wellness-tourism-mlops-5dcw6mnappvyvqmhgz56vcp.streamlit.app/
📓 **Full project report:** [`wellness_tourism_full_project.ipynb`](wellness_tourism_full_project.ipynb) — executed end to end, with observations after every step.

## Pipeline

```
push to main
   │
   ▼
1. data_registration   3 validation gates (columns, types, binary target) + summary
   ▼
2. data_preparation    clean → stratified 80/20 split ─► artifact: data-splits
   ▼
3. model_building      download splits → tune 6 models (5-fold CV) → MLflow
   │                   → quality gate (test F1 ≥ 0.60) ─► artifacts: best-model, mlflow-tracking
   ▼
4. deploy_model        commit model.joblib + metrics.json to main  [skip ci]
   ▼
5. output_report       folder structure + model-comparison table in the run summary
   │
   ▼
Streamlit Community Cloud sees the new commit on main and redeploys the app
```

## Repository structure

```
.github/workflows/pipeline.yml          CI/CD workflow (5 jobs)
tourism_project/                        master project folder
├── data/
│   └── tourism.csv                     registered dataset
├── model_building/
│   ├── config.py                       shared paths, schema, feature lists
│   ├── data_register.py                stage 1 – validation + summary
│   ├── prep.py                         stage 2 – cleaning + split
│   └── train.py                        stage 3 – tuning, MLflow, evaluation
├── deployment/
│   ├── app.py                          Streamlit app
│   ├── requirements.txt                app dependencies (Streamlit Cloud)
│   ├── model.joblib                    champion model  ← committed by the pipeline
│   └── metrics.json                    its metrics     ← committed by the pipeline
└── requirements.txt                    pipeline dependencies (pinned)
wellness_tourism_full_project.ipynb     executed project notebook
```

## Results

The champion is chosen on **cross-validated F1**; the held-out test set is only used to report final performance.

| Model | CV F1 | Test F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** (champion) | **0.785** | **0.826** | 0.860 | 0.793 | 0.954 |
| XGBoost | 0.781 | 0.839 | 0.853 | 0.826 | 0.958 |
| Bagging | 0.776 | 0.816 | 0.778 | 0.858 | 0.968 |
| Random Forest | 0.742 | 0.767 | 0.685 | 0.871 | 0.951 |
| Decision Tree | 0.654 | 0.745 | 0.776 | 0.716 | 0.833 |
| AdaBoost | 0.618 | 0.613 | 0.555 | 0.684 | 0.870 |

## Key design decisions

- **Cleaning.** The pipeline drops `Unnamed: 0` and `CustomerID`. It also drops `ProductPitched`, which maps one-to-one to `Designation`. It fixes 155 `Fe Male` typos and removes 117 duplicates. `Unmarried` is kept separate from `Single`, because the two groups convert at 24% and 36%.
- **Class imbalance.** Only 19% of customers buy, so all six models are trained with balanced sample weights.
- **Honest model selection.** Each model's decision threshold is tuned on out-of-fold predictions, and the champion is picked by cross-validated F1, never by test-set results.
- **Experiment tracking.** Every parameter combination (60 in total) is logged to MLflow as a nested run. There is also one parent run per model family, holding its best parameters and metrics.
- **Quality gate.** If test F1 is below 0.60, the job fails, so nothing is committed or deployed.
- **Reproducibility.** Library versions are pinned identically for training and the app, so `model.joblib` always loads.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse every tuned run
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

## Deploying

1. Push to GitHub. In *Settings → Actions → General*, set workflow permissions to **Read and write**.
2. On [Streamlit Community Cloud](https://share.streamlit.io), deploy repo → branch `main` →
   main file `tourism_project/deployment/app.py` → **Python 3.11** (Advanced settings).
