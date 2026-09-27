"""Stage 3 - Model building with experiment tracking.

1. Loads the train/test splits (downloaded from the "data-splits" workflow
   artifact in CI).
2. Defines six candidate algorithms - Decision Tree, Bagging, Random Forest,
   AdaBoost, Gradient Boosting and XGBoost - each with a parameter grid.
3. Tunes every model with stratified 5-fold GridSearchCV, optimising F1.
   Only ~19% of customers buy, so every model is trained with balanced
   sample weights to stop it ignoring the buyers.
4. Logs EVERY tuned parameter combination to MLflow as a nested run, plus a
   parent run per model family with its best parameters and metrics.
5. Tunes each model's decision threshold on out-of-fold predictions and picks
   the champion on that cross-validated F1. The test set is never used to
   choose anything - it is only used for the final, unbiased report.
6. Saves the champion (preprocessing + model + threshold) to
   tourism_project/deployment/, where the workflow commits it to main.

A quality gate fails the job if the champion's test F1 is below MIN_TEST_F1,
so a weak model is never committed or deployed.
"""
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (CATEGORICAL_FEATURES, CV_FOLDS, DEPLOY_DIR, METRICS_PATH,
                    MLFLOW_EXPERIMENT, MLFLOW_URI, MODEL_PATH,
                    NUMERIC_FEATURES, POST_CONTACT_FEATURES, RANDOM_STATE,
                    SPLIT_DIR)

EXCLUDE_POST_CONTACT = os.environ.get("EXCLUDE_POST_CONTACT", "false").lower() == "true"
MIN_TEST_F1 = float(os.environ.get("MIN_TEST_F1", "0.60"))
RS = RANDOM_STATE


# ----------------------------------------------------------------- models
def get_model_grid():
    """The six candidate algorithms, each with a hyperparameter grid."""
    return {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RS),
            {"model__max_depth": [4, 6, 8, None],
             "model__min_samples_leaf": [1, 5, 10]},
        ),
        "Bagging": (
            BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS),
                              random_state=RS, n_jobs=1),
            {"model__n_estimators": [50, 100],
             "model__max_samples": [0.7, 1.0],
             "model__max_features": [0.7, 1.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=1),
            {"model__max_depth": [10, None],
             "model__min_samples_leaf": [1, 3],
             "model__max_features": ["sqrt", 0.5]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RS),
                               random_state=RS),
            {"model__n_estimators": [100, 200],
             "model__learning_rate": [0.5, 1.0],
             "model__estimator__max_depth": [1, 2, 3]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(subsample=0.8, random_state=RS),
            {"model__n_estimators": [200, 400],
             "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [3, 5]},
        ),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                          random_state=RS, n_jobs=1),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.05, 0.1]},
        ),
    }


def feature_lists():
    drop = set(POST_CONTACT_FEATURES) if EXCLUDE_POST_CONTACT else set()
    return ([c for c in NUMERIC_FEATURES if c not in drop],
            [c for c in CATEGORICAL_FEATURES if c not in drop])


def build_pipeline(estimator, num, cat):
    """Impute + one-hot encode, then classify.

    The cleaned training data has no missing values, but the imputers keep
    the deployed app robust to incomplete rows in uploaded CSVs, and
    handle_unknown='ignore' stops it crashing on an unseen category.
    (Tree models don't need feature scaling, so none is applied.)
    """
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", Pipeline([
             ("impute", SimpleImputer(strategy="most_frequent")),
             ("onehot", OneHotEncoder(handle_unknown="ignore")),
         ]), cat)],
        remainder="drop",
    )
    return Pipeline([("preprocessor", pre), ("model", estimator)])


# ------------------------------------------------------------- evaluation
def evaluate(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def tune_threshold(y, oof_proba):
    """Probability cut-off that maximises F1 on out-of-fold predictions."""
    grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
    scores = [f1_score(y, (oof_proba >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def short(params):
    return {k.replace("model__", ""): v for k, v in params.items()}


# ------------------------------------------------------------------- main
def load_splits():
    Xtrain = pd.read_csv(SPLIT_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(SPLIT_DIR / "Xtest.csv")
    ytrain = pd.read_csv(SPLIT_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(SPLIT_DIR / "ytest.csv").squeeze("columns")
    print(f"Loaded splits: train {Xtrain.shape}, test {Xtest.shape}")
    return Xtrain, Xtest, ytrain, ytest


def main():
    Xtrain, Xtest, ytrain, ytest = load_splits()
    num, cat = feature_lists()
    weights = compute_sample_weight("balanced", ytrain)
    fit_params = {"model__sample_weight": weights}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_tag = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Excluding post-contact features: {EXCLUDE_POST_CONTACT} "
          f"({len(num + cat)} features)\n")

    results, best = [], None
    for name, (estimator, grid) in get_model_grid().items():
        with mlflow.start_run(run_name=f"{name}-{run_tag}"):
            search = GridSearchCV(build_pipeline(estimator, num, cat), grid,
                                  scoring="f1", cv=cv, n_jobs=-1, refit=True)
            search.fit(Xtrain, ytrain, **fit_params)

            # Log EVERY parameter combination the grid search tried.
            res = search.cv_results_
            for i, params in enumerate(res["params"]):
                with mlflow.start_run(run_name=f"{name}_combo_{i:02d}", nested=True):
                    mlflow.set_tag("model_family", name)
                    mlflow.log_params(short(params))
                    mlflow.log_metric("mean_cv_f1", res["mean_test_score"][i])
                    mlflow.log_metric("std_cv_f1", res["std_test_score"][i])
                    mlflow.log_metric("rank_cv_f1", int(res["rank_test_score"][i]))

            # Threshold from out-of-fold predictions (training data only).
            model = search.best_estimator_
            oof = cross_val_predict(model, Xtrain, ytrain, cv=cv, method="predict_proba",
                                    n_jobs=-1, params=fit_params)[:, 1]
            threshold, oof_f1 = tune_threshold(ytrain, oof)
            train_m = evaluate(ytrain, model.predict_proba(Xtrain)[:, 1], threshold)
            test_m = evaluate(ytest, model.predict_proba(Xtest)[:, 1], threshold)

            # Parent run = summary for this model family.
            mlflow.set_tag("model_family", name)
            mlflow.log_params(short(search.best_params_))
            mlflow.log_params({"threshold": threshold,
                               "exclude_post_contact": EXCLUDE_POST_CONTACT})
            mlflow.log_metrics({"best_cv_f1": search.best_score_, "oof_f1_tuned": oof_f1,
                                "n_param_combinations_tried": len(res["params"]),
                                **{f"train_{k}": v for k, v in train_m.items()},
                                **{f"test_{k}": v for k, v in test_m.items()}})

        print(f"[{name:16s}] {len(res['params']):2d} combos | best {short(search.best_params_)}")
        print(f"{'':19s} CV F1 {search.best_score_:.3f} | OOF F1 {oof_f1:.3f} @ threshold "
              f"{threshold:.2f} | test F1 {test_m['f1']:.3f}, ROC-AUC {test_m['roc_auc']:.3f}")

        results.append({"model": name, "best_params": short(search.best_params_),
                        "cv_f1": round(float(search.best_score_), 4),
                        "oof_f1": round(oof_f1, 4), "threshold": threshold,
                        "train_f1": train_m["f1"], **test_m})

        # Champion chosen on cross-validated (training) F1 - never on test.
        if best is None or oof_f1 > best["oof_f1"]:
            best = {"name": name, "model": model, "threshold": threshold, "oof_f1": oof_f1,
                    "params": short(search.best_params_), "train": train_m, "test": test_m}

    comparison = pd.DataFrame(results).sort_values("oof_f1", ascending=False)
    print("\nModel comparison (ranked by cross-validated F1; test set shown for reference):")
    print(comparison[["model", "cv_f1", "oof_f1", "threshold", "train_f1", "accuracy",
                      "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

    test_pred = (best["model"].predict_proba(Xtest)[:, 1] >= best["threshold"]).astype(int)
    print(f"\nCHAMPION: {best['name']}  (threshold {best['threshold']:.2f})")
    print("Test metrics:", best["test"])

    # ---------------------------------------- artefacts committed by the workflow
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["model"], "model_name": best["name"],
                 "threshold": best["threshold"], "features": num + cat,
                 "numeric_features": num, "categorical_features": cat}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best["name"], "best_params": best["params"],
                   "threshold": best["threshold"], "cv_f1": round(best["oof_f1"], 4),
                   "metrics": best["test"], "train_metrics": best["train"],
                   "confusion_matrix": confusion_matrix(ytest, test_pred).tolist(),
                   "exclude_post_contact": EXCLUDE_POST_CONTACT,
                   "all_results": comparison.to_dict(orient="records"),
                   "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2, default=str)
    print(f"Saved champion -> {MODEL_PATH.name}, metrics -> {METRICS_PATH.name}")

    # ---------------------------------------- quality gate
    if best["test"]["f1"] < MIN_TEST_F1:
        sys.exit(f"QUALITY GATE FAILED: test F1 {best['test']['f1']:.3f} < {MIN_TEST_F1}")
    print(f"Quality gate PASSED (test F1 {best['test']['f1']:.3f} >= {MIN_TEST_F1})")


if __name__ == "__main__":
    main()
