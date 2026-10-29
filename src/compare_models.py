"""Refit every candidate model with ONE common protocol and cache test-set predictions.

Why this file exists (Person 3 / Evaluation)
--------------------------------------------
The modeling notebook reports baselines on a *validation* split (fit on 80% of train)
but the final model on the *test* split. To compare models fairly and run paired
statistical tests we need every model's probabilities on the SAME test rows.

Protocol (identical to 0_2_Modeling.ipynb, nothing new is tuned here):
  1. GroupShuffleSplit(test_size=0.2, random_state=42) on train, grouped by patient_nbr
  2. fit on the 'fit' part, pick the F1-optimal threshold on the validation part
  3. refit on the full train set, predict the test set, apply that threshold

Usage (from the repo root):
    python src/compare_models.py            # uses cache if it exists
    python src/compare_models.py --force    # recompute everything
Output: outputs/evaluation/model_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the modeling notebook (copied, not changed)."""
    df = df.copy()
    df["total_previous_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    h = df["time_in_hospital"].clip(lower=1)
    df["medications_per_day"] = df["num_medications"] / h
    df["labs_per_day"] = df["num_lab_procedures"] / h
    df["procedures_per_day"] = df["num_procedures"] / h
    df["treatment_intensity"] = df["num_lab_procedures"] + df["num_procedures"] + df["num_medications"]
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if c in X.columns and c not in cat:
            cat.append(c)
    num = [c for c in X.select_dtypes(include="number").columns if c not in cat]
    return ColumnTransformer(
        [("num", Pipeline([("scaler", RobustScaler())]), num),
         ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)],
        remainder="drop",
    )


def build_candidates(pre: ColumnTransformer, tuned_params: dict) -> dict[str, Pipeline]:
    from catboost import CatBoostClassifier   # imported lazily: only needed when models are actually re-fit

    def pipe(clf):
        return Pipeline([("preprocessor", clone(pre)), ("classifier", clf)])

    cb = dict(iterations=200, depth=6, learning_rate=0.1, loss_function="Logloss", random_seed=SEED, verbose=0)
    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=1000, random_state=SEED)),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=SEED)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                                     random_state=SEED, n_jobs=-1)),
        "CatBoost": pipe(CatBoostClassifier(**cb)),
        "Logistic Regression + Class Weight": pipe(LogisticRegression(max_iter=1000, class_weight="balanced",
                                                                      random_state=SEED)),
        "Decision Tree + Class Weight": pipe(DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                                                    class_weight="balanced", random_state=SEED)),
        "Random Forest + Class Weight": pipe(RandomForestClassifier(n_estimators=200, max_depth=12,
                                                                    min_samples_leaf=10, class_weight="balanced",
                                                                    random_state=SEED, n_jobs=-1)),
        "CatBoost + Class Weight": pipe(CatBoostClassifier(**cb, auto_class_weights="Balanced")),
        "Tuned CatBoost": pipe(CatBoostClassifier(**tuned_params, loss_function="Logloss",
                                                  random_seed=SEED, verbose=0)),
    }


def f1_optimal_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p, pos_label=1)
    d = prec[:-1] + rec[:-1]
    f1 = np.divide(2 * prec[:-1] * rec[:-1], d, out=np.zeros_like(d), where=d != 0)
    return float(thr[np.argmax(f1)])


def run(root: Path, force: bool = False) -> pd.DataFrame:
    out = root / "outputs" / "evaluation" / "model_predictions.csv"
    if out.exists() and not force:
        print(f"[cache] {out} exists - skipping refit (use --force to recompute)")
        return pd.read_csv(out)

    train = pd.read_csv(root / "data" / "train_clean.csv")
    test = pd.read_csv(root / "data" / "test_clean.csv")
    groups = pd.read_csv(root / "data" / "train_patient_groups.csv")["patient_nbr"].to_numpy()
    params = json.load(open(root / "outputs" / "best_params.json"))["best_params"]

    Xtr, ytr = add_advanced_features(train).drop(columns="target"), train["target"]
    Xte, yte = add_advanced_features(test).drop(columns="target"), test["target"]

    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(Xtr, ytr, groups))
    Xf, Xv, yf, yv = Xtr.iloc[fit_idx], Xtr.iloc[val_idx], ytr.iloc[fit_idx], ytr.iloc[val_idx]

    part = out.with_name("model_predictions_partial.csv")
    thr_file = out.with_name("model_thresholds.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if part.exists() and thr_file.exists() and not force:       # resume an interrupted run
        res, thresholds = pd.read_csv(part), json.load(open(thr_file))
    else:
        res, thresholds = pd.DataFrame({"y_true": yte.to_numpy()}), {}
    for name, pipe in build_candidates(build_preprocessor(Xtr), params).items():
        if name in res.columns:
            print(f"[resume] {name} already done", flush=True)
            continue
        t0 = time.time()
        m = clone(pipe).fit(Xf, yf)
        thr = f1_optimal_threshold(yv, m.predict_proba(Xv)[:, 1])
        m = clone(pipe).fit(Xtr, ytr)
        res[name] = m.predict_proba(Xte)[:, 1]
        thresholds[name] = thr
        res.to_csv(part, index=False)
        json.dump(thresholds, open(thr_file, "w"), indent=2)
        print(f"{name:38s} thr={thr:.4f}  ({time.time() - t0:5.1f}s)", flush=True)

    res.to_csv(out, index=False)
    print("saved", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    run(Path(a.root), a.force)
