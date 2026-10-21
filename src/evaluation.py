"""Evaluation toolkit for 30-day diabetic readmission prediction.

Everything here works from three things only: y_true, predicted probabilities and
(optionally) the test features / patient ids.  No model object is needed, so the
module is independent of the sklearn / CatBoost versions used for training.

Sections
--------
1. Metrics                (threshold-free and threshold-based)
2. Uncertainty            (bootstrap CIs, optionally clustered by patient)
3. Significance tests     (DeLong, paired bootstrap, McNemar, Nadeau-Bengio)
4. Threshold + cost       (sweep, cost-sensitive optimum, decision curve)
5. Operational views      (Recall@Top-k, gain / lift)
6. Calibration
7. Error / subgroup analysis
8. Plot helpers           (all return the matplotlib Figure and optionally save it)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upwards until a folder containing both `data/` and `src/` is found."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data").is_dir() and (cand / "src").is_dir():
            return cand
    raise FileNotFoundError("Could not locate repo root (needs data/ and src/ folders).")


PALETTE = {"main": "#1f5f99", "alt": "#d1495b", "grey": "#8d99ae", "green": "#2a9d8f", "gold": "#e9a03b"}


def _save(fig: plt.Figure, path: str | Path | None, dpi: int = 200) -> plt.Figure:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# 1. Metrics
# --------------------------------------------------------------------------- #

def confusion_counts(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _div(a, b):
    return float(a / b) if b else 0.0


def threshold_metrics(y, p, thr: float) -> dict:
    """All decision-based metrics at one threshold."""
    c = confusion_counts(y, (np.asarray(p) >= thr).astype(int))
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    f1 = _div(2 * prec * rec, prec + rec)
    f2 = _div(5 * prec * rec, 4 * prec + rec)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        **c, "Threshold": float(thr), "Precision": prec, "Recall": rec, "Specificity": spec,
        "NPV": _div(tn, tn + fn), "F1": f1, "F2": f2, "Balanced Acc": (rec + spec) / 2,
        "MCC": mcc, "Accuracy": _div(tp + tn, tp + fp + fn + tn),
        "Flagged %": 100 * _div(tp + fp, tp + fp + fn + tn),
    }


def ranking_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    prev = y.mean()
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    brier_ref = prev * (1 - prev)  # Brier of always predicting the prevalence
    return {
        "ROC-AUC": roc_auc_score(y, p), "AP (PR-AUC)": ap, "AP / prevalence": ap / prev,
        "Brier": brier, "Brier skill": 1 - brier / brier_ref, "LogLoss": log_loss(y, p),
    }


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    d = prec[:-1] + rec[:-1]
    f1 = np.divide(2 * prec[:-1] * rec[:-1], d, out=np.zeros_like(d), where=d != 0)
    return float(thr[np.argmax(f1)])


def threshold_for_recall(y, p, target_recall: float) -> float:
    """Largest threshold that still reaches `target_recall` (i.e. fewest flagged patients)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target_recall)[0]
    return float(thr[ok[0]])


# --------------------------------------------------------------------------- #
# 2. Bootstrap confidence intervals
# --------------------------------------------------------------------------- #

_METRIC_FUNCS: dict[str, Callable] = {
    "ROC-AUC": lambda y, p, t: roc_auc_score(y, p),
    "AP (PR-AUC)": lambda y, p, t: average_precision_score(y, p),
    "Precision": lambda y, p, t: _div(((p >= t) & (y == 1)).sum(), (p >= t).sum()),
    "Recall": lambda y, p, t: _div(((p >= t) & (y == 1)).sum(), (y == 1).sum()),
    "Specificity": lambda y, p, t: _div(((p < t) & (y == 0)).sum(), (y == 0).sum()),
    "F1": lambda y, p, t: threshold_metrics(y, p, t)["F1"],
    "Brier": lambda y, p, t: brier_score_loss(y, p),
}


def _resample_indices(n: int, rng: np.random.Generator, groups_index: list[np.ndarray] | None) -> np.ndarray:
    if groups_index is None:
        return rng.integers(0, n, n)
    picks = rng.integers(0, len(groups_index), len(groups_index))
    return np.concatenate([groups_index[i] for i in picks])


def _group_index(groups) -> list[np.ndarray] | None:
    if groups is None:
        return None
    return [np.asarray(ix) for ix in pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(list)]


def bootstrap_ci(y, p, thr: float, metrics: Sequence[str] = tuple(_METRIC_FUNCS), n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42, groups=None) -> pd.DataFrame:
    """Percentile bootstrap CI.  If `groups` (patient ids) is given, patients are resampled
    instead of encounters (cluster bootstrap), which respects repeated encounters."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    gi = _group_index(groups)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        ix = _resample_indices(len(y), rng, gi)
        yb, pb = y[ix], p[ix]
        if yb.min() == yb.max():
            continue
        for m in metrics:
            draws[m].append(_METRIC_FUNCS[m](yb, pb, thr))
    rows = []
    for m in metrics:
        d = np.asarray(draws[m])
        rows.append({"Metric": m, "Estimate": _METRIC_FUNCS[m](y, p, thr),
                     "CI low": np.quantile(d, alpha / 2), "CI high": np.quantile(d, 1 - alpha / 2),
                     "SE": d.std(ddof=1)})
    return pd.DataFrame(rows)


def fmt_ci(row: pd.Series, digits: int = 3) -> str:
    return f"{row['Estimate']:.{digits}f} [{row['CI low']:.{digits}f}, {row['CI high']:.{digits}f}]"


# --------------------------------------------------------------------------- #
# 3. Significance tests between two models
# --------------------------------------------------------------------------- #

def _fast_delong(preds: np.ndarray, m: int):
    """Sun & Xu (2014). preds: (k_models, n) with the m positives first."""
    k, total = preds.shape
    n = total - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([stats.rankdata(r) for r in pos])
    ty = np.vstack([stats.rankdata(r) for r in neg])
    tz = np.vstack([stats.rankdata(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y, p_a, p_b) -> dict:
    """DeLong test for the difference of two correlated ROC-AUCs (same test rows)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    preds = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(preds, int(y.sum()))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    se = np.sqrt(max(var, 1e-18))
    z = diff / se
    return {"AUC_A": aucs[0], "AUC_B": aucs[1], "diff": diff, "SE": se, "z": z,
            "CI low": diff - 1.96 * se, "CI high": diff + 1.96 * se, "p": float(2 * stats.norm.sf(abs(z)))}


def delong_auc_ci(y, p, alpha: float = 0.05) -> tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    aucs, cov = _fast_delong(np.asarray(p)[order][None, :], int(y.sum()))
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(aucs[0] - z * se), float(aucs[0] + z * se)


def paired_bootstrap_diff(y, p_a, p_b, metric: str = "AP (PR-AUC)", thr_a: float = 0.5, thr_b: float = 0.5,
                          n_boot: int = 1000, seed: int = 42, groups=None) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on shared resamples."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    f = _METRIC_FUNCS[metric]
    rng = np.random.default_rng(seed)
    gi = _group_index(groups)
    d = []
    for _ in range(n_boot):
        ix = _resample_indices(len(y), rng, gi)
        if y[ix].min() == y[ix].max():
            continue
        d.append(f(y[ix], p_a[ix], thr_a) - f(y[ix], p_b[ix], thr_b))
    d = np.asarray(d)
    est = f(y, p_a, thr_a) - f(y, p_b, thr_b)
    B = len(d)
    p_val = min(1.0, 2 * min(((d <= 0).sum() + 1) / (B + 1), ((d >= 0).sum() + 1) / (B + 1)))
    return {"metric": metric, "diff": est, "CI low": np.quantile(d, 0.025),
            "CI high": np.quantile(d, 0.975), "p": float(p_val), "p floor": 2.0 / (B + 1)}


def mcnemar_test(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on correctness of two classifiers, plus the same test restricted to
    positives (<=> recall difference) and to negatives (<=> specificity difference)."""
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    out = {}
    for label, mask in [("all", np.ones_like(y, bool)), ("positives (recall)", y == 1), ("negatives (specificity)", y == 0)]:
        ca, cb = (a[mask] == y[mask]), (b[mask] == y[mask])
        n01 = int((ca & ~cb).sum())   # A right, B wrong
        n10 = int((~ca & cb).sum())   # A wrong, B right
        p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
        out[label] = {"A right/B wrong": n01, "A wrong/B right": n10, "p": p}
    return out


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected paired t-test for k-fold / repeated CV scores."""
    d = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean diff": d.mean(), "t": np.nan, "p": 1.0, "df": k - 1}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return {"mean diff": d.mean(), "t": t, "p": float(2 * stats.t.sf(abs(t), k - 1)), "df": k - 1}


def holm_correction(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_comparison(y, proba: dict[str, np.ndarray], thresholds: dict[str, float], reference: str,
                        n_boot: int = 1000, groups=None) -> pd.DataFrame:
    """Reference model vs every other model: DeLong (AUC), paired bootstrap (AP, Recall, F1),
    McNemar (decisions).  Holm-adjusted p-values within each test family."""
    rows = []
    for name, p in proba.items():
        if name == reference:
            continue
        dl = delong_test(y, proba[reference], p)
        ap = paired_bootstrap_diff(y, proba[reference], p, "AP (PR-AUC)", n_boot=n_boot, groups=groups)
        f1 = paired_bootstrap_diff(y, proba[reference], p, "F1", thresholds[reference], thresholds[name],
                                   n_boot=n_boot, groups=groups)
        mc = mcnemar_test(y, (proba[reference] >= thresholds[reference]).astype(int),
                          (p >= thresholds[name]).astype(int))
        rows.append({
            "Model": name, "ΔAUC": dl["diff"], "ΔAUC 95% CI": f"[{dl['CI low']:+.4f}, {dl['CI high']:+.4f}]",
            "p DeLong": dl["p"], "ΔAP": ap["diff"], "ΔAP 95% CI": f"[{ap['CI low']:+.4f}, {ap['CI high']:+.4f}]",
            "p AP boot": ap["p"], "ΔF1": f1["diff"], "p F1 boot": f1["p"],
            "p McNemar": mc["all"]["p"], "p McNemar recall": mc["positives (recall)"]["p"],
            "p McNemar spec": mc["negatives (specificity)"]["p"],
        })
    df = pd.DataFrame(rows)
    for col in ["p DeLong", "p AP boot", "p F1 boot", "p McNemar", "p McNemar recall", "p McNemar spec"]:
        df[col + " (Holm)"] = holm_correction(df[col].values)
    return df


# --------------------------------------------------------------------------- #
# 4. Threshold sweep, cost-sensitive analysis, decision curve
# --------------------------------------------------------------------------- #

def threshold_sweep(y, p, thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    p = np.asarray(p)
    if thresholds is None:
        thresholds = np.unique(np.round(np.quantile(p, np.linspace(0.01, 0.995, 120)), 5))
    return pd.DataFrame([threshold_metrics(y, p, t) for t in thresholds])


def cost_curve(y, p, cost_ratio: float, thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    """Expected cost per patient, cost(FP)=1, cost(FN)=cost_ratio."""
    sw = threshold_sweep(y, p, thresholds)
    sw["Cost ratio"] = cost_ratio
    sw["Cost / patient"] = (sw["FP"] + cost_ratio * sw["FN"]) / len(y)
    return sw


def cost_analysis(y, p, cost_ratios: Sequence[float] = (1, 2, 5, 10, 20), thresholds=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all curves, summary of the optimum for each cost ratio).

    Baselines: 'treat none' (flag nobody -> pay every FN) and 'treat all' (flag everybody -> pay every FP).
    `Bayes thr` = 1/(1+ratio): the theoretically optimal cut-off if probabilities were perfectly calibrated.
    """
    y = np.asarray(y)
    n, pos = len(y), int(y.sum())
    curves, rows = [], []
    for k in cost_ratios:
        c = cost_curve(y, p, k, thresholds)
        curves.append(c)
        best = c.loc[c["Cost / patient"].idxmin()]
        none, allc = k * pos / n, (n - pos) / n
        rows.append({
            "FN:FP cost ratio": k, "Bayes thr": 1 / (1 + k), "Optimal thr": best["Threshold"],
            "Cost / patient": best["Cost / patient"], "Treat-none cost": none, "Treat-all cost": allc,
            "Saving vs best trivial %": 100 * (1 - best["Cost / patient"] / min(none, allc)),
            "Recall": best["Recall"], "Precision": best["Precision"], "Flagged %": best["Flagged %"],
        })
    return pd.concat(curves, ignore_index=True), pd.DataFrame(rows)


def decision_curve(y, p, thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    """Net benefit (Vickers & Elkin 2006)."""
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), y.mean()
    thresholds = np.linspace(0.02, 0.40, 77) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = p >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        w = t / (1 - t)
        rows.append({"Threshold": t, "Model": tp / n - fp / n * w, "Treat all": prev - (1 - prev) * w, "Treat none": 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Operational views: Recall@Top-k, gain / lift
# --------------------------------------------------------------------------- #

def topk_table(y, p, fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    n, pos = len(y), int(y.sum())
    rows = []
    for f in fractions:
        k = int(round(f * n))
        tp = int(ys[:k].sum())
        rows.append({"Top %": int(f * 100), "Patients flagged": k, "Readmissions caught": tp,
                     "Precision@k": tp / k, "Recall@k": tp / pos, "Lift": (tp / k) / (pos / n)})
    return pd.DataFrame(rows)


def decile_table(y, p) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    n, pos = len(y), int(y.sum())
    rows, cum = [], 0
    for d, ix in enumerate(np.array_split(np.arange(n), 10), start=1):
        tp = int(ys[ix].sum())
        cum += tp
        rows.append({"Decile": d, "n": len(ix), "Readmitted": tp, "Observed rate": tp / len(ix),
                     "Mean predicted": ps[ix].mean(), "Lift": (tp / len(ix)) / (pos / n), "Cum. recall": cum / pos})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Calibration
# --------------------------------------------------------------------------- #

def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index(drop=True)
    sem = np.sqrt(g["obs_rate"] * (1 - g["obs_rate"]) / g["n"])
    g["ci_low"], g["ci_high"] = g["obs_rate"] - 1.96 * sem, g["obs_rate"] + 1.96 * sem
    return g


def calibration_summary(y, p, n_bins: int = 10) -> dict:
    """ECE, calibration intercept ('calibration-in-the-large') and slope (1 = perfect, <1 = over-confident)."""
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    g = calibration_table(y, p, n_bins)
    ece = float((g["n"] / g["n"].sum() * (g["mean_pred"] - g["obs_rate"]).abs()).sum())
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    # offset = how far the average prediction is from the observed rate, on the logit scale
    offset = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))
    return {"ECE": ece, "Cal. slope": float(lr.coef_[0, 0]), "Cal. intercept (fit)": float(lr.intercept_[0]),
            "Mean predicted": float(p.mean()), "Observed rate": float(y.mean()), "Offset (logit)": offset}


# --------------------------------------------------------------------------- #
# 7. Error + subgroup analysis
# --------------------------------------------------------------------------- #

def outcome_labels(y, pred) -> np.ndarray:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)], ["TP", "FP", "FN"], "TN")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return (centre - half, centre + half)


def subgroup_report(df: pd.DataFrame, y, p, thr: float, col: str, min_n: int = 500, min_pos: int = 50) -> pd.DataFrame:
    """Performance per level of one column. Levels with too few rows/positives are flagged, not hidden."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for level, ix in df.groupby(col, observed=True).indices.items():
        yy, pp = y[ix], p[ix]
        n, npos = len(ix), int(yy.sum())
        ok = n >= min_n and npos >= min_pos and npos < n
        m = threshold_metrics(yy, pp, thr)
        if 0 < npos < n and npos >= 5 and n - npos >= 5:
            a, alo, ahi = delong_auc_ci(yy, pp)
        else:
            a, alo, ahi = (roc_auc_score(yy, pp) if 0 < npos < n else np.nan), np.nan, np.nan
        rlo, rhi = _wilson(int(m["TP"]), npos)
        plo, phi = _wilson(npos, n)
        rows.append({
            "Feature": col, "Group": level, "n": n, "Positives": npos, "Prevalence": yy.mean(),
            "Prev low": plo, "Prev high": phi, "AUC low": alo, "AUC high": ahi, "Recall low": rlo, "Recall high": rhi,
            "ROC-AUC": a,
            "AP": average_precision_score(yy, pp) if npos > 0 else np.nan,
            "Recall": m["Recall"], "Precision": m["Precision"], "FPR": 1 - m["Specificity"],
            "Flagged %": m["Flagged %"], "Mean predicted": pp.mean(),
            "Reliable": ok,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def subgroup_gaps(report: pd.DataFrame, metrics: Sequence[str] = ("ROC-AUC", "Recall", "FPR")) -> dict:
    r = report[report["Reliable"]]
    return {m: float(r[m].max() - r[m].min()) if len(r) > 1 else np.nan for m in metrics}


def compare_groups(df: pd.DataFrame, mask_a, mask_b, num_cols: Sequence[str], cat_cols: Sequence[str],
                   label_a: str = "A", label_b: str = "B") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile two sets of encounters (e.g. FN vs TP). Numeric: Cohen's d + Mann-Whitney. Categorical: chi-square."""
    mask_a, mask_b = np.asarray(mask_a), np.asarray(mask_b)
    num_rows = []
    for c in num_cols:
        a, b = df.loc[mask_a, c].astype(float), df.loc[mask_b, c].astype(float)
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(len(a) + len(b) - 2, 1))
        d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
        pval = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan
        num_rows.append({"Feature": c, f"Mean {label_a}": a.mean(), f"Mean {label_b}": b.mean(),
                         "Cohen d": d, "p (Mann-Whitney)": pval})
    cat_rows = []
    for c in cat_cols:
        sub = pd.concat([df.loc[mask_a, [c]].assign(g=label_a), df.loc[mask_b, [c]].assign(g=label_b)])
        tab = pd.crosstab(sub[c], sub["g"])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, pval, _, _ = stats.chi2_contingency(tab)
        cramer = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        share = tab.div(tab.sum(axis=0), axis=1)
        diff = (share[label_a] - share[label_b])
        top = diff.abs().idxmax()
        cat_rows.append({"Feature": c, "Cramér V": cramer, "p (chi2)": pval, "Most different level": str(top),
                         f"% {label_a}": 100 * share.loc[top, label_a], f"% {label_b}": 100 * share.loc[top, label_b]})
    num = pd.DataFrame(num_rows).assign(abs_d=lambda d: d["Cohen d"].abs()).sort_values("abs_d", ascending=False).drop(columns="abs_d")
    cat = pd.DataFrame(cat_rows).sort_values("Cramér V", ascending=False) if cat_rows else pd.DataFrame()
    return num.reset_index(drop=True), cat.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 8. Plots
# --------------------------------------------------------------------------- #

def _style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold", "figure.dpi": 110})


_style()


def plot_roc_pr(y, p, thr: float, path=None, label="Tuned CatBoost"):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    fpr, tpr, _ = roc_curve(y, p)
    auc, lo, hi = delong_auc_ci(y, p)
    ax[0].plot(fpr, tpr, color=PALETTE["main"], lw=2, label=f"{label}\nAUC = {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color=PALETTE["grey"], label="Random (AUC = 0.500)")
    m = threshold_metrics(y, p, thr)
    ax[0].scatter([1 - m["Specificity"]], [m["Recall"]], color=PALETTE["alt"], zorder=5, s=55,
                  label=f"Operating point (thr = {thr:.3f})")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate (Recall)", title="ROC curve")
    ax[0].legend(loc="lower right", fontsize=8.5)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    ax[1].plot(rec, prec, color=PALETTE["main"], lw=2, label=f"{label}\nAP = {ap:.3f}")
    ax[1].axhline(y.mean(), ls="--", color=PALETTE["grey"], label=f"Random (prevalence = {y.mean():.3f})")
    ax[1].scatter([m["Recall"]], [m["Precision"]], color=PALETTE["alt"], zorder=5, s=55, label="Operating point")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve", ylim=(0, max(0.6, prec[1:].max() + 0.05)))
    ax[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(y, p, thr: float, path=None):
    c = confusion_counts(y, (np.asarray(p) >= thr).astype(int))
    cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%} of row)", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=11)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred: no readmit", "Pred: <30d readmit"],
           yticklabels=["Actual: no", "Actual: <30d"], title=f"Confusion matrix (threshold = {thr:.3f})")
    fig.tight_layout()
    return _save(fig, path)


def plot_score_distribution(y, p, thr: float, path=None):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, min(0.6, p.max()), 60)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, density=True, color=PALETTE["grey"], label="No <30d readmission")
    ax.hist(p[y == 1], bins=bins, alpha=0.65, density=True, color=PALETTE["alt"], label="Readmitted <30d")
    ax.axvline(thr, color="black", ls="--", label=f"Threshold {thr:.3f}")
    ax.set(xlabel="Predicted probability", ylabel="Density", title="Score distribution by true class")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_threshold_sweep(sweep: pd.DataFrame, thr: float, path=None):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for col, colr in [("Precision", PALETTE["main"]), ("Recall", PALETTE["alt"]), ("F1", PALETTE["green"]), ("F2", PALETTE["gold"])]:
        ax.plot(sweep["Threshold"], sweep[col], label=col, color=colr, lw=2)
    ax.axvline(thr, color="black", ls="--", lw=1, label=f"Chosen thr = {thr:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Precision / Recall / F-score vs. threshold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_cost_curves(curves: pd.DataFrame, summary: pd.DataFrame, path=None):
    ratios = sorted(curves["Cost ratio"].unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(ratios)))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for r, col in zip(ratios, cmap):
        c = curves[curves["Cost ratio"] == r]
        ax.plot(c["Threshold"], c["Cost / patient"], color=col, lw=2, label=f"FN = {r:g} x FP")
        s = summary[summary["FN:FP cost ratio"] == r].iloc[0]
        ax.scatter([s["Optimal thr"]], [s["Cost / patient"]], color=col, edgecolor="black", zorder=5, s=55)
    ax.set(xlabel="Decision threshold", ylabel="Expected cost per patient (units of one FP)",
           title="Cost-sensitive threshold selection (dots = optimum)")
    ax.legend(title="Cost ratio")
    fig.tight_layout()
    return _save(fig, path)


def plot_decision_curve(dc: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(dc["Threshold"], dc["Model"], color=PALETTE["main"], lw=2.2, label="Model")
    ax.plot(dc["Threshold"], dc["Treat all"], color=PALETTE["grey"], lw=1.6, label="Flag everybody")
    ax.axhline(0, color="black", lw=1.2, label="Flag nobody")
    top = dc["Model"].max()
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve analysis",
           ylim=(-0.02, top * 1.6 if top > 0 else 0.05))
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_gain_lift(y, p, path=None):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    frac = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(frac, cum, color=PALETTE["main"], lw=2, label="Model")
    ax[0].plot([0, 1], [0, 1], "--", color=PALETTE["grey"], label="Random")
    for f in (0.1, 0.2, 0.3):
        v = cum[int(f * len(y)) - 1]
        ax[0].scatter([f], [v], color=PALETTE["alt"], zorder=5)
        ax[0].annotate(f"{v:.0%}", (f, v), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax[0].set(xlabel="Share of patients flagged (highest risk first)", ylabel="Share of readmissions caught",
              title="Cumulative gain")
    ax[0].legend()
    dt = decile_table(y, p)
    ax[1].bar(dt["Decile"], dt["Lift"], color=PALETTE["main"])
    ax[1].axhline(1, color="black", ls="--", lw=1)
    ax[1].set(xlabel="Risk decile (1 = highest)", ylabel="Lift vs. average", title="Lift by decile", xticks=range(1, 11))
    fig.tight_layout()
    return _save(fig, path)


def plot_calibration(y, p, n_bins: int = 10, path=None):
    g = calibration_table(y, p, n_bins)
    summ = calibration_summary(y, p, n_bins)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    lim = max(g["mean_pred"].max(), g["obs_rate"].max(), g["ci_high"].max()) * 1.08
    ax[0].plot([0, lim], [0, lim], "--", color=PALETTE["grey"], label="Perfect calibration")
    ax[0].errorbar(g["mean_pred"], g["obs_rate"], yerr=[g["obs_rate"] - g["ci_low"], g["ci_high"] - g["obs_rate"]],
                   fmt="o-", color=PALETTE["main"], capsize=3, label="Model (decile bins, 95% CI)")
    ax[0].set(xlabel="Mean predicted probability", ylabel="Observed readmission rate", xlim=(0, lim), ylim=(0, lim),
              title=f"Calibration (ECE = {summ['ECE']:.3f}, slope = {summ['Cal. slope']:.2f})")
    ax[0].legend(loc="upper left")
    ax[1].hist(np.asarray(p), bins=50, color=PALETTE["main"], alpha=0.8)
    ax[1].set(xlabel="Predicted probability", ylabel="Count", title="Distribution of predictions")
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, metric_cols: Sequence[str], path=None, highlight: str | None = None):
    """Dot-and-whisker plot (honest for truncated axes). `table` has 'Model' and, per metric m,
    columns m, 'm low', 'm high'."""
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5.0 * len(metric_cols), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    t = table.sort_values(metric_cols[0], ascending=True).reset_index(drop=True)
    ypos = np.arange(len(t))
    for ax, m in zip(axes, metric_cols):
        for i, row in t.iterrows():
            colr = PALETTE["alt"] if row["Model"] == highlight else PALETTE["main"]
            ax.errorbar(row[m], i, xerr=[[row[m] - row[f"{m} low"]], [row[f"{m} high"] - row[m]]], fmt="o", color=colr,
                        capsize=3, ms=6, lw=1.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(t["Model"])
        ax.set(title=m)
    fig.tight_layout()
    return _save(fig, path)


def plot_subgroups(rep: pd.DataFrame, feature_label: str, overall: dict, path=None, min_reliable: bool = True):
    """Dot-and-whisker per subgroup: ROC-AUC (DeLong CI), Recall (Wilson CI), Prevalence (Wilson CI)."""
    r = rep[rep["Reliable"]] if min_reliable else rep
    r = r.sort_values("Group").reset_index(drop=True)
    labels = [f"{g}  (n={n:,})" for g, n in zip(r["Group"].astype(str), r["n"])]
    fig, ax = plt.subplots(1, 3, figsize=(14, max(3.4, 0.45 * len(r) + 1.8)), sharey=True)
    spec = [("ROC-AUC", "AUC low", "AUC high", overall["ROC-AUC"]), ("Recall", "Recall low", "Recall high", overall["Recall"]),
            ("Prevalence", "Prev low", "Prev high", overall["Prevalence"])]
    for a, (m, lo, hi, ref) in zip(ax, spec):
        for i, row in r.iterrows():
            lo_v = row[lo] if pd.notna(row[lo]) else row[m]
            hi_v = row[hi] if pd.notna(row[hi]) else row[m]
            a.errorbar(row[m], i, xerr=[[max(row[m] - lo_v, 0)], [max(hi_v - row[m], 0)]], fmt="o", color=PALETTE["main"], capsize=3)
        a.axvline(ref, color=PALETTE["alt"], ls="--", label=f"Overall {ref:.3f}")
        a.set(title=m)
        a.legend(fontsize=8, loc="best")
    ax[0].set_yticks(range(len(r)))
    ax[0].set_yticklabels(labels)
    ax[0].invert_yaxis()
    fig.suptitle(f"Performance by {feature_label} (points = estimate, bars = 95% CI; groups with <500 rows or <50 positives omitted)",
                 fontweight="bold", y=1.02, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_error_rate_by(df: pd.DataFrame, y, pred, col: str, order: list | None = None, path=None, min_n: int = 100):
    """Recall (1 - FN rate among actual positives) and false-alarm rate by levels of `col`."""
    d = pd.DataFrame({"g": df[col].values, "y": np.asarray(y), "pred": np.asarray(pred)})
    pos = d[d["y"] == 1].groupby("g", observed=True).agg(n_pos=("y", "size"), recall=("pred", "mean"))
    neg = d[d["y"] == 0].groupby("g", observed=True).agg(n_neg=("y", "size"), fpr=("pred", "mean"))
    t = pos.join(neg, how="outer")
    t = t[(t["n_pos"].fillna(0) + t["n_neg"].fillna(0)) >= min_n]
    if order:
        t = t.reindex([o for o in order if o in t.index])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["recall"], 0.4, color=PALETTE["green"], label="Recall (caught)")
    ax.bar(x + 0.2, t["fpr"], 0.4, color=PALETTE["alt"], label="False-alarm rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in t.index], rotation=35, ha="right")
    ax.set(title=f"Recall and false-alarm rate by {col}", ylabel="Rate")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)
