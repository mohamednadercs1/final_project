"""Step 5 - Train, validate and explain the models.

Validation
  * Stratified K-Fold CV on the training period (out-of-fold predictions -> model selection and
    decision thresholds; the test set is never used for tuning).
  * Temporal hold-out: the most recent 20% of films are the test set.
Models: Logistic Regression (baseline), Random Forest, LightGBM, XGBoost.
Metrics: ROC-AUC, PR-AUC, F1, precision, recall, plus a cost-benefit confusion matrix
(a greenlit flop costs COST_FP $M, a passed-on hit costs COST_FN $M).
Greenlight vs +critic comparison on identical films, with a paired-bootstrap CI.
SHAP summary + feature-importance plots for the selected gradient-boosting model.

Run:
    python train_models.py                     # ROI >= 2.5x budget target
    python train_models.py --target revenue    # revenue >= $100M target
    python train_models.py --target tier       # Low / Medium / High (multiclass)
"""
import argparse
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from features import CareerLookup, FeatureBuilder, temporal_split

warnings.filterwarnings("ignore")
TARGET_COLS = {"roi": "success_roi", "revenue": "success_rev", "tier": "tier"}
TARGET_TEXT = {"roi": f"revenue >= {config.ROI_THRESHOLD}x budget",
               "revenue": f"revenue >= ${config.REVENUE_THRESHOLD / 1e6:.0f}M",
               "tier": "ROI tier (Low / Medium / High)"}
GB_MODELS = ("LightGBM", "XGBoost")


# ------------------------------------------------------------------ models
def make_models(y, multiclass=False):
    pos_weight = 1.0 if multiclass else (y == 0).sum() / max((y == 1).sum(), 1)
    xgb_kw = {} if multiclass else {"scale_pos_weight": pos_weight}
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=config.SEED)),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
            random_state=config.SEED, verbose=-1, n_jobs=1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=2, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
            random_state=config.SEED, n_jobs=1, **xgb_kw),
    }


def oof_proba(model, X, y, cv):
    p = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
    return p if p.shape[1] > 2 else p[:, 1]


# ------------------------------------------------------------------ thresholds / costs
def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def total_cost(y, pred, cost_fp, cost_fn):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return fp * cost_fp + fn * cost_fn, (tn, fp, fn, tp)


def best_cost_threshold(y, p, cost_fp, cost_fn):
    grid = np.linspace(0.05, 0.95, 91)
    costs = [total_cost(y, p >= t, cost_fp, cost_fn)[0] for t in grid]
    return float(grid[int(np.argmin(costs))])


def binary_metrics(y, p, thr):
    pred = p >= thr
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "F1": f1_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred)}


# ------------------------------------------------------------------ binary evaluation
def evaluate_binary(models, Xtr, ytr, Xte, yte, cost_fp, cost_fn):
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    rows, cost_rows, fitted, p_test, thresholds = [], [], {}, {}, {}
    naive_all = (yte == 0).sum() * cost_fp          # greenlight everything
    naive_none = (yte == 1).sum() * cost_fn         # greenlight nothing
    naive = min(naive_all, naive_none)

    for name, model in models.items():
        p_oof = oof_proba(model, Xtr, ytr, cv)
        thr_f1 = best_f1_threshold(ytr, p_oof)
        thr_cost = best_cost_threshold(ytr, p_oof, cost_fp, cost_fn)
        fitted[name] = clone(model).fit(Xtr, ytr)
        p = fitted[name].predict_proba(Xte)[:, 1]
        p_test[name], thresholds[name] = p, thr_f1

        rows.append({"model": name,
                     "CV ROC-AUC": roc_auc_score(ytr, p_oof),
                     "CV PR-AUC": average_precision_score(ytr, p_oof),
                     **{f"Test {k}": v for k, v in binary_metrics(yte, p, thr_f1).items()},
                     "threshold": thr_f1})
        cost, (tn, fp, fn, tp) = total_cost(yte, p >= thr_cost, cost_fp, cost_fn)
        cost_rows.append({"model": name, "cost threshold": thr_cost, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "cost ($M)": cost, "saving vs best naive ($M)": naive - cost})
    results = pd.DataFrame(rows).set_index("model")
    costs = pd.DataFrame(cost_rows).set_index("model")
    return results, costs, fitted, p_test, thresholds, {"greenlight all": naive_all, "greenlight none": naive_none}


def plot_curves(yte, p_test, out_dir, tag):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in p_test.items():
        pr, rc, _ = precision_recall_curve(yte, p)
        ax[0].plot(rc, pr, label=f"{name} (AP={average_precision_score(yte, p):.2f})")
        fpr, tpr, _ = roc_curve(yte, p)
        ax[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, p):.2f})")
    ax[0].axhline(yte.mean(), color="grey", ls="--", lw=1); ax[0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=1); ax[1].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
    for a in ax: a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir / f"curves_{tag}.png", dpi=150); plt.close()


def plot_cost_matrix(yte, p, thr, cost_fp, cost_fn, name, out_dir, tag):
    cm = confusion_matrix(yte, p >= thr, labels=[0, 1])
    weights = np.array([[0, cost_fp], [cost_fn, 0]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(weights, cmap="Reds", vmin=0, vmax=max(cost_fp, cost_fn) * 1.4)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n" + (f"${cm[i, j] * weights[i, j]:,.0f}M" if weights[i, j] else "no cost"),
                    ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], ["Pred flop", "Pred hit"]); ax.set_yticks([0, 1], ["Actual flop", "Actual hit"])
    ax.set_title(f"{name}: cost-benefit matrix\n(FP=${cost_fp:g}M, FN=${cost_fn:g}M, thr={thr:.2f})", fontsize=10)
    plt.tight_layout(); plt.savefig(out_dir / f"cost_matrix_{tag}.png", dpi=150); plt.close()


# ------------------------------------------------------------------ critic comparison
def paired_bootstrap_delta(y, p_a, p_b, n=1000, seed=config.SEED):
    """95% CI of PR-AUC(b) - PR-AUC(a) over resampled films."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(average_precision_score(y[idx], p_b[idx]) - average_precision_score(y[idx], p_a[idx]))
    return np.percentile(deltas, [2.5, 97.5])


def critic_comparison(models, best_gb, Xtr, ytr, Xte, yte, gl_cols):
    if "critic_score" not in Xtr or Xtr["critic_score"].notna().sum() < 150 or Xte["critic_score"].notna().sum() < 30:
        print("\nCritic comparison skipped (not enough films with critic scores; run scraper.py).")
        return None
    mtr, mte = Xtr["critic_score"].notna(), Xte["critic_score"].notna()
    y_a, y_b = ytr[mtr.values], yte[mte.values]
    if len(np.unique(y_b)) < 2:
        return None
    rows = []
    for name in ("Logistic Regression", best_gb):
        pa = clone(models[name]).fit(Xtr.loc[mtr, gl_cols], y_a).predict_proba(Xte.loc[mte, gl_cols])[:, 1]
        pb = clone(models[name]).fit(Xtr.loc[mtr, gl_cols + ["critic_score"]], y_a).predict_proba(
            Xte.loc[mte, gl_cols + ["critic_score"]])[:, 1]
        lo, hi = paired_bootstrap_delta(y_b, pa, pb)
        rows += [{"model": name, "features": "greenlight", "ROC-AUC": roc_auc_score(y_b, pa), "PR-AUC": average_precision_score(y_b, pa), "delta PR-AUC [95% CI]": ""},
                 {"model": name, "features": "+ critic score", "ROC-AUC": roc_auc_score(y_b, pb), "PR-AUC": average_precision_score(y_b, pb),
                  "delta PR-AUC [95% CI]": f"{average_precision_score(y_b, pb) - average_precision_score(y_b, pa):+.3f} [{lo:+.3f}, {hi:+.3f}]"}]
    print(f"\nCritic-score comparison on the same {int(mte.sum())} test films (base rate {y_b.mean():.2f}):")
    out = pd.DataFrame(rows).set_index(["model", "features"])
    print(out.round(3).to_string())
    return out


# ------------------------------------------------------------------ explainability
def explain(model, X, name, out_dir, tag):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    for kind, fname in [("dot", f"shap_summary_{tag}.png"), ("bar", f"shap_bar_{tag}.png")]:
        shap.summary_plot(sv, X, plot_type=kind, max_display=15, show=False)
        plt.title(f"SHAP - {name}"); plt.tight_layout(); plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight"); plt.close()
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(15)
    imp.plot.barh(figsize=(6, 5), title=f"Feature importance - {name}"); plt.tight_layout()
    plt.savefig(out_dir / f"feature_importance_{tag}.png", dpi=150); plt.close()
    top = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False).head(8)
    print("\nTop SHAP features (mean |SHAP|):\n" + top.round(3).to_string())


# ------------------------------------------------------------------ multiclass
def run_multiclass(models, Xtr, ytr, Xte, yte, out_dir):
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    rows = []
    for name, model in models.items():
        p_oof = oof_proba(model, Xtr, ytr, cv)
        fit = clone(model).fit(Xtr, ytr)
        p = fit.predict_proba(Xte); pred = p.argmax(1)
        rows.append({"model": name,
                     "CV macro-F1": f1_score(ytr, p_oof.argmax(1), average="macro"),
                     "Test accuracy": accuracy_score(yte, pred),
                     "Test balanced acc": balanced_accuracy_score(yte, pred),
                     "Test macro-F1": f1_score(yte, pred, average="macro"),
                     "Test OvR ROC-AUC": roc_auc_score(yte, p, multi_class="ovr", average="macro"),
                     "Test log-loss": log_loss(yte, p)})
        if name == "LightGBM":
            print("\nLightGBM confusion matrix (rows=actual, cols=predicted; Low/Medium/High):")
            print(pd.DataFrame(confusion_matrix(yte, pred), index=config.TIER_NAMES, columns=config.TIER_NAMES).to_string())
    res = pd.DataFrame(rows).set_index("model")
    res.round(3).to_csv(out_dir / "metrics_tier.csv")
    print("\n" + res.round(3).to_string())


# ------------------------------------------------------------------ deployment artifacts
def export_artifacts(df, target, y_col, best_name, models, thresholds, base_metrics, args):
    """Refit the selected model on ALL films and save everything the Streamlit app needs."""
    builder = FeatureBuilder().fit(df)
    X, y = builder.transform(df), df[y_col].values
    gl = builder.greenlight_features_
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)

    gl_model = clone(models[best_name]).fit(X[gl], y)
    art = {"target": target, "target_text": TARGET_TEXT[target], "model_name": best_name,
           "builder": builder, "lookup": CareerLookup().fit(df), "genres": builder.genres_,
           "base_rate": float(y.mean()), "trained_through": str(df["release_date"].max().date()),
           "greenlight": {"model": gl_model, "features": gl, "threshold": thresholds[best_name]},
           "critic": None, "metrics": base_metrics}

    if "critic_score" in X and X["critic_score"].notna().sum() >= 150:
        m = X["critic_score"].notna().values
        cols = gl + ["critic_score"]
        if len(np.unique(y[m])) == 2:
            p_oof = oof_proba(models[best_name], X.loc[m, cols], y[m], cv)
            art["critic"] = {"model": clone(models[best_name]).fit(X.loc[m, cols], y[m]), "features": cols,
                             "threshold": best_f1_threshold(y[m], p_oof)}
    config.MODELS_DIR.mkdir(exist_ok=True)
    path = config.MODELS_DIR / f"artifacts_{target}.joblib"
    joblib.dump(art, path)
    print(f"\nSaved deployment artifacts -> {path} (critic model: {'yes' if art['critic'] else 'no'})")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=list(TARGET_COLS), default="roi")
    ap.add_argument("--cost-fp", type=float, default=config.COST_FP, help="$M lost per greenlit flop")
    ap.add_argument("--cost-fn", type=float, default=config.COST_FN, help="$M forgone per missed hit")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    out_dir = config.REPORTS_DIR; out_dir.mkdir(exist_ok=True)
    df = pd.read_pickle(config.MODELING_FRAME)
    y_col, multiclass = TARGET_COLS[args.target], args.target == "tier"
    train, test = temporal_split(df)
    builder = FeatureBuilder().fit(train)
    Xtr, Xte = builder.transform(train), builder.transform(test)
    gl_cols = builder.greenlight_features_
    leak = [c for c in Xtr if c in config.POST_RELEASE_COLS or c.startswith("post_")]
    assert not leak, f"post-release columns leaked into features: {leak}"
    ytr, yte = train[y_col].values, test[y_col].values

    print(f"Target: {TARGET_TEXT[args.target]}")
    print(f"Train {len(train)} films ({train.release_date.min().year}-{train.release_date.max().year}) | "
          f"Test {len(test)} films ({test.release_date.min().year}-{test.release_date.max().year})")
    print(f"Class balance  train: {np.bincount(ytr) / len(ytr)}  test: {np.bincount(yte) / len(yte)}")

    models = make_models(ytr, multiclass)
    if multiclass:
        run_multiclass(models, Xtr[gl_cols], ytr, Xte[gl_cols], yte, out_dir)
        print("\n(Deployment artifacts and SHAP are produced for the binary targets only.)")
        return

    res, costs, fitted, p_test, thr, naive = evaluate_binary(
        models, Xtr[gl_cols], ytr, Xte[gl_cols], yte, args.cost_fp, args.cost_fn)
    print(f"\nGreenlight models (temporal test; random-model PR-AUC = {yte.mean():.3f}):")
    print(res.round(3).to_string())
    print(f"\nBusiness cost on test (FP=${args.cost_fp:g}M, FN=${args.cost_fn:g}M; "
          f"naive: greenlight-all ${naive['greenlight all']:,.0f}M, greenlight-none ${naive['greenlight none']:,.0f}M):")
    print(costs.round(2).to_string())

    best_gb = res.loc[list(GB_MODELS), "CV PR-AUC"].idxmax()     # chosen on CV only, never on test
    print(f"\nSelected gradient-boosting model (best CV PR-AUC): {best_gb}")
    tag = args.target
    res.round(4).to_csv(out_dir / f"metrics_{tag}.csv"); costs.round(3).to_csv(out_dir / f"cost_benefit_{tag}.csv")
    plot_curves(yte, p_test, out_dir, tag)
    plot_cost_matrix(yte, p_test[best_gb], thr[best_gb], args.cost_fp, args.cost_fn, best_gb, out_dir, tag)

    critic = critic_comparison(models, best_gb, Xtr, ytr, Xte, yte, gl_cols)
    if critic is not None:
        critic.round(4).to_csv(out_dir / f"critic_comparison_{tag}.csv")
    if not args.no_shap:
        explain(fitted[best_gb], Xte[gl_cols], best_gb, out_dir, tag)

    export_artifacts(df, args.target, y_col, best_gb, models, thr,
                     res.loc[best_gb, ["CV PR-AUC", "Test PR-AUC", "Test ROC-AUC"]].round(3).to_dict(), args)
    print(f"Reports saved in {out_dir}/")


if __name__ == "__main__":
    main()
