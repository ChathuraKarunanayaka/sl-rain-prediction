"""Phase 5 - Evaluation and comparison. Run:  python src/evaluate.py

Scores Random Forest, XGBoost, Soft Voting and Stacking on the SAME time-ordered
test set, and saves a comparison table and figures to reports/.
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import config
from features import MODEL_FEATURES
from plot_style import BLUE, GREY, INK, MODEL_COLORS, ORANGE, apply_style

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "Soft Voting": "model.pkl",
    "Stacking": "stacking.pkl",
}
BASELINE = "Baseline (same as today)"


def score(y_true, y_pred, y_prob=None):
    """All metrics for one model. 'macro' treats both classes equally,
    'weighted' gives the larger class (Rain) more weight."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg)
        out[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg)
    out["roc_auc"] = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return out


def clean_feature_name(name):
    return (name.replace("skewed__", "").replace("numeric__", "").replace("categorical__", "")
            .replace("city_", "city: ").replace("monsoon_season_", "season: "))


def main():
    apply_style()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.TRAIN_FILE)
    test = pd.read_csv(config.TEST_FILE)
    X_train, y_train = train[MODEL_FEATURES], train[config.TARGET]
    X_test, y_test = test[MODEL_FEATURES], test[config.TARGET]
    times = json.loads((config.REPORTS_DIR / "training_times.json").read_text())
    models = {name: pickle.load(open(config.MODELS_DIR / f, "rb")) for name, f in MODEL_FILES.items()}

    rows, probs, preds = [], {}, {}
    # Baseline with no machine learning: "tomorrow will be the same as today".
    # A model is only useful if it beats this simple rule.
    preds[BASELINE] = (test["rain_sum"] >= config.RAIN_THRESHOLD_MM).astype(int).to_numpy()
    base_train = (train["rain_sum"] >= config.RAIN_THRESHOLD_MM).astype(int)
    rows.append({"model": BASELINE, **score(y_test, preds[BASELINE]),
                 "train_accuracy": accuracy_score(y_train, base_train), "training_time_s": 0.0})

    for name, model in models.items():
        probs[name] = model.predict_proba(X_test)[:, 1]
        preds[name] = (probs[name] >= 0.5).astype(int)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        rows.append({"model": name, **score(y_test, preds[name], probs[name]),
                     "train_accuracy": train_acc, "training_time_s": times[name]})

    table = pd.DataFrame(rows).set_index("model")
    # A large gap between train and test accuracy means overfitting.
    table["train_minus_test_accuracy"] = table["train_accuracy"] - table["accuracy"]
    table = table.round(4)
    print(table.T.to_string())
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv")

    # ---- Figure 1: confusion matrices ----------------------------------------
    labels = [config.CLASS_LABELS[0], config.CLASS_LABELS[1]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, name in zip(axes, models):
        cm = confusion_matrix(y_test, preds[name])
        share = cm / cm.sum(axis=1, keepdims=True)          # row share = recall per class
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center",
                        color="white" if share[i, j] > 0.55 else INK, fontsize=10)
        ax.set_xticks([0, 1], labels)
        ax.set_xlabel("Predicted"); ax.set_title(name); ax.grid(False)
    axes[0].set_yticks([0, 1], labels)
    axes[0].set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the test set (count and share of each actual class)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(config.FIGURES_DIR / "confusion_matrices.png"); plt.close(fig)

    # ---- Figure 2: ROC curves ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        ax.plot(fpr, tpr, color=MODEL_COLORS[name],
                label=f"{name} (AUC = {table.loc[name, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False positive rate (dry days called rainy)")
    ax.set_ylabel("True positive rate (rainy days found)")
    ax.set_title("ROC curves on the test set"); ax.legend(loc="lower right")
    fig.savefig(config.FIGURES_DIR / "roc_curves.png"); plt.close(fig)

    # ---- Figure 3: feature importances -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    importance_rows = []
    for ax, name, color in [(axes[0], "Random Forest", BLUE), (axes[1], "XGBoost", ORANGE)]:
        pipe = models[name]
        names = [clean_feature_name(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
        imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values()
        importance_rows.append(imp.rename(name))
        top = imp.tail(12)
        ax.barh(top.index, top.values, color=color, height=0.65)
        for y_pos, v in enumerate(top.values):
            ax.text(v, y_pos, f" {v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(f"{name}: top 12 features"); ax.set_xlabel("Importance (share of total)")
        ax.grid(axis="y", visible=False); ax.set_xlim(0, top.max() * 1.15)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png"); plt.close(fig)
    pd.concat(importance_rows, axis=1).sort_values("XGBoost", ascending=False).round(4).to_csv(
        config.REPORTS_DIR / "feature_importance.csv")

    # ---- Figure 4: metric comparison ------------------------------------------------
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"), ("roc_auc", "ROC-AUC")]
    order = list(models) + [BASELINE]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        values = table.loc[order, col]
        ax.barh(order, values.fillna(0), color=[MODEL_COLORS[m] for m in order], height=0.62)
        for y_pos, v in enumerate(values):
            ax.text((v if pd.notna(v) else 0.5) + 0.004, y_pos,
                    f"{v:.3f}" if pd.notna(v) else "not available", va="center", fontsize=9, color=INK)
        ax.set_xlim(0.5, 0.95); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_title(title); ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.suptitle("Model comparison on the test set (axis starts at 0.5 to show the small gaps)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "metric_comparison.png"); plt.close(fig)

    # ---- Where do the errors happen? Accuracy by season and by city ---------------
    result = test[["date", "city", "monsoon_season", config.TARGET]].copy()
    for name in order:
        result[name] = (preds[name] == y_test.to_numpy()).astype(int)
    by_season = result.groupby("monsoon_season")[order].mean()
    by_season["rain_share"] = result.groupby("monsoon_season")[config.TARGET].mean()
    by_season["test_rows"] = result.groupby("monsoon_season").size()
    by_city = result.groupby("city")[order].mean()
    by_city["rain_share"] = result.groupby("city")[config.TARGET].mean()
    by_season.round(4).to_csv(config.REPORTS_DIR / "accuracy_by_season.csv")
    by_city.round(4).sort_values("Soft Voting").to_csv(config.REPORTS_DIR / "accuracy_by_city.csv")
    print("\nAccuracy by monsoon season\n", by_season.round(3).to_string())
    print("\nLowest and highest accuracy by city (Soft Voting)\n",
          by_city["Soft Voting"].sort_values().round(3).iloc[[0, 1, 2, -3, -2, -1]].to_string())

    seasons = ["First inter-monsoon", "Southwest monsoon", "Second inter-monsoon", "Northeast monsoon"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.38
    x = np.arange(len(seasons))
    for k, name in enumerate(["Soft Voting", BASELINE]):
        vals = by_season.loc[seasons, name]
        ax.bar(x + (k - 0.5) * width, vals, width=width - 0.04, color=MODEL_COLORS[name], label=name)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * width, v + 0.006, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [s.replace(" monsoon", "\nmonsoon").replace("inter-\nmonsoon", "inter-monsoon")
                      for s in seasons])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Accuracy"); ax.grid(axis="x", visible=False)
    ax.set_title("Test accuracy by monsoon season"); ax.legend(loc="upper left", ncols=2)
    fig.savefig(config.FIGURES_DIR / "accuracy_by_season.png"); plt.close(fig)

    print("\nSaved reports/model_comparison.csv and figures in reports/figures/")


if __name__ == "__main__":
    main()
