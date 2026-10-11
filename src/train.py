"""Phase 4 - Models with designed hyperparameters. Run:  python src/train.py

Trains Random Forest and XGBoost with hand-chosen (non-default) parameters,
refines them with a randomized search, then builds a soft-voting ensemble and a
stacking ensemble. Every fitted model is saved as a pickle file in models/.
"""
import json
import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config
from features import (LOCATION_COLUMNS, MODEL_FEATURES, RAW_NUMERIC_INPUTS,
                      make_preprocessor)

# ==========================================================================
# 1. Designed hyperparameters (my own starting values, none left at default)
#    The data is large (about 118,000 training rows) but the target is noisy:
#    tomorrow's rain can never be known exactly from today's weather. So the
#    goal of every value below is to stop the trees from memorising noise.
# ==========================================================================
RF_DESIGNED = dict(
    n_estimators=150,        # enough trees for a stable average; more trees only add training time
    max_depth=8,             # shallow trees: the main signal (rain today, season) needs few splits
    min_samples_split=100,   # a node needs 100 days before it may split, so it cannot chase single days
    min_samples_leaf=50,     # every leaf averages at least 50 days, which gives smoother probabilities
    max_features="sqrt",     # each split sees a random subset of features, so the trees differ from each other
    bootstrap=True,          # bagging: each tree trains on a random sample of rows
    max_samples=0.5,         # each tree sees half the rows: more variety between trees and faster training
    class_weight="balanced", # rainy days are 66% of the data; this stops the model ignoring dry days
)

XGB_DESIGNED = dict(
    n_estimators=300,        # many small boosting steps ...
    learning_rate=0.05,      # ... each step corrects only 5% of the remaining error (slow, careful learning)
    max_depth=5,             # boosting works best with small trees; depth 5 allows a few feature interactions
    subsample=0.8,           # each tree uses 80% of the rows (adds randomness, reduces overfitting)
    colsample_bytree=0.8,    # each tree uses 80% of the features
    min_child_weight=5,      # a leaf must hold enough days, so one unusual day cannot create a rule
    gamma=0.1,               # a split must reduce the loss by at least 0.1, which prunes weak splits
    reg_alpha=0.1,           # L1 penalty: pushes the weights of useless leaves to zero
    reg_lambda=1.5,          # L2 penalty: shrinks leaf weights, a little stronger than the default of 1
    objective="binary:logistic",  # two classes (rain / no rain), output is a probability
    eval_metric="logloss",   # checks how good the predicted probabilities are
)

# ==========================================================================
# 2. Hand-designed search spaces around the starting values
# ==========================================================================
RF_SEARCH = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [6, 8, 10],
    "model__min_samples_split": [40, 100, 200],
    "model__min_samples_leaf": [20, 50, 100],
    "model__max_features": ["sqrt", 0.5],
    "model__max_samples": [0.3, 0.5, 0.7],
}
XGB_SEARCH = {
    "model__n_estimators": [200, 300, 400],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [3, 4, 5, 6],
    "model__subsample": [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__min_child_weight": [1, 5, 10],
    "model__gamma": [0, 0.1, 0.5],
    "model__reg_alpha": [0, 0.1, 1.0],
    "model__reg_lambda": [1.0, 1.5, 3.0],
}
N_SEARCH_ITERATIONS = 10     # 10 random combinations x 5 folds = 50 fits per model


def make_rf(params):
    return RandomForestClassifier(**params, random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)


def make_xgb(params, scale_pos_weight):
    return XGBClassifier(**params, scale_pos_weight=scale_pos_weight, tree_method="hist",
                         random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)


def with_preprocessing(model):
    """Preprocessing and the model in one pipeline, so it is fitted on training folds only."""
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def tune(name, pipeline, search_space, X, y):
    """Randomized search with time-ordered folds.

    TimeSeriesSplit always trains on earlier days and validates on later days,
    which is how the model is really used. A normal shuffled K-fold would let
    the model "see the future" and report scores that are too optimistic.
    ROC-AUC is the score because it does not depend on one probability cut-off.
    """
    search = RandomizedSearchCV(
        pipeline, search_space, n_iter=N_SEARCH_ITERATIONS, scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=5), random_state=config.RANDOM_STATE, n_jobs=1, refit=True,
    )
    start = time.time()
    search.fit(X, y)
    print(f"{name}: best CV ROC-AUC = {search.best_score_:.4f} "
          f"(search took {time.time() - start:.0f} s)")
    return search


def timed_fit(name, pipeline, X, y):
    start = time.time()
    pipeline.fit(X, y)
    seconds = time.time() - start
    print(f"Fitted {name} in {seconds:.1f} s")
    return seconds


def save_feature_info(train):
    """Everything the app needs to build its input form."""
    cities = (train.groupby("city")[LOCATION_COLUMNS].first().round(4)
              .reset_index().to_dict(orient="records"))
    numeric = {c: {"type": "number", "min": float(train[c].min()), "max": float(train[c].max()),
                   "median": float(train[c].median())} for c in RAW_NUMERIC_INPUTS}
    info = {
        "model_name": "Soft-voting ensemble (Random Forest + XGBoost)",
        "target": config.TARGET,
        "class_labels": {str(k): v for k, v in config.CLASS_LABELS.items()},
        "rain_threshold_mm": config.RAIN_THRESHOLD_MM,
        "raw_input_fields": {"date": {"type": "date"},
                             "city": {"type": "category",
                                      "allowed_values": [c["city"] for c in cities]},
                             **numeric},
        "city_locations": cities,
        "training_period": [str(train["date"].min())[:10], str(train["date"].max())[:10]],
    }
    config.FEATURE_INFO_FILE.write_text(json.dumps(info, indent=2))


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.TRAIN_FILE)        # already sorted by date in preprocess.py
    X, y = train[MODEL_FEATURES], train[config.TARGET]

    # Class imbalance: XGBoost's weight for the positive class = dry days / rainy days.
    # It is below 1 here because the positive class (Rain) is the larger one.
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"Training rows: {len(X):,} | Rain share: {y.mean():.1%} | "
          f"scale_pos_weight: {scale_pos_weight:.3f}")

    # ---- tune Random Forest and XGBoost ---------------------------------
    rf_search = tune("Random Forest", with_preprocessing(make_rf(RF_DESIGNED)), RF_SEARCH, X, y)
    xgb_search = tune("XGBoost", with_preprocessing(make_xgb(XGB_DESIGNED, scale_pos_weight)),
                      XGB_SEARCH, X, y)

    rf_tuned = {**RF_DESIGNED, **{k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}}
    xgb_tuned = {**XGB_DESIGNED, **{k.replace("model__", ""): v for k, v in xgb_search.best_params_.items()}}

    rows = [("Random Forest", k, RF_DESIGNED[k], rf_tuned[k]) for k in RF_DESIGNED]
    rows += [("XGBoost", k, XGB_DESIGNED[k], xgb_tuned[k]) for k in XGB_DESIGNED]
    table = pd.DataFrame(rows, columns=["model", "parameter", "designed", "tuned"])
    table["changed"] = table["designed"].astype(str) != table["tuned"].astype(str)
    print("\nDesigned vs tuned hyperparameters")
    print(table.to_string(index=False))
    table.to_csv(config.REPORTS_DIR / "hyperparameters.csv", index=False)

    # ---- final models ----------------------------------------------------
    times = {}
    rf_model = with_preprocessing(make_rf(rf_tuned))
    xgb_model = with_preprocessing(make_xgb(xgb_tuned, scale_pos_weight))
    times["Random Forest"] = timed_fit("Random Forest", rf_model, X, y)
    times["XGBoost"] = timed_fit("XGBoost", xgb_model, X, y)

    # Soft voting: a FIXED rule. The two models' predicted probabilities are
    # averaged and the class with the higher average wins.
    voting = with_preprocessing(VotingClassifier(
        estimators=[("rf", make_rf(rf_tuned)), ("xgb", make_xgb(xgb_tuned, scale_pos_weight))],
        voting="soft",
    ))
    times["Soft Voting"] = timed_fit("Soft Voting (RF + XGBoost)", voting, X, y)

    # Stacking: the combination is LEARNED. A Logistic Regression meta-learner
    # is trained on the base models' out-of-fold probabilities and decides how
    # much to trust each one.
    # cv note: StackingClassifier needs folds in which every row is predicted
    # exactly once, and TimeSeriesSplit does not do that (its first block is
    # never predicted). KFold WITHOUT shuffling is used instead: 5 blocks of
    # consecutive days, so neighbouring days are never mixed between folds.
    stacking = with_preprocessing(StackingClassifier(
        estimators=[("rf", make_rf(rf_tuned)), ("xgb", make_xgb(xgb_tuned, scale_pos_weight))],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_STATE),
        stack_method="predict_proba",
        cv=KFold(n_splits=5, shuffle=False),
    ))
    times["Stacking"] = timed_fit("Stacking (RF + XGBoost -> Logistic Regression)", stacking, X, y)

    # ---- export as pickle (as shown by the lecturer) --------------------------
    pickle.dump(voting, open(config.MODEL_FILE, "wb"))          # models/model.pkl, used by the app
    pickle.dump(rf_model, open(config.MODELS_DIR / "random_forest.pkl", "wb"))
    pickle.dump(xgb_model, open(config.MODELS_DIR / "xgboost.pkl", "wb"))
    pickle.dump(stacking, open(config.MODELS_DIR / "stacking.pkl", "wb"))
    (config.REPORTS_DIR / "training_times.json").write_text(json.dumps(times, indent=2))
    save_feature_info(train)
    print("\nSaved models:", sorted(p.name for p in config.MODELS_DIR.iterdir()))


if __name__ == "__main__":
    main()
