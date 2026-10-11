"""Phase 7 - Front end. Run from the project folder:  streamlit run app/app.py

Tab 1: pickle loader + input form -> prediction from the saved back-end model.
Tab 2: small ensemble dashboard -> upload data, pick a model, train it, compare results.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))           # so the app can import the training code

import config                                    # noqa: E402
from features import (MODEL_FEATURES, RAW_NUMERIC_INPUTS, build_features,   # noqa: E402
                      make_preprocessor)
from plot_style import INK, apply_style          # noqa: E402

st.set_page_config(page_title="Sri Lanka Rain Prediction", page_icon="🌧️", layout="wide")
apply_style()

# Friendly labels for the input form: (label, step)
FIELD_LABELS = {
    "temperature_2m_max": ("Maximum temperature today (°C)", 0.1),
    "temperature_2m_min": ("Minimum temperature today (°C)", 0.1),
    "temperature_2m_mean": ("Mean temperature today (°C)", 0.1),
    "shortwave_radiation_sum": ("Sunshine energy today (MJ/m²)", 0.1),
    "rain_sum": ("Rain today (mm)", 0.1),
    "precipitation_hours": ("Hours with rain today", 1.0),
    "windspeed_10m_max": ("Maximum wind speed (km/h)", 0.1),
    "windgusts_10m_max": ("Maximum wind gust (km/h)", 0.1),
    "winddirection_10m_dominant": ("Wind direction (degrees, 0 = north)", 1.0),
    "et0_fao_evapotranspiration": ("Evapotranspiration ET0 (mm)", 0.01),
    "rain_yesterday": ("Rain yesterday (mm)", 0.1),
    "rain_2_days_ago": ("Rain 2 days ago (mm)", 0.1),
    "rain_last_7_days": ("Total rain in the last 7 days, including today (mm)", 0.1),
}
TODAY_FIELDS = RAW_NUMERIC_INPUTS[:10]
HISTORY_FIELDS = RAW_NUMERIC_INPUTS[10:]


# ==========================================================================
# Pickle loader (runs once, then Streamlit keeps the result in memory)
# ==========================================================================
@st.cache_resource
def load_model_and_info():
    with open(config.MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    info = json.loads(config.FEATURE_INFO_FILE.read_text())
    return model, info


def set_form_defaults(info):
    """First visit: fill the form with the training medians."""
    st.session_state.setdefault("city", "Colombo")
    st.session_state.setdefault("date", pd.Timestamp("2023-06-16").date())
    for field in RAW_NUMERIC_INPUTS:
        st.session_state.setdefault(field, float(info["raw_input_fields"][field]["median"]))
    st.session_state.setdefault("true_label", None)


def load_holdout_sample():
    """Button callback: copy the held-out record into the form."""
    sample = pd.read_csv(config.INFERENCE_SAMPLE_FILE).iloc[0]
    st.session_state["city"] = sample["city"]
    st.session_state["date"] = pd.Timestamp(sample["date"]).date()
    for field in RAW_NUMERIC_INPUTS:
        st.session_state[field] = float(sample[field])
    st.session_state["true_label"] = int(sample[config.TARGET])


def prediction_tab():
    model, info = load_model_and_info()
    set_form_defaults(info)
    locations = {c["city"]: c for c in info["city_locations"]}

    st.subheader("Will it rain tomorrow?")
    st.markdown(
        f"This page predicts whether a Sri Lankan city gets **at least "
        f"{info['rain_threshold_mm']:.0f} mm of rain tomorrow**, from today's weather and the "
        f"rain of the last few days.  \n"
        f"**Model:** {info['model_name']}, loaded from `models/model.pkl`.  \n"
        f"**Data:** Sri Lanka Weather Dataset (Kaggle, rasulmah; source Open-Meteo), 30 cities, "
        f"trained on {info['training_period'][0]} to {info['training_period'][1]}."
    )
    st.button("Load held-out sample", on_click=load_holdout_sample,
              help="Fills the form with the one record that was kept out of training and testing.")

    with st.form("inputs"):
        left, right = st.columns(2)
        left.selectbox("City", sorted(locations), key="city")
        right.date_input("Today's date", key="date")

        st.markdown("**Today's weather**")
        cols = st.columns(2)
        for i, field in enumerate(TODAY_FIELDS):
            label, step = FIELD_LABELS[field]
            cols[i % 2].number_input(label, min_value=0.0, step=step, key=field)

        st.markdown("**Rain in the last days**")
        cols = st.columns(3)
        for i, field in enumerate(HISTORY_FIELDS):
            label, step = FIELD_LABELS[field]
            cols[i].number_input(label, min_value=0.0, step=step, key=field)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    # Build one raw row, then create features with the SAME function used in training.
    raw = {"date": pd.Timestamp(st.session_state["date"]), "city": st.session_state["city"]}
    raw.update({f: st.session_state[f] for f in RAW_NUMERIC_INPUTS})
    raw.update({k: locations[raw["city"]][k] for k in ("latitude", "longitude", "elevation")})
    features = build_features(pd.DataFrame([raw]))[MODEL_FEATURES]

    predicted = int(model.predict(features)[0])
    prob = model.predict_proba(features)[0]
    label = info["class_labels"][str(predicted)]
    color = "#2a78d6" if predicted == 1 else "#eda100"      # blue = rain, yellow = dry
    icon = "🌧️" if predicted == 1 else "☀️"

    result, chart = st.columns([1, 1.4])
    result.markdown(
        f"<div style='background:{color};color:white;padding:14px 18px;border-radius:10px;"
        f"font-size:1.5rem;font-weight:700;display:inline-block'>{icon} Tomorrow: {label}</div>",
        unsafe_allow_html=True)
    result.metric("Probability of rain", f"{prob[1]:.1%}")
    result.caption(f"Season used by the model: {features['monsoon_season'].iloc[0]}")
    true_label = st.session_state.get("true_label")
    if true_label is not None:
        truth = info["class_labels"][str(true_label)]
        verdict = "correct ✅" if true_label == predicted else "incorrect ❌"
        result.info(f"Held-out sample: the true answer was **{truth}**, so the prediction is {verdict}. "
                    "This only applies if you did not change the loaded values.")

    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    names = [info["class_labels"]["0"], info["class_labels"]["1"]]
    ax.barh(names, prob, color=["#eda100", "#2a78d6"], height=0.55)
    for y, v in enumerate(prob):
        ax.text(v + 0.01, y, f"{v:.1%}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Predicted probability"); ax.grid(axis="y", visible=False)
    ax.set_title("Class probabilities")
    chart.pyplot(fig)
    plt.close(fig)


# ==========================================================================
# Dashboard: upload data -> choose model -> train -> compare
# ==========================================================================
INDIVIDUAL_MODELS = ["Decision Tree", "Logistic Regression"]
ENSEMBLE_MODELS = ["Random Forest (bagging)", "XGBoost (boosting)", "Gradient Boosting (boosting)",
                   "Soft Voting (RF + XGBoost)", "Stacking (RF + XGBoost -> Logistic Regression)"]


@st.cache_data
def load_project_sample(rows=20000):
    """The most recent rows of the project's processed data (kept small so training is quick)."""
    data = pd.concat([pd.read_csv(config.TRAIN_FILE), pd.read_csv(config.TEST_FILE)])
    return data.sort_values("date").tail(rows)[MODEL_FEATURES + [config.TARGET]].reset_index(drop=True)


def generic_preprocessor(X):
    """For an uploaded file: numbers are imputed + scaled, text columns are imputed + one-hot encoded."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = X.select_dtypes("number").columns.tolist()
    text = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), text),
    ])


def build_model(name, p):
    """Create the chosen model with the parameters from the sliders (no defaults left to chance)."""
    from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                                  StackingClassifier, VotingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    seed = config.RANDOM_STATE

    def rf():
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=20, max_features="sqrt", random_state=seed, n_jobs=-1)

    def xgb():
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=min(p["max_depth"], 6),
                             learning_rate=p["learning_rate"], subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, tree_method="hist", random_state=seed, n_jobs=-1)

    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=20, random_state=seed)
    if name == "Logistic Regression":
        return LogisticRegression(C=p["C"], max_iter=1000, random_state=seed)
    if name.startswith("Random Forest"):
        return rf()
    if name.startswith("XGBoost"):
        return xgb()
    if name.startswith("Gradient Boosting"):
        return GradientBoostingClassifier(n_estimators=p["n_estimators"], max_depth=min(p["max_depth"], 5),
                                          learning_rate=p["learning_rate"], subsample=0.8, random_state=seed)
    if name.startswith("Soft Voting"):
        return VotingClassifier([("rf", rf()), ("xgb", xgb())], voting="soft")
    return StackingClassifier([("rf", rf()), ("xgb", xgb())],
                              final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
                              cv=KFold(n_splits=3, shuffle=False))


def parameter_controls(name, key):
    """Sliders for the main hyperparameters of the chosen model."""
    p = {}
    cols = st.columns(3)
    if name == "Logistic Regression":
        p["C"] = cols[0].select_slider("C (smaller = stronger regularisation)",
                                       [0.01, 0.1, 1.0, 10.0], value=1.0, key=f"{key}_C")
        return p
    p["max_depth"] = cols[0].slider("max_depth", 2, 12, 6, key=f"{key}_depth")
    if name != "Decision Tree":
        p["n_estimators"] = cols[1].slider("n_estimators", 50, 300, 100, step=50, key=f"{key}_trees")
        p["learning_rate"] = cols[2].select_slider("learning_rate (boosting only)",
                                                   [0.03, 0.05, 0.1, 0.3], value=0.1, key=f"{key}_lr")
    return p


def train_and_score(name, params, X, y, classes, time_ordered, project_data):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    if time_ordered:                       # earlier rows train, the last 20% test
        cut = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)

    prep = make_preprocessor() if project_data else generic_preprocessor(X)
    pipeline = Pipeline([("prep", prep), ("model", build_model(name, params))])
    start = time.time()
    pipeline.fit(X_train, y_train)
    seconds = time.time() - start

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)
    auc = (roc_auc_score(y_test, prob[:, 1]) if len(classes) == 2
           else roc_auc_score(y_test, prob, multi_class="ovr"))
    return {
        "Model": name, "Accuracy": accuracy_score(y_test, pred),
        "Macro F1": f1_score(y_test, pred, average="macro"), "ROC-AUC": auc,
        "Train accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
        "Training time (s)": seconds, "Test rows": len(y_test),
        "confusion": confusion_matrix(y_test, pred), "params": params,
    }


def show_result(result, classes):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{result['Accuracy']:.3f}")
    c2.metric("Macro F1", f"{result['Macro F1']:.3f}")
    c3.metric("ROC-AUC", f"{result['ROC-AUC']:.3f}")
    c4.metric("Training time", f"{result['Training time (s)']:.1f} s")
    st.caption(f"Train accuracy {result['Train accuracy']:.3f} vs test accuracy {result['Accuracy']:.3f}. "
               "A large gap means the model is overfitting.")
    cm = pd.DataFrame(result["confusion"], index=[f"Actual: {c}" for c in classes],
                      columns=[f"Predicted: {c}" for c in classes])
    st.markdown("**Confusion matrix (test rows)**")
    st.dataframe(cm, width="content")


def model_panel(group_key, model_names, X, y, classes, time_ordered, project_data):
    name = st.selectbox("Model", model_names, key=f"{group_key}_model")
    params = parameter_controls(name, group_key)
    if st.button(f"Train {name}", key=f"{group_key}_train", type="primary"):
        with st.spinner(f"Training {name} ..."):
            result = train_and_score(name, params, X, y, classes, time_ordered, project_data)
        st.session_state["results"][name] = result
        st.session_state[f"{group_key}_last"] = name
    last = st.session_state.get(f"{group_key}_last")
    if last in st.session_state["results"]:
        st.markdown(f"#### Result: {last}")
        show_result(st.session_state["results"][last], classes)


def dashboard_tab():
    from sklearn.preprocessing import LabelEncoder
    st.subheader("Ensemble model dashboard")
    st.markdown("Upload a data file (or use the project's rain data), choose a model, train it, "
                "and compare the results of every model you trained.")
    st.session_state.setdefault("results", {})

    source = st.radio("Data", ["Project data: Sri Lanka rain (latest 20,000 rows)", "Upload a CSV file"],
                      horizontal=True)
    project_data = source.startswith("Project")
    if project_data:
        data, target = load_project_sample(), config.TARGET
    else:
        file = st.file_uploader("CSV file with one row per example", type="csv")
        if file is None:
            st.info("Upload a CSV file to continue.")
            return
        data = pd.read_csv(file)
        target = st.selectbox("Target column (the class to predict)", data.columns,
                              index=len(data.columns) - 1)
        if data[target].nunique() > 10:
            st.error("This dashboard is for classification. The target has more than 10 different "
                     "values, so please choose a column with a few classes.")
            return
        data = data.dropna(subset=[target])
        # Text columns with very many different values (IDs, free text, dates) cannot be one-hot encoded.
        too_many = [c for c in data.columns if c != target
                    and not pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() > 50]
        if too_many:
            st.warning(f"Ignored text columns with more than 50 different values: {', '.join(too_many)}")
            data = data.drop(columns=too_many)

    time_ordered = st.checkbox("Rows are in time order (use the last 20% as the test set, no shuffling)",
                               value=project_data)
    data_id = (source, target, len(data), time_ordered)
    if st.session_state.get("data_id") != data_id:        # new data: old results no longer comparable
        st.session_state["results"] = {}
        st.session_state["data_id"] = data_id

    X = data.drop(columns=[target])
    encoder = LabelEncoder().fit(data[target])
    y = encoder.transform(data[target])
    classes = ([config.CLASS_LABELS[c] for c in encoder.classes_] if project_data
               else [str(c) for c in encoder.classes_])
    st.caption(f"{len(data):,} rows, {X.shape[1]} input columns, target `{target}` "
               f"with classes: {', '.join(classes)}")
    with st.expander("Preview of the data"):
        st.dataframe(data.head(20))

    individual, ensemble, comparison = st.tabs(["Individual models", "Ensemble models", "Comparison"])
    with individual:
        model_panel("individual", INDIVIDUAL_MODELS, X, y, classes, time_ordered, project_data)
    with ensemble:
        model_panel("ensemble", ENSEMBLE_MODELS, X, y, classes, time_ordered, project_data)
    with comparison:
        results = st.session_state["results"]
        if not results:
            st.info("Train at least one model in the other tabs to see the comparison.")
            return
        table = (pd.DataFrame(results.values())
                 .drop(columns=["confusion", "params"]).set_index("Model")
                 .sort_values("ROC-AUC", ascending=False))
        st.dataframe(
            table.reset_index().style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "ROC-AUC": "{:.3f}",
                                              "Train accuracy": "{:.3f}", "Training time (s)": "{:.1f}"}),
            hide_index=True, column_config={"Model": st.column_config.TextColumn(width="large")})
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.5 * len(table) + 1.6), sharey=True)
        for ax, metric in zip(axes, ["Accuracy", "Macro F1", "ROC-AUC"]):
            ax.barh(table.index, table[metric], color="#2a78d6", height=0.6)
            for y_pos, v in enumerate(table[metric]):
                ax.text(v + 0.005, y_pos, f"{v:.3f}", va="center", fontsize=9, color=INK)
            ax.set_xlim(0, 1.12); ax.set_title(metric); ax.grid(axis="y", visible=False)
            ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Best ROC-AUC so far: {table.index[0]}. All models were tested on the same rows.")


st.title("🌧️ Sri Lanka Rain Prediction")
st.caption("MSc Applied Machine Learning class activity: Random Forest vs XGBoost ensembles on local weather data")
tab_predict, tab_dashboard = st.tabs(["Predict with the saved model", "Train and compare models"])
with tab_predict:
    prediction_tab()
with tab_dashboard:
    dashboard_tab()
