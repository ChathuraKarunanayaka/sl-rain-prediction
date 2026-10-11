"""Feature engineering shared by training AND the Streamlit app.

Both sides call build_features(), so the model always sees features that were
built in exactly the same way. If this logic lived in two places, the app and
the model could slowly drift apart and give wrong predictions.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# 1. Raw input fields: what a person (or the app form) can provide for "today"
# --------------------------------------------------------------------------
RAW_NUMERIC_INPUTS = [
    "temperature_2m_max",          # deg C, today's maximum
    "temperature_2m_min",          # deg C, today's minimum
    "temperature_2m_mean",         # deg C, today's mean
    "shortwave_radiation_sum",     # MJ/m2, sunshine energy today (low = cloudy)
    "rain_sum",                    # mm, rain today
    "precipitation_hours",         # hours with rain today
    "windspeed_10m_max",           # km/h
    "windgusts_10m_max",           # km/h
    "winddirection_10m_dominant",  # degrees (0 = north, 90 = east)
    "et0_fao_evapotranspiration",  # mm, evaporation demand (high = dry, sunny)
    "rain_yesterday",              # mm
    "rain_2_days_ago",             # mm
    "rain_last_7_days",            # mm, total of the last 7 days including today
]
LOCATION_COLUMNS = ["latitude", "longitude", "elevation"]   # looked up from the city
RAW_INPUT_COLUMNS = ["date", "city"] + RAW_NUMERIC_INPUTS + LOCATION_COLUMNS

# --------------------------------------------------------------------------
# 2. Feature groups used by the model (after build_features)
# --------------------------------------------------------------------------
# Rain amounts and elevation have a long right tail (many small values, a few
# very large ones), so they get a log1p transform before scaling.
SKEWED_FEATURES = [
    "rain_sum", "rain_yesterday", "rain_2_days_ago",
    "rain_last_7_days", "rain_last_3_days_mean", "elevation",
]
NUMERIC_FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "temp_range", "shortwave_radiation_sum", "precipitation_hours",
    "windspeed_10m_max", "windgusts_10m_max", "wind_dir_sin", "wind_dir_cos",
    "et0_fao_evapotranspiration", "latitude", "longitude",
    "month", "day_of_year_sin", "day_of_year_cos",
]
CATEGORICAL_FEATURES = ["city", "monsoon_season"]
MODEL_FEATURES = SKEWED_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def monsoon_season(month: int) -> str:
    """Sri Lanka's four climate seasons (Department of Meteorology definition)."""
    if month in (3, 4):
        return "First inter-monsoon"      # March - April: afternoon thunder showers
    if 5 <= month <= 9:
        return "Southwest monsoon"        # May - September: rain in the south-west
    if month in (10, 11):
        return "Second inter-monsoon"     # October - November: rain island-wide
    return "Northeast monsoon"            # December - February: rain in the north and east


def add_history_columns(df: pd.DataFrame, rain_threshold_mm: float) -> pd.DataFrame:
    """Add past-rain columns and the target, one city at a time.

    Every history column uses ONLY today and earlier days (shift / rolling look
    backwards), so no information from the future leaks into the features.
    The target looks one day forward: did it rain tomorrow?
    """
    df = df.sort_values(["city", "date"]).copy()
    rain = df.groupby("city")["rain_sum"]
    df["rain_yesterday"] = rain.shift(1)
    df["rain_2_days_ago"] = rain.shift(2)
    df["rain_last_7_days"] = rain.transform(lambda s: s.rolling(7).sum())

    rain_next_day = rain.shift(-1)
    df["rain_tomorrow"] = (rain_next_day >= rain_threshold_mm).astype(float)
    df.loc[rain_next_day.isna(), "rain_tomorrow"] = np.nan   # last day: tomorrow is unknown
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw input fields into model features. Used by training and the app."""
    df = df.copy()
    date = pd.to_datetime(df["date"])
    df["month"] = date.dt.month
    df["monsoon_season"] = df["month"].map(monsoon_season)

    # Day of year is a circle: 31 Dec is next to 1 Jan. Sine and cosine keep
    # that closeness, which a plain 1..365 number cannot do.
    angle = 2 * np.pi * date.dt.dayofyear / 365.25
    df["day_of_year_sin"] = np.sin(angle)
    df["day_of_year_cos"] = np.cos(angle)

    # Wind direction is also a circle (359 degrees is next to 0 degrees).
    wind = np.deg2rad(df["winddirection_10m_dominant"].astype(float))
    df["wind_dir_sin"] = np.sin(wind)
    df["wind_dir_cos"] = np.cos(wind)

    # A small day-night temperature gap usually means a cloudy, humid day.
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Short wet spell: average rain over today, yesterday and the day before.
    df["rain_last_3_days_mean"] = (
        df["rain_sum"] + df["rain_yesterday"] + df["rain_2_days_ago"]
    ) / 3.0
    return df


def make_preprocessor() -> ColumnTransformer:
    """Imputation + transformation + scaling + encoding in one object.

    It is fitted on the training data only (inside each model pipeline), so the
    test data and the held-out sample never influence medians, means or categories.

    Tree models (Random Forest, XGBoost) do not need scaling, because they only
    compare "is the value above or below a split point". The scaler is kept so
    that every model receives the same input, and because Logistic Regression
    (the stacking meta-learner and a dashboard model) does need scaled inputs.
    """
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),     # median: not pulled by extreme rain days
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("skewed", skewed, SKEWED_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
