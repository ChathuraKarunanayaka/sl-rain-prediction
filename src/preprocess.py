"""Phase 3 - Preprocessing. Run:  python src/preprocess.py

Reads data/raw/, cleans it, builds features, keeps ONE record out for the
inference demo, splits the rest by time and saves everything to data/processed/.
Each step prints what it did.
"""
import numpy as np
import pandas as pd

import config
from features import (LOCATION_COLUMNS, MODEL_FEATURES, RAW_NUMERIC_INPUTS,
                      SKEWED_FEATURES, add_history_columns, build_features)


def step(title):
    print(f"\n=== {title} ===")


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILE)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    step("1. Null value detection")
    nulls = df.isna().sum()
    print(nulls.to_string())
    print(f"Total missing values: {int(nulls.sum())}")
    # This dataset has no missing values. Imputers are still part of the model
    # pipeline (median for numbers, most frequent for categories) so that a blank
    # value in the app or in new data is handled instead of crashing the model.

    step("2. Impossible values check")
    print("Negative rain values      :", int((df["rain_sum"] < 0).sum()))
    print("Min temp above max temp   :", int((df["temperature_2m_min"] > df["temperature_2m_max"]).sum()))
    print("Wind direction outside 0-360:", int((~df["winddirection_10m_dominant"].between(0, 360)).sum()))

    step("3. Data cleaning")
    df["city"] = df["city"].str.strip()                      # fix stray spaces in names
    print("Exact duplicate rows      :", int(df.duplicated().sum()))
    print("Duplicate (city, date)    :", int(df.duplicated(["city", "date"]).sum()))
    df = df.drop_duplicates(["city", "date"])

    # Columns that carry no information are removed:
    #  country           - always "Sri Lanka" (constant)
    #  snowfall_sum      - always 0 (it never snows here)
    #  precipitation_sum - identical to rain_sum because there is no snow
    print("country unique values     :", df["country"].nunique())
    print("snowfall_sum non-zero rows:", int((df["snowfall_sum"] != 0).sum()))
    print("precipitation_sum == rain_sum for all rows:",
          bool((df["precipitation_sum"] == df["rain_sum"]).all()))
    # Columns dropped as repeats of information we already keep:
    #  apparent_temperature_* - "feels like" values calculated from temperature
    #  weathercode            - a code that summarises today's rain amount
    #  sunrise, sunset        - day length changes very little near the equator;
    #                           the season is already described by the date features
    df = df.drop(columns=["country", "snowfall_sum", "precipitation_sum",
                          "apparent_temperature_max", "apparent_temperature_min",
                          "apparent_temperature_mean", "weathercode", "sunrise", "sunset"])
    print("Columns kept:", list(df.columns))

    # Data limitation worth knowing: the weather model stores rain on a coarse
    # grid, so nearby cities can share exactly the same rain series.
    rain_table = df.pivot(index="date", columns="city", values="rain_sum")
    groups = {}
    for city in rain_table.columns:
        groups.setdefault(tuple(rain_table[city].round(2)), []).append(city)
    shared = [g for g in groups.values() if len(g) > 1]
    print(f"Distinct rain series: {len(groups)} for {rain_table.shape[1]} cities")
    for g in shared:
        print("  same rain grid cell:", ", ".join(g))
    return df


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    step("0. Load raw data")
    df = load_raw()
    print("Shape:", df.shape, "| cities:", df["city"].nunique(),
          "| dates:", df["date"].min().date(), "to", df["date"].max().date())

    df = clean(df)

    step("4. Feature engineering (past values only)")
    df = add_history_columns(df, config.RAIN_THRESHOLD_MM)
    before = len(df)
    # The first 6 days of each city have no 7-day history, and the last day of
    # each city has no "tomorrow". These rows cannot be used and are removed.
    df = df.dropna(subset=["rain_last_7_days", config.TARGET])
    df[config.TARGET] = df[config.TARGET].astype(int)
    print(f"Rows without full history or without a next day removed: {before - len(df)}")
    df = build_features(df)
    print("Model features:", MODEL_FEATURES)

    step("5. Transformation check: skewness before and after log1p")
    skew = pd.DataFrame({
        "skew_before": df[SKEWED_FEATURES].skew(),
        "skew_after_log1p": np.log1p(df[SKEWED_FEATURES]).skew(),
    }).round(2)
    print(skew.to_string())
    skew.to_csv(config.REPORTS_DIR / "skewness_before_after.csv")
    # The log1p transform itself is applied inside the model pipeline
    # (features.make_preprocessor), followed by StandardScaler and one-hot encoding.

    step("6. Single-sample hold-out (never used for training or testing)")
    last_date = df.loc[df["city"] == config.HOLDOUT_CITY, "date"].max()
    is_holdout = (df["city"] == config.HOLDOUT_CITY) & (df["date"] == last_date)
    holdout = df[is_holdout]
    df = df[~is_holdout]
    print(f"Held out: {config.HOLDOUT_CITY}, {last_date.date()} "
          f"(true label = {int(holdout[config.TARGET].iloc[0])}) -> {config.INFERENCE_SAMPLE_FILE.name}")

    step("7. Chronological train / test split (no shuffling)")
    dates = np.sort(df["date"].unique())
    cut_date = dates[int(len(dates) * (1 - config.TEST_FRACTION))]
    df = df.sort_values(["date", "city"])           # time order, needed for TimeSeriesSplit
    train = df[df["date"] < cut_date]
    test = df[df["date"] >= cut_date]
    print(f"Train: {len(train):,} rows, {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Test : {len(test):,} rows, {test['date'].min().date()} to {test['date'].max().date()}")

    step("8. Class balance")
    for name, part in [("train", train), ("test", test)]:
        share = part[config.TARGET].mean()
        print(f"{name}: Rain = {share:.1%}, No rain = {1 - share:.1%}")
    # About two thirds of days are rainy, so the classes are moderately
    # imbalanced. This is handled in train.py with class weights.

    columns = ["date", "city"] + RAW_NUMERIC_INPUTS + LOCATION_COLUMNS + [
        c for c in MODEL_FEATURES if c not in RAW_NUMERIC_INPUTS + LOCATION_COLUMNS + ["city"]
    ] + [config.TARGET]
    # Rounding keeps the files small; it does not change any prediction.
    for part, path in [(train, config.TRAIN_FILE), (test, config.TEST_FILE),
                       (holdout, config.INFERENCE_SAMPLE_FILE)]:
        out = part[columns].copy()
        numeric = out.select_dtypes("number").columns
        out[numeric] = out[numeric].round(4)
        out.to_csv(path, index=False)
    print(f"\nSaved {config.TRAIN_FILE.name}, {config.TEST_FILE.name} and "
          f"{config.INFERENCE_SAMPLE_FILE.name} in data/processed/")


if __name__ == "__main__":
    main()
