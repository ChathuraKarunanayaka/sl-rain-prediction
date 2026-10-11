"""Optional extension - Hybrid ARIMA + Random Forest. Run:  python src/hybrid_arima_rf.py

This is a SEPARATE regression experiment (the main project is classification).
It forecasts the monthly rainfall total of one city in millimetres.

Idea of the hybrid:
  1. SARIMA learns the regular part of the series (the yearly monsoon cycle).
  2. A Random Forest learns what SARIMA got wrong (its residuals) from extra features.
  3. Final forecast = SARIMA forecast + Random Forest residual forecast.

Every forecast is "one month ahead": to predict a month, only data up to the
end of the previous month is used.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
from plot_style import AQUA, BLUE, INK, ORANGE, apply_style

CITY = "Colombo"
# Small dataset (about 115 usable training months), so the forest is kept very simple.
RF_PARAMS = dict(
    n_estimators=300,      # many trees: averaging matters more when data is small
    max_depth=4,           # very shallow trees, otherwise they memorise single months
    min_samples_leaf=5,    # each leaf averages at least 5 months
    max_features=0.5,      # half of the features per split, so trees differ
    random_state=config.RANDOM_STATE,
    n_jobs=config.N_JOBS,
)


def monthly_series():
    df = pd.read_csv(config.RAW_FILE, parse_dates=["time"])
    city = df[df["city"] == CITY].set_index("time")
    monthly = city.resample("MS").agg(rain=("rain_sum", "sum"),
                                      temp=("temperature_2m_mean", "mean"),
                                      radiation=("shortwave_radiation_sum", "mean"),
                                      days=("rain_sum", "size"))
    monthly = monthly[monthly["days"] >= 28]       # drop the last, incomplete month
    return monthly.drop(columns="days")


def make_features(monthly, sarima_forecast):
    """Features for month t that are all known at the end of month t-1."""
    f = pd.DataFrame(index=monthly.index)
    f["month"] = f.index.month
    f["rain_lag_1"] = monthly["rain"].shift(1)
    f["rain_lag_2"] = monthly["rain"].shift(2)
    f["rain_lag_12"] = monthly["rain"].shift(12)                      # same month last year
    f["rain_roll_3"] = monthly["rain"].shift(1).rolling(3).mean()     # last 3 months
    f["temp_lag_1"] = monthly["temp"].shift(1)
    f["radiation_lag_1"] = monthly["radiation"].shift(1)
    f["sarima_forecast"] = sarima_forecast
    return f


def errors(actual, forecast):
    return {"MAE_mm": mean_absolute_error(actual, forecast),
            "RMSE_mm": float(np.sqrt(mean_squared_error(actual, forecast)))}


def main():
    apply_style()
    warnings.filterwarnings("ignore")              # statsmodels prints many convergence notes
    monthly = monthly_series()
    n_train = int(len(monthly) * (1 - config.TEST_FRACTION))
    train, test = monthly.iloc[:n_train], monthly.iloc[n_train:]
    print(f"{CITY} monthly rainfall: {len(monthly)} months | "
          f"train {train.index[0]:%Y-%m} to {train.index[-1]:%Y-%m} | "
          f"test {test.index[0]:%Y-%m} to {test.index[-1]:%Y-%m}")

    # ---- 1. SARIMA on the training months ----------------------------------------------
    # (1,0,1): this month depends on last month and on last month's error.
    # (0,1,1,12): seasonal differencing removes the 12-month monsoon cycle.
    sarima = SARIMAX(train["rain"], order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    fitted_train = sarima.fittedvalues                       # one-step-ahead, inside training
    # append() adds the test months WITHOUT re-estimating the parameters, so each
    # test forecast uses real data up to the previous month but training-only parameters.
    sarima_test = sarima.append(test["rain"], refit=False).get_prediction(
        start=test.index[0], end=test.index[-1]).predicted_mean

    sarima_all = pd.concat([fitted_train, sarima_test])
    features = make_features(monthly, sarima_all)
    residual = monthly["rain"] - sarima_all                  # what SARIMA missed

    # The first 13 months have no lag-12 value and no reliable SARIMA fit: skip them.
    usable = features.dropna().index
    train_idx = usable[(usable <= train.index[-1]) & (usable >= train.index[13])]
    test_idx = test.index

    # ---- 2. Random Forest on SARIMA's residuals -----------------------------------------
    rf_residual = RandomForestRegressor(**RF_PARAMS).fit(features.loc[train_idx], residual.loc[train_idx])
    hybrid = sarima_test + rf_residual.predict(features.loc[test_idx])
    hybrid = hybrid.clip(lower=0)                            # rainfall cannot be negative

    # ---- 3. Plain Random Forest regression for comparison ------------------------------
    plain_cols = [c for c in features.columns if c != "sarima_forecast"]
    rf_plain = RandomForestRegressor(**RF_PARAMS).fit(features.loc[train_idx, plain_cols],
                                                       monthly.loc[train_idx, "rain"])
    plain = pd.Series(rf_plain.predict(features.loc[test_idx, plain_cols]), index=test_idx)

    actual = test["rain"]
    table = pd.DataFrame({
        "SARIMA only": errors(actual, sarima_test.clip(lower=0)),
        "Random Forest only": errors(actual, plain),
        "Hybrid SARIMA + RF": errors(actual, hybrid),
    }).T.round(1)
    print("\nOne-month-ahead forecast errors on the test months")
    print(table.to_string())
    table.to_csv(config.REPORTS_DIR / "hybrid_arima_rf_results.csv")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(actual.index, actual, color=INK, linewidth=2.2, label="Actual")
    ax.plot(sarima_test.index, sarima_test.clip(lower=0), color=BLUE, label="SARIMA only")
    ax.plot(plain.index, plain, color=ORANGE, label="Random Forest only")
    ax.plot(hybrid.index, hybrid, color=AQUA, label="Hybrid SARIMA + RF")
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.set_title(f"{CITY}: monthly rainfall, actual vs one-month-ahead forecasts (test period)")
    ax.legend(ncols=4, loc="upper left"); ax.set_ylim(0, actual.max() * 1.2)
    fig.savefig(config.FIGURES_DIR / "hybrid_arima_rf.png"); plt.close(fig)
    print("Saved reports/hybrid_arima_rf_results.csv and reports/figures/hybrid_arima_rf.png")


if __name__ == "__main__":
    main()
