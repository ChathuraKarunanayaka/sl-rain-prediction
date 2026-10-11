# Prompt used to generate the front end (GenAI-assisted)

The Streamlit front end in `app/app.py` was generated with a GenAI assistant (Claude) from the specification below, then run and tested in a browser.

---

I have trained a machine-learning model in Python that predicts whether a Sri Lankan city will get at least 1 mm of rain tomorrow. Build a Streamlit app in `app/app.py` for it.

**Back end that already exists**

- `models/model.pkl`: a scikit-learn `Pipeline` (preprocessing + soft-voting ensemble of Random Forest and XGBoost), saved with `pickle`. It has `predict` and `predict_proba`.
- `models/feature_info.json`: the raw input field names, their types, min, max and median, the list of cities with latitude, longitude and elevation, the class labels (`0` = No rain, `1` = Rain) and the model name.
- `src/features.py`: `build_features(df)` turns raw inputs into model features (month, monsoon season, day-of-year sine and cosine, wind direction sine and cosine, temperature range, 3-day rain mean). `MODEL_FEATURES` is the list of columns the model expects. The app must call this same function so that training and the app never drift apart.
- `data/processed/inference_sample.csv`: one record that was kept out of training and testing.

**Raw input fields (one widget each)**

- `city`: dropdown of the 30 cities; latitude, longitude and elevation are looked up from the city, not typed
- `date`: date picker (today's date)
- Today's weather, number boxes: `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean` (°C), `shortwave_radiation_sum` (MJ/m²), `rain_sum` (mm), `precipitation_hours`, `windspeed_10m_max`, `windgusts_10m_max` (km/h), `winddirection_10m_dominant` (degrees), `et0_fao_evapotranspiration` (mm)
- Rain history, number boxes: `rain_yesterday`, `rain_2_days_ago`, `rain_last_7_days` (mm)

**Tab 1: "Predict with the saved model"**

1. Pickle loader: load `model.pkl` and `feature_info.json` once with `st.cache_resource`.
2. Header text: what the model predicts, the dataset source, which model is used.
3. An input form with the widgets above, with friendly labels and units. Default values are the training medians.
4. A "Load held-out sample" button that fills the form with `inference_sample.csv`.
5. A "Predict" button. Show the predicted class as a coloured badge (blue for Rain, yellow for No rain), the probability of rain, and a bar chart of the two class probabilities. If the held-out sample was loaded, also show its true label and whether the prediction is correct.

**Tab 2: "Train and compare models" (ensemble dashboard)**

1. Data: either the project's processed rain data (latest 20,000 rows) or a CSV file uploaded by the user. For an uploaded file the user chooses the target column; numeric columns are median-imputed and scaled, text columns are filled with the most frequent value and one-hot encoded, and text columns with more than 50 different values are ignored.
2. A checkbox "rows are in time order": if ticked, the last 20% of rows are the test set (no shuffling); otherwise a stratified random 80/20 split.
3. Sub-tab "Individual models": Decision Tree, Logistic Regression.
4. Sub-tab "Ensemble models": Random Forest (bagging), XGBoost (boosting), Gradient Boosting (boosting), Soft Voting (RF + XGBoost), Stacking (RF + XGBoost with a Logistic Regression meta-learner).
5. Each sub-tab has a model dropdown, sliders for the main hyperparameters (`max_depth`, `n_estimators`, `learning_rate`, or `C`), and a "Train" button. After training show accuracy, macro F1, ROC-AUC, training time, train vs test accuracy and the confusion matrix.
6. Sub-tab "Comparison": a table and a bar chart of every model trained so far on the current data, sorted by ROC-AUC. Clear the results when the data changes.

**Rules**

- Keep the code simple and readable (student level), with short comments.
- Use `RANDOM_STATE = 42` from `src/config.py`.
- Run command: `streamlit run app/app.py` from the project folder.
