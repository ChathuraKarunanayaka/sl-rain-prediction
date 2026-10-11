# Results: Random Forest vs XGBoost vs ensembles

Prediction task: will a Sri Lankan city get at least 1 mm of rain tomorrow?
Test set: 29,459 city-days from 8 October 2020 to 16 June 2023 (the latest 20% of dates, never used for training or tuning).

Numbers come from `python src/evaluate.py` (`reports/model_comparison.csv`), run on Windows with the models in `models/`. XGBoost gives slightly different results on other operating systems (third decimal place).

## 1. Comparison table

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC | Train accuracy | Train minus test | Training time |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: tomorrow = today | 0.8033 | 0.7652 | 0.7653 | 0.7653 | 0.8033 | not available | 0.8179 | 0.0146 | 0 s |
| Random Forest | 0.7965 | 0.7609 | 0.7882 | 0.7702 | 0.8015 | 0.8732 | 0.8325 | 0.0361 | 5.0 s |
| XGBoost | 0.7996 | 0.7634 | 0.7879 | 0.7723 | 0.8041 | 0.8753 | 0.8268 | 0.0271 | 2.7 s |
| Soft Voting (RF + XGBoost) | 0.7985 | 0.7626 | 0.7886 | 0.7717 | 0.8032 | 0.8750 | 0.8298 | 0.0313 | 7.3 s |
| Stacking (RF + XGBoost, Logistic Regression) | **0.8154** | **0.7804** | 0.7747 | **0.7774** | **0.8145** | 0.8750 | 0.8392 | 0.0238 | 30.6 s |

The baseline uses no machine learning: it says "rain tomorrow" if it rained today. It has no probability output, so it has no ROC-AUC.

Figures: `figures/metric_comparison.png`, `figures/confusion_matrices.png`, `figures/roc_curves.png`, `figures/feature_importance.png`, `figures/accuracy_by_season.png`.

## 2. What the numbers say

1. **All four models rank days well (ROC-AUC about 0.875).**
2. **On plain accuracy they are about equal to the baseline (0.80).** Rain is very persistent in Sri Lanka: 86% of rainy days are followed by another rainy day. A yes/no rule based on today is hard to beat.
3. **The models are better at treating both classes fairly.** Macro F1 is 0.770 to 0.777 against 0.765 for the baseline, and macro recall is 0.775 to 0.789 against 0.765.
4. **The models give a probability, the baseline does not.** A probability lets a user rank days by risk or choose a stricter cut-off.
5. **No serious overfitting.** Train accuracy is only 2.4 to 3.6 points above test accuracy.

## 3. Strengths and limitations of each model

### Random Forest (bagging)

- **General strengths:** averages many trees built on random samples, which reduces variance; hard to overfit; few sensitive parameters; trees train in parallel.
- **General limitations:** larger and slower than boosting; cannot correct a shared bias of its trees because every tree is built independently.
- **Observed here:** accuracy 0.7965, macro F1 0.7702, ROC-AUC 0.8732. It puts about 70% of its importance on two features (rain today 0.40, rain hours today 0.29), so it behaves close to the baseline. Slowest single model (5.0 s) and largest file (3.8 MB).

### XGBoost (boosting)

- **General strengths:** each new tree corrects the errors of the earlier ones, which reduces bias; built-in regularisation (`gamma`, L1, L2); fast histogram training; one parameter for class imbalance.
- **General limitations:** many parameters that interact; a high learning rate or deep trees overfit quickly.
- **Observed here:** best single model on accuracy (0.7996), macro F1 (0.7723) and ROC-AUC (0.8753); its macro recall is practically equal to Random Forest's. It trains about 2 times faster (2.7 s) and its file is about 10 times smaller (0.35 MB). It uses the season more than Random Forest does: "Northeast monsoon" is its third most important feature.

### Soft Voting (RF + XGBoost)

- **General strengths:** simple fixed rule (average the probabilities); more stable than one model; nothing extra to train.
- **General limitations:** helps only when the members make different mistakes; costs the training time of all members.
- **Observed here:** lands between its two members (accuracy 0.7985, ROC-AUC 0.8750). Random Forest and XGBoost use the same features and make very similar errors, so averaging adds little.

### Stacking (RF + XGBoost with a Logistic Regression meta-learner)

- **General strengths:** learns how much to trust each base model instead of using a fixed rule.
- **General limitations:** most expensive (base models are trained 6 times: 5 folds plus the final fit); harder to explain; the meta-learner can overfit if the folds leak.
- **Observed here:** best accuracy (0.8154) and macro F1 (0.7774), but the same ROC-AUC as soft voting (0.8750). Same ranking with higher accuracy means the meta-learner mainly **moved the decision threshold**: the base models use class weights and lean towards "No rain", while the unweighted meta-learner leans back towards "Rain". It finds about 88% of rainy days but only about 67% of dry days (the other models find 76 to 77% of dry days).

## 4. Interpretation in the Sri Lankan context

**Which features drive the predictions?**
Rain today, the number of rain hours today and the 3-day rain average carry about 80% of the importance for Random Forest and about 69% for XGBoost. The models mostly learned "wet spells continue". Season and calendar features come next (monsoon season, month, day of year), then evaporation (ET0), which is high on sunny, dry days.

**How do the monsoon seasons affect the errors?** (`accuracy_by_season.csv`)

| Season | Share of rainy days in test | Soft Voting accuracy | Baseline accuracy |
|---|---|---|---|
| Southwest monsoon (May to September) | 81% | 0.866 | 0.859 |
| Second inter-monsoon (October, November) | 82% | 0.859 | 0.854 |
| First inter-monsoon (March, April) | 71% | 0.755 | 0.768 |
| Northeast monsoon (December to February) | 48% | 0.700 | 0.721 |

Accuracy is high when the weather is stable and mostly wet. It drops to about 0.70 in the Northeast monsoon, when only half of the days are rainy, wet and dry spells alternate, and the rain moves to the north and east. That is the season in which a forecast is most useful and hardest.

**Where are the errors?** (`accuracy_by_city.csv`)
Lowest accuracy: Badulla (0.71), Hambantota (0.72), Matale (0.73), Kalmunai (0.73), which are dry-zone or hill-country cities with irregular rain. Highest: Ratnapura and the Colombo area (0.83), in the wet zone where rain is regular.

**Recommendation.**
XGBoost is the best single model: slightly better than Random Forest on accuracy, macro F1 and ROC-AUC, faster and much smaller. The soft-voting ensemble is exported as `models/model.pkl` and used by the app, as the activity asks; its performance is almost the same. Stacking has the highest accuracy, but mainly through a different threshold and at about 4 times the training cost of soft voting. For real use, the predicted probability is more valuable than the yes/no answer.

## 5. Designed vs tuned hyperparameters

From `reports/hyperparameters.csv` (RandomizedSearchCV, 10 combinations, TimeSeriesSplit with 5 folds, ROC-AUC).

| Model | Parameter | Designed | Tuned |
|---|---|---|---|
| Random Forest | n_estimators | 150 | 100 |
| Random Forest | max_depth | 8 | 10 |
| Random Forest | min_samples_split | 100 | 200 |
| Random Forest | min_samples_leaf | 50 | 50 |
| Random Forest | max_features | sqrt | 0.5 |
| Random Forest | max_samples | 0.5 | 0.3 |
| XGBoost | n_estimators | 300 | 200 |
| XGBoost | learning_rate | 0.05 | 0.03 |
| XGBoost | max_depth | 5 | 4 |
| XGBoost | subsample | 0.8 | 0.9 |
| XGBoost | colsample_bytree | 0.8 | 0.9 |
| XGBoost | min_child_weight | 5 | 5 |
| XGBoost | gamma | 0.1 | 0.5 |
| XGBoost | reg_alpha | 0.1 | 0 |
| XGBoost | reg_lambda | 1.5 | 1.0 |

The best cross-validated ROC-AUC is about 0.894 for both models.

- Random Forest: slightly deeper trees, but each tree sees fewer rows (30%) and each split may use half of the features. With two dominant features, letting more features compete at each split helps.
- XGBoost: the search moved to a simpler model: fewer trees (200), smaller steps (learning rate 0.03), shallower trees (depth 4) and a stronger split penalty (`gamma` 0.5). For a noisy target, simpler trees generalise better to later years.
- Several XGBoost combinations score almost the same in cross-validation, so the winning combination can change between machines. No setting has a large effect: the data limits the accuracy more than the parameters do.

## 6. Optional extension: hybrid SARIMA + Random Forest

A separate **regression** experiment (`src/hybrid_arima_rf.py`): one-month-ahead forecast of Colombo's monthly rainfall total. Training months: January 2010 to August 2020. Test months: September 2020 to May 2023 (33 months).

| Model | MAE (mm) | RMSE (mm) |
|---|---|---|
| SARIMA(1,0,1)(0,1,1,12) only | 73.1 | 96.6 |
| Random Forest regression only | 81.8 | 101.2 |
| **Hybrid: SARIMA + Random Forest on SARIMA's residuals** | **70.9** | **94.4** |

The hybrid is the best of the three: SARIMA captures the yearly monsoon cycle and the Random Forest corrects part of what SARIMA missed. The gain over SARIMA alone is small (about 2 mm, or 3%) and the test period is only 33 months, so this is a sign that the idea works, not proof. All three models miss the extreme months (for example May 2022). Figure: `figures/hybrid_arima_rf.png`.

## 7. Limitations of this work

1. The data comes from a weather model (Open-Meteo reanalysis), not from Department of Meteorology rain gauges. Rain is stored on a coarse grid: only 15 distinct rain series exist for the 30 cities.
2. Important drivers of rain are not in the dataset: air pressure, humidity, cloud cover, and wider systems such as depressions in the Bay of Bengal.
3. Inputs are end-of-day values, so the model predicts the next day only.
4. The 1 mm threshold treats a light shower and a flood as the same class.
5. The models do not clearly beat the "same as today" rule on accuracy. Their value is in the probability and in the fairer treatment of dry days.
