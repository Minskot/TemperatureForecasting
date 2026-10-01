# Temperature Forecasting

This repository contains the final submission for a next-day weather forecasting project.

## Final Result

The project predicts the next day's mean temperature at Basel, Switzerland, using daily observations from 18 European weather stations between 2000 and 2010.

The selected model is `HistGradientBoostingRegressor`, chosen on the validation period and then refitted on the training plus validation data.

| Metric | Validation | Test |
|---|---:|---:|
| MAE | 1.217 C | **1.228 C** |
| RMSE | 1.546 C | 1.619 C |
| R2 | 0.949 | 0.957 |
| MAE skill vs. persistence | 25.8% | **26.3%** |

Additional experiment facts:

- 3,654 daily records and 163 source feature columns.
- 239 engineered features and 3,624 usable supervised samples.
- 36 physically invalid values repaired with causal forward filling.
- Chronological split: training through 2006-12-31, validation through 2008-12-31, and test from 2009-01-01 through 2010-01-01.
- An expanding-window backtest is included in `weather_prediction_project/outputs/`.

## Report

The submitted report is available here:

- [Task Report PDF](Task_Report.pdf)

## Project Contents

- `weather_prediction_project/`: source code, dataset, trained model, metrics, predictions, and figures.
- `weather_prediction_project/src/experiment.py`: end-to-end experiment pipeline.
- `weather_prediction_project/outputs/metrics/metrics.csv`: validation and test metrics.
- `weather_prediction_project/outputs/models/best_model.joblib`: fitted selected model.
- `weather_prediction_project/outputs/predictions/test_predictions.csv`: test predictions.

## Reproduce

```powershell
cd weather_prediction_project
python -m pip install -r requirements.txt
python run_experiment.py
```

The source dataset is the `weather_prediction_dataset` published by Florian Huber and based on ECA&D observations. Dataset details and attribution are included in `weather_prediction_project/data/metadata.txt`.
