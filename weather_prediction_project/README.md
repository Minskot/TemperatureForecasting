# Weather Prediction Project

Next-day mean temperature forecasting for Basel using the
`weather_prediction_dataset` (18 European stations, 2000-2009).

## Task

Given all weather observations available through day `t`, predict the mean temperature at Basel on day `t + 1`. The full dataset is available at:

https://github.com/florian-huber/weather_prediction_dataset



## Method summary

- Data quality: physically impossible or sentinel-like values are detected with range checks for pressure, humidity, cloud cover, precipitation, sunshine, wind, radiation, and temperature. Invalid values are replaced with causal forward-filled values, using only past observations.
- Chronological split by forecast target date: divided the dataset into 3 parts,  for trainning, validation and test.
  - training: target dates up to 2006-12-31
  - validation: 2007-01-01 to 2008-12-31
  - test: 2009-01-01 to 2010-01-01
- Baselines: persistence and monthly climatology.
- Models: ridge regression, random forest, gradient boosting, and a multi-layer perceptron.
- Features: current-day observations from 18 stations, calendar encoding, lags and rolling means for Basel temperature, and cross-station temperature lags.
- Model selection uses the validation period only. The selected model is refitted on training plus validation data and evaluated once on the test period.
- Metrics: MAE, RMSE, R2, and MAE skill score against persistence.
- Additional check: an expanding-window backtest that retrains before each month of 2009.

## Reproduce

Install the dependencies and run:

```bash
python run_experiment.py
```

The script writes data, metrics, predictions, figures, and the fitted model to the `outputs/` directory.

## Outputs

- `outputs/metrics/metrics.csv`: validation and test metrics
- `outputs/metrics/tuning_results.csv`: hyperparameter search results
- `outputs/metrics/data_quality_report.csv`: every invalid value, rule, and replacement
- `outputs/metrics/run_metadata.json`: run summary and warning counts
- `outputs/metrics/dataset_summary.json`: dataset and quality summary
- `outputs/predictions/test_predictions.csv`: test-period predictions
- `outputs/predictions/rolling_origin_predictions.csv`: expanding-window predictions
- `outputs/figures/`: analysis figures
- `outputs/models/best_model.joblib`: fitted validation-selected model
- `outputs/models/model_metadata.json`: feature order, target definition, cleaning rules, and metrics needed for inference

## Saved model

```python
import json
import joblib

model = joblib.load("outputs/models/best_model.joblib")
metadata = json.loads(
    open("outputs/models/model_metadata.json", encoding="utf-8").read()
)

# Build the input columns in metadata["feature_columns"] order.
```

## Data files

The `data/` directory contains the original CSV files and the dataset metadata provided by the dataset authors.



## Project Running

The project can run after all dependencies are installed.

```powershell
cd D:\Desktop\ModelTask\weather_prediction_project 
(Your own path here:)

pip install -r requirements.txt

D:\Desktop\ModelTask\weather_prediction_project\run_experiment.py
(Your own path here:)
```



```requirements.txt
To notify, these are depenencies to be installed:
numpy
pandas
scikit-learn
matplotlib
Pillow
joblib
```



Otherwise you can simply double click "run.bat", which does the preparations automatically.
