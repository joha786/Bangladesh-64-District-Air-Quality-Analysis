# Multi-Horizon Air Quality Index (AQI) Forecasting Across 64 Districts of Bangladesh

## Overview

This project presents a horizon-adaptive XGBoost framework for multi-horizon Air Quality Index (AQI) forecasting across all 64 districts of Bangladesh. It integrates air-quality, meteorological, historical, temporal, and spatial features to predict future AQI conditions at **1, 3, 6, 12, and 24-hour horizons**.

The framework uses multiclass classification for short-term forecasting (1–12 hours) and numerical regression followed by EPA-based AQI category mapping for 24-hour forecasting. It also evaluates spatial generalization by reserving 12 geographically unseen districts exclusively for final testing.

## Research Objectives

- Develop a nationwide multi-horizon AQI forecasting framework for Bangladesh.
- Investigate the impact of forecast horizons on prediction performance.
- Evaluate generalization across geographically unseen districts.
- Identify important air-quality and meteorological predictors using SHAP.
- Support the development of data-driven AQI early-warning systems.

## Dataset

The dataset contains hourly air-quality and meteorological observations covering all 64 districts of Bangladesh from **2023 to 2026**.

### Data Sources

| Source | Description |
|---|---|
| CAMS Global | Air-quality and pollutant variables |
| ERA5-Land | Meteorological variables |
| Open-Meteo | Data access interface |

### Variables

**Air-quality variables**
- PM2.5
- PM10
- Carbon Monoxide (CO)
- Nitrogen Dioxide (NO2)
- Sulphur Dioxide (SO2)
- Ozone (O3)

**Meteorological variables**
- Temperature
- Relative Humidity
- Dew Point
- Surface Pressure
- Precipitation
- Cloud Cover
- Wind Speed
- Wind Direction

Geographic coordinates and timestamps support spatial and temporal analysis.

## Methodology

The proposed framework consists of the following stages:

1. **Data Collection:** Collect hourly air-quality and meteorological data for 64 districts.
2. **Preprocessing:** Perform data validation, missing-value checks, temporal alignment, and AQI category preparation.
3. **Feature Engineering:** Construct historical, lag-based, rolling, rate-of-change, spatial, and temporal features using information available at or before the prediction time.
4. **Target Construction:** Generate future AQI targets at 1, 3, 6, 12, and 24-hour horizons.
5. **Horizon-Adaptive XGBoost:** Apply classification for 1–12 hours and regression with AQI category mapping for 24 hours.
6. **Evaluation:** Assess forecasting performance on seen and geographically unseen districts.
7. **Interpretability:** Apply SHAP to investigate feature contributions.

## Forecasting Strategy

| Forecast Horizon | Prediction Task | Approach |
|---|---|---|
| 1 Hour | Classification | XGBoost |
| 3 Hours | Classification | XGBoost |
| 6 Hours | Classification | XGBoost |
| 12 Hours | Classification | XGBoost |
| 24 Hours | Regression + Category Mapping | XGBoost |

The five AQI categories are:

1. Good
2. Moderate
3. Unhealthy for Sensitive Groups
4. Unhealthy
5. Very Unhealthy

## Experimental Design

A chronological data partitioning strategy was used alongside a geographical holdout.

| Subset | Period | Districts |
|---|---|---|
| Training | Up to June 30, 2025 | Seen |
| Validation | July 1 – December 31, 2025 | Seen |
| Final Testing | After December 31, 2025 | Seen + 12 Unseen |

Twelve geographically unseen districts were excluded from training and validation to evaluate spatial generalization. A 24-hour embargo was applied at temporal split boundaries to reduce temporal leakage.

## Evaluation Metrics

**Classification metrics**
- Accuracy
- Macro-F1 Score
- Quadratic Weighted Kappa (QWK)

**Regression metric**
- Root Mean Squared Error (RMSE)

Results are reported separately for seen and unseen districts.

## Experimental Results

The framework demonstrates strong short-term forecasting performance, with a gradual decline as the prediction horizon increases.

### Overall Performance

| Horizon | Seen Accuracy | Seen Macro-F1 | Seen QWK |
|---|---:|---:|---:|
| 1h | 98.98% | 98.52% | 99.48% |
| 3h | — | — | — |
| 6h | — | — | — |
| 12h | — | — | — |
| 24h | 79.54% | 71.18% | 88.88% |

| Horizon | Unseen Accuracy | Unseen Macro-F1 | Unseen QWK |
|---|---:|---:|---:|
| 1h | 98.92% | 98.41% | 99.41% |
| 3h | — | — | — |
| 6h | — | — | — |
| 12h | — | — | — |
| 24h | 79.65% | 71.72% | 88.02% |

**24-hour regression performance**

| Evaluation Group | RMSE |
|---|---:|
| Seen Districts | 14.43 AQI points |
| Unseen Districts | 14.50 AQI points |

*Note: Intermediate-horizon values are omitted here until the complete finalized results are added.*

## Model Interpretability

SHAP (SHapley Additive exPlanations) is used to investigate the contribution of input features to model predictions.

The analysis examines:
- Overall feature importance.
- Pollutant contributions to the Very Unhealthy AQI category.
- Feature contribution patterns across forecasting horizons.

Among the evaluated raw pollutants, PM2.5 shows the strongest contribution, followed by ozone (O3). Current AQI is also an important predictor in the full feature-level analysis.


## Author

**Md. Joha Mondal**  
Department of Computer Science and Engineering  
American International University-Bangladesh (AIUB)
