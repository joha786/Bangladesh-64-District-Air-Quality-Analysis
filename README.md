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

## Dataset Citation

The dataset used in this research is publicly available through Mendeley Data:

> Joha, Md Joha Mondal (2026), "Bangladesh 64-District Hourly Air Quality and Weather Dataset (2023–2026)", Mendeley Data, V1. https://doi.org/10.17632/c7trkj4hzv.1

**DOI:** [10.17632/c7trkj4hzv.1](https://doi.org/10.17632/c7trkj4hzv.1)

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

### Overall Performance

## Experimental Results

The proposed framework was evaluated across five forecasting horizons (1h, 3h, 6h, 12h, and 24h) using Accuracy, Macro-F1, and Quadratic Weighted Kappa (QWK). Performance was reported separately for seen and geographically unseen districts.

### Overall Classification Performance

**Seen Districts**

| Horizon | Accuracy | Macro-F1 | QWK |
|---|---:|---:|---:|
| 1h | 99.0% | 98.5% | 99.5% |
| 3h | 97.9% | 96.6% | 98.9% |
| 6h | 95.1% | 92.7% | 97.5% |
| 12h | 89.0% | 83.9% | 94.3% |
| 24h | 79.5% | 71.2% | 88.9% |

**Geographically Unseen Districts**

| Horizon | Accuracy | Macro-F1 | QWK |
|---|---:|---:|---:|
| 1h | 98.9% | 98.4% | 99.4% |
| 3h | 97.8% | 96.5% | 98.8% |
| 6h | 95.2% | 92.4% | 97.4% |
| 12h | 89.0% | 83.8% | 93.9% |
| 24h | 79.7% | 71.7% | 88.0% |

### 24-Hour Regression Performance

The 24-hour forecasting task uses numerical AQI regression followed by AQI category mapping.

| Evaluation Group | RMSE |
|---|---:|
| Seen Districts | 14.43 AQI points |
| Unseen Districts | 14.50 AQI points |

### Key Findings

- **Strong short-term forecasting:** The 1-hour horizon achieves approximately 99% Accuracy and Macro-F1 above 98% for both evaluation groups.
- **Gradual performance decline:** Classification performance decreases as the forecasting horizon increases.
- **Spatial generalization:** Seen and unseen districts demonstrate closely comparable performance across all horizons.
- **Ordinal consistency:** QWK remains relatively high, reaching approximately 88% even at the 24-hour horizon.
- **24-hour prediction:** Regression RMSE remains comparable between seen and unseen districts, at 14.43 and 14.50 AQI points, respectively.

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
