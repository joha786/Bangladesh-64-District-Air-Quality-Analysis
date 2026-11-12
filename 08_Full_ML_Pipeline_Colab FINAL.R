# ==============================================================================
# FINAL ML PIPELINE – MULTI-HORIZON ORDINAL AQI EARLY‑WARNING MODEL
# Bangladesh 64‑District Hourly Data (2023‑01‑01 to 2026‑08‑20)
#
# FEATURES (34): Weather (8) + Raw Pollutants (6) + Overall AQI (1) +
#               Lags (7) + Rolling/Trend (3) + Volatility (4) +
#               Acceleration (1) + Spatial (2) + Temporal (2)
#               (Sub‑AQI columns removed)
#
# MODELS:  Baseline (multinomial logistic regression) for all horizons.
#          XGBoost: classification for 1h–12h, regression for 24h.
#
# OUTPUTS: Processed dataset saved as CSV; all tables and figures saved
#          to designated folders (PNG + PDF).
# ==============================================================================

# ---- 0. PACKAGES ------------------------------------------------------------

required_packages <- c("data.table", "lubridate", "xgboost", "nnet",
                       "Metrics", "ggplot2", "scales")
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) install.packages(missing_packages)

library(data.table)
library(lubridate)
library(xgboost)
library(nnet)
library(Metrics)
library(ggplot2)
library(scales)

set.seed(42)

# ---- 1. CONFIG ---------------------------------------------------------------

CONFIG <- list(
  input_path  = "F:/Semester 9/Data Science/Final Research/Main Project Work/05_Future_AQI_Target_Construction/outputs/Bangladesh_64_District_Hourly_2023_2026_MULTI_HORIZON_AQI_TARGETS.csv",
  output_dir  = "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs",
  plot_dir    = "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs/figures",
  horizons = c("1h", "3h", "6h", "12h", "24h"),
  train_end      = as.Date("2025-06-30"),
  validation_end = as.Date("2025-12-31"),
  n_unseen_districts = 12,
  baseline_sample_size = 200000,
  category_levels = c("Good", "Moderate", "Unhealthy for Sensitive Groups",
                      "Unhealthy", "Very Unhealthy"),
  epa_breaks = c(-Inf, 50, 100, 150, 200, Inf),
  shap_sample_size = 20000,
  shap_top_n = 12,
  seed = 42
)

if (!dir.exists(CONFIG$output_dir)) dir.create(CONFIG$output_dir, recursive = TRUE)
if (!dir.exists(CONFIG$plot_dir))   dir.create(CONFIG$plot_dir,   recursive = TRUE)

# ---- Publication theme ----
theme_paper <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 1.5, hjust = 0),
      plot.subtitle    = element_text(size = base_size - 1, color = "grey35"),
      plot.caption     = element_text(size = base_size - 3, color = "grey45", hjust = 1),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      legend.position  = "right",
      legend.title     = element_text(face = "bold", size = base_size - 1),
      strip.background = element_rect(fill = "grey15", color = NA),
      strip.text       = element_text(face = "bold", color = "white", size = base_size - 1)
    )
}

save_plot <- function(plot, filename, width = 8, height = 5) {
  ggsave(file.path(CONFIG$plot_dir, paste0(filename, ".png")), plot,
         width = width, height = height, dpi = 320, bg = "white")
  ggsave(file.path(CONFIG$plot_dir, paste0(filename, ".pdf")), plot,
         width = width, height = height, bg = "white")
}

# ---- 2. LOAD DATA -------------------------------------------------------------

cat("Loading dataset...\n")
dt <- fread(CONFIG$input_path)
cat("Rows:", format(nrow(dt), big.mark = ","), "| Columns:", ncol(dt), "\n")

# ---- 3. FIX KNOWN BUGS ---------------------------------------------------------

cat("\nFixing known data issues...\n")

horizon_cat_cols <- paste0("aqi_category_t_plus_", CONFIG$horizons)
for (col in horizon_cat_cols) {
  dt[get(col) == "", (col) := NA_character_]
}
cat("Fixed: empty-string missing values converted to NA in target columns.\n")

dt[, datetime_utc := as.POSIXct(datetime_utc, tz = "UTC")]
dt[, datetime_bd_fixed := with_tz(datetime_utc, tzone = "Asia/Dhaka")]
dt[, year_bd_fixed  := year(datetime_bd_fixed)]
dt[, month_bd_fixed := month(datetime_bd_fixed)]
dt[, hour_bd_fixed  := hour(datetime_bd_fixed)]
cat("Fixed: datetime_bd recomputed with a genuine Asia/Dhaka shift.\n")

utc_hour_check   <- hour(dt$datetime_utc[1])
local_hour_check <- dt$hour_bd_fixed[1]
offset_check <- (local_hour_check - utc_hour_check) %% 24
if (offset_check != 6) {
  stop("Timezone offset check failed -- expected +6h, got ", offset_check, "h")
}
cat("Timezone offset verified: +6h (Asia/Dhaka).\n")

# ---- 4. ORDER DATA -------------------------------------------------------------

setorder(dt, district_key, datetime_utc)

# ---- 5. MERGE RARE "HAZARDOUS" CLASS INTO "VERY UNHEALTHY" ---------------------

cat("\nMerging rare 'Hazardous' class into 'Very Unhealthy' for modeling...\n")

for (h in CONFIG$horizons) {
  src_col <- paste0("aqi_category_t_plus_", h)
  new_col <- paste0("aqi_class_", h)
  dt[, (new_col) := fifelse(get(src_col) == "Hazardous", "Very Unhealthy", get(src_col))]
  dt[, (new_col) := factor(get(new_col), levels = CONFIG$category_levels, ordered = TRUE)]
}

# ---- 6. FEATURE ENGINEERING (FINAL SET) ----------------------------------------

cat("\nBuilding final feature set (34 predictors)...\n")

# Basic lags (6h, 24h)
dt[, us_aqi_lag_6h   := shift(us_aqi, n = 6,  type = "lag"), by = district_key]
dt[, us_aqi_lag_24h  := shift(us_aqi, n = 24, type = "lag"), by = district_key]
dt[, pm2_5_lag_6h    := shift(pm2_5,  n = 6,  type = "lag"), by = district_key]
dt[, pm2_5_lag_24h   := shift(pm2_5,  n = 24, type = "lag"), by = district_key]

# Rolling mean (24h)
dt[, us_aqi_roll_24h := frollmean(us_aqi, n = 24, align = "right"), by = district_key]

# Change (6h)
dt[, us_aqi_change_6h := us_aqi - us_aqi_lag_6h]
dt[, pm2_5_change_6h  := pm2_5  - pm2_5_lag_6h]

# Longer lags (48h, 72h)
dt[, us_aqi_lag_48h := shift(us_aqi, n = 48, type = "lag"), by = district_key]
dt[, pm2_5_lag_48h  := shift(pm2_5,  n = 48, type = "lag"), by = district_key]
dt[, us_aqi_lag_72h := shift(us_aqi, n = 72, type = "lag"), by = district_key]

# Volatility (SD, min, max, range over 24h)
dt[, us_aqi_sd_24h   := suppressWarnings(frollapply(us_aqi, n = 24, FUN = sd,   align = "right")), by = district_key]
dt[, us_aqi_min_24h  := suppressWarnings(frollapply(us_aqi, n = 24, FUN = min,  align = "right")), by = district_key]
dt[, us_aqi_max_24h  := suppressWarnings(frollapply(us_aqi, n = 24, FUN = max,  align = "right")), by = district_key]
dt[, us_aqi_range_24h := us_aqi_max_24h - us_aqi_min_24h]

# Acceleration (change of 6h change)
dt[, us_aqi_lag_6h_change := shift(us_aqi_change_6h, n = 6, type = "lag"), by = district_key]
dt[, us_aqi_accel_6h := us_aqi_change_6h - us_aqi_lag_6h_change]

cat("Features built.\n")

# ---- DEFINE FINAL PREDICTOR COLUMNS (34 features, sub-AQI removed) ----
predictor_columns <- c(
  # 8 Weather
  "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
  "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
  # 6 Raw pollutants
  "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
  # 1 Overall AQI
  "us_aqi",
  # 4 Lags (6h, 24h for us_aqi and pm2_5)
  "us_aqi_lag_6h", "us_aqi_lag_24h", "pm2_5_lag_6h", "pm2_5_lag_24h",
  # 3 Rolling/trend
  "us_aqi_roll_24h", "us_aqi_change_6h", "pm2_5_change_6h",
  # Longer lags (48h, 72h for us_aqi; 48h for pm2_5)
  "us_aqi_lag_48h", "pm2_5_lag_48h", "us_aqi_lag_72h",
  # 4 Volatility
  "us_aqi_sd_24h", "us_aqi_min_24h", "us_aqi_max_24h", "us_aqi_range_24h",
  # 1 Acceleration
  "us_aqi_accel_6h",
  # 2 Spatial
  "latitude", "longitude",
  # 2 Temporal
  "month_bd_fixed", "hour_bd_fixed"
)

cat("Final predictor count:", length(predictor_columns), "\n")

# ---- 7. SPLITS: CHRONOLOGICAL + SPATIAL ------------------------------------------

dt[, split_time := fcase(
  as.Date(datetime_bd_fixed) <= CONFIG$train_end,                                    "train",
  as.Date(datetime_bd_fixed) >  CONFIG$train_end &
    as.Date(datetime_bd_fixed) <= CONFIG$validation_end,                              "validation",
  as.Date(datetime_bd_fixed) >  CONFIG$validation_end,                                "test"
)]

set.seed(CONFIG$seed)
all_districts    <- sort(unique(dt$district_key))
unseen_districts <- sample(all_districts, CONFIG$n_unseen_districts)
seen_districts   <- setdiff(all_districts, unseen_districts)

dt[, district_role := fifelse(district_key %in% unseen_districts, "unseen", "seen")]
dt[district_role == "unseen" & split_time %in% c("train", "validation"),
   split_time := "excluded_unseen_district"]

cat("\nSplit sizes (rows):\n")
print(dt[, .N, by = split_time])
cat("Unseen districts (n =", length(unique(dt$district_key[dt$district_role == "unseen"])), "):\n")
print(sort(unseen_districts))

# ---- 7.5 EMBARGO: Remove rows near split boundaries to prevent label crossing ----
cat("\nApplying embargo (removing last 24h of train & validation per district)...\n")

max_horizon <- 24

# For each district, find the maximum datetime in train and validation
embargo_dt <- dt[
  split_time %in% c("train", "validation"),
  .(max_datetime = max(datetime_bd_fixed)),
  by = .(district_key, split_time)
]

# Compute the embargo cutoff (subtract max_horizon hours)
embargo_dt[, cutoff := max_datetime - hours(max_horizon)]

# Mark rows that fall in the embargo period for train/validation
dt[, embargo := FALSE]
for (i in 1:nrow(embargo_dt)) {
  dist <- embargo_dt$district_key[i]
  split <- embargo_dt$split_time[i]
  cutoff <- embargo_dt$cutoff[i]
  
  dt[
    district_key == dist &
      split_time == split &
      datetime_bd_fixed > cutoff,
    embargo := TRUE
  ]
}

# Remove embargoed rows from train and validation
dt[embargo == TRUE, split_time := "excluded_embargo"]

cat("Embargoed rows removed:", dt[embargo == TRUE, .N], "\n")
cat("Remaining rows:", dt[split_time != "excluded_embargo", .N], "\n")

# ---- 8. SAVE PROCESSED DATASET (FEATURES + TARGETS + NUMERIC TARGETS) -----------

numeric_target_cols <- paste0("aqi_t_plus_", gsub("h", "", CONFIG$horizons), "h")
model_cols <- c(predictor_columns, numeric_target_cols,
                paste0("aqi_class_", CONFIG$horizons),
                "split_time", "district_role", "district_key")
dt_model <- dt[, ..model_cols]

model_dataset_path <- file.path(CONFIG$output_dir, "model_dataset_final.csv")
fwrite(dt_model, model_dataset_path)
cat("\nFinal model dataset saved to:", model_dataset_path, "\n")
cat("Dimensions:", nrow(dt_model), "rows x", ncol(dt_model), "columns\n")

# ---- 9. HELPER FUNCTIONS (for modelling) ---------------------------------------

compute_class_weights <- function(y) {
  tab <- table(y)
  n_total <- length(y)
  n_classes <- length(tab)
  w_lookup <- n_total / (n_classes * tab)
  as.numeric(w_lookup[as.character(y)])
}

build_horizon_dataset <- function(dt, horizon) {
  class_col <- paste0("aqi_class_", horizon)
  numeric_col <- paste0("aqi_t_plus_", gsub("h", "", horizon), "h")
  cols_needed <- c(predictor_columns, class_col, numeric_col,
                   "split_time", "district_role", "district_key")
  d <- dt[!is.na(get(class_col)), ..cols_needed]
  d <- d[complete.cases(d[, ..predictor_columns])]
  setnames(d, class_col, "aqi_class")
  d[, aqi_class_int := as.integer(aqi_class) - 1L]
  d
}

n_classes_total <- length(CONFIG$category_levels)
all_class_ints  <- 0:(n_classes_total - 1)

per_class_metrics <- function(actual_int, predicted_int, level_names = CONFIG$category_levels) {
  rbindlist(lapply(seq_along(all_class_ints), function(i) {
    cl <- all_class_ints[i]
    tp <- sum(predicted_int == cl & actual_int == cl)
    fp <- sum(predicted_int == cl & actual_int != cl)
    fn <- sum(predicted_int != cl & actual_int == cl)
    support   <- sum(actual_int == cl)
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    recall    <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1        <- if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
    data.table(class = level_names[i], support = support,
               precision = round(precision, 4), recall = round(recall, 4), f1 = round(f1, 4))
  }))
}

evaluate_predictions <- function(actual_int, predicted_int, label) {
  acc <- mean(actual_int == predicted_int)
  pcm <- per_class_metrics(actual_int, predicted_int)
  macro_f1 <- mean(pcm$f1)
  qwk <- Metrics::ScoreQuadraticWeightedKappa(
    actual_int, predicted_int,
    min.rating = 0, max.rating = n_classes_total - 1
  )
  list(
    summary = data.table(
      subset   = label,
      n        = length(actual_int),
      accuracy = round(acc, 4),
      macro_f1 = round(macro_f1, 4),
      qwk      = round(qwk, 4)
    ),
    per_class = pcm[, subset := label]
  )
}

# Robust SHAP extractor (handles 3D arrays)
extract_mean_abs_shap <- function(xgb_model, newdata_matrix, predictor_columns) {
  n_feat <- length(predictor_columns)
  contrib <- predict(xgb_model, newdata = newdata_matrix, predcontrib = TRUE)
  
  if (is.list(contrib) && !is.array(contrib)) {
    abs_contribs <- lapply(contrib, function(m) abs(m[, seq_len(n_feat), drop = FALSE]))
    combined <- Reduce(`+`, abs_contribs) / length(abs_contribs)
    mean_abs <- colMeans(combined)
  } else if (length(dim(contrib)) == 3) {
    dims <- dim(contrib)
    feat_dim <- which(dims == n_feat + 1)
    if (length(feat_dim) == 0) stop("Could not identify feature dimension.")
    perm_order <- c(setdiff(seq_along(dims), feat_dim), feat_dim)
    arr_perm <- aperm(contrib, perm = perm_order)
    mean_abs <- apply(abs(arr_perm), 3, mean)
  } else {
    mean_abs <- colMeans(abs(contrib[, seq_len(n_feat), drop = FALSE]))
  }
  data.table(feature = predictor_columns, mean_abs_shap = as.numeric(mean_abs))
}

# ---- 10. MAIN LOOP: TRAIN + EVALUATE PER HORIZON (with regression for 24h) ----

all_results     <- list()
all_per_class   <- list()
all_shap        <- list()
all_evals_log   <- list()
all_importance  <- list()
trained_models  <- list()

for (h in CONFIG$horizons) {
  
  cat("\n==============================================================\n")
  cat("HORIZON:", h, "\n")
  cat("==============================================================\n")
  
  hd <- build_horizon_dataset(dt_model, h)
  
  train_d       <- hd[split_time == "train"]
  validation_d  <- hd[split_time == "validation"]
  test_seen_d   <- hd[split_time == "test" & district_role == "seen"]
  test_unseen_d <- hd[split_time == "test" & district_role == "unseen"]
  
  cat("Train:", nrow(train_d), "| Validation:", nrow(validation_d),
      "| Test (seen):", nrow(test_seen_d), "| Test (unseen):", nrow(test_unseen_d), "\n")
  
  # ---- Baseline: multinomial logistic regression ----
  cat("\nTraining baseline (multinomial logistic regression)...\n")
  
  baseline_train <- train_d
  if (nrow(baseline_train) > CONFIG$baseline_sample_size) {
    set.seed(CONFIG$seed)
    baseline_train <- baseline_train[sample(.N, CONFIG$baseline_sample_size)]
  }
  baseline_weights <- compute_class_weights(baseline_train$aqi_class)
  
  baseline_formula <- as.formula(paste("aqi_class ~", paste(predictor_columns, collapse = " + ")))
  baseline_model <- multinom(
    baseline_formula, data = baseline_train, weights = baseline_weights,
    MaxNWts = 5000, trace = FALSE
  )
  
  baseline_pred_seen   <- predict(baseline_model, newdata = test_seen_d)
  baseline_pred_unseen <- predict(baseline_model, newdata = test_unseen_d)
  
  eval_base_seen   <- evaluate_predictions(as.integer(test_seen_d$aqi_class) - 1L,
                                           as.integer(baseline_pred_seen) - 1L, "test_seen")
  eval_base_unseen <- evaluate_predictions(as.integer(test_unseen_d$aqi_class) - 1L,
                                           as.integer(baseline_pred_unseen) - 1L, "test_unseen")
  
  baseline_metrics <- rbind(eval_base_seen$summary, eval_base_unseen$summary)
  baseline_metrics[, `:=`(horizon = h, model = "baseline_multinom", rmse = NA_real_)]
  
  # ---- XGBoost: classification or regression for 24h ----
  use_regression <- (h == "24h")
  
  if (use_regression) {
    cat("Using REGRESSION for 24h horizon (predict numeric AQI then map to classes)\n")
    
    num_h <- gsub("h", "", h)  # "24h" -> "24"
    num_target_col <- paste0("aqi_t_plus_", num_h, "h")
    
    if (!num_target_col %in% names(train_d)) {
      stop(paste("Numeric target column", num_target_col, "not found."))
    }
    
    train_d_reg <- train_d[!is.na(get(num_target_col))]
    valid_d_reg <- validation_d[!is.na(get(num_target_col))]
    
    dtrain_reg <- xgb.DMatrix(
      data = as.matrix(train_d_reg[, ..predictor_columns]),
      label = train_d_reg[[num_target_col]]
    )
    dvalid_reg <- xgb.DMatrix(
      data = as.matrix(valid_d_reg[, ..predictor_columns]),
      label = valid_d_reg[[num_target_col]]
    )
    
    xgb_model <- xgb.train(
      params = list(
        objective = "reg:squarederror",
        eval_metric = "rmse",
        max_depth = 8,
        eta = 0.02,
        subsample = 0.6,
        colsample_bytree = 0.6,
        min_child_weight = 5,
        gamma = 0.1
      ),
      data = dtrain_reg,
      nrounds = 1500,
      evals = list(train = dtrain_reg, valid = dvalid_reg),
      early_stopping_rounds = 50,
      verbose = 1,
      print_every_n = 50
    )
    
    pred_num_seen <- predict(xgb_model, as.matrix(test_seen_d[, ..predictor_columns]))
    pred_num_unseen <- predict(xgb_model, as.matrix(test_unseen_d[, ..predictor_columns]))
    
    # ---- Compute RMSE for numeric predictions ----
    rmse_seen <- sqrt(mean((pred_num_seen - test_seen_d[[num_target_col]])^2, na.rm = TRUE))
    rmse_unseen <- sqrt(mean((pred_num_unseen - test_unseen_d[[num_target_col]])^2, na.rm = TRUE))
    
    pred_class_seen <- cut(pred_num_seen,
                           breaks = CONFIG$epa_breaks,
                           labels = CONFIG$category_levels,
                           right = FALSE)
    pred_class_unseen <- cut(pred_num_unseen,
                             breaks = CONFIG$epa_breaks,
                             labels = CONFIG$category_levels,
                             right = FALSE)
    
    xgb_pred_seen_int   <- as.integer(pred_class_seen) - 1L
    xgb_pred_unseen_int <- as.integer(pred_class_unseen) - 1L
    
    eval_xgb_seen   <- evaluate_predictions(test_seen_d$aqi_class_int, xgb_pred_seen_int, "test_seen")
    eval_xgb_unseen <- evaluate_predictions(test_unseen_d$aqi_class_int, xgb_pred_unseen_int, "test_unseen")
    
    # ---- Merge metrics and add RMSE ----
    xgb_metrics <- rbind(eval_xgb_seen$summary, eval_xgb_unseen$summary)
    xgb_metrics[, `:=`(
      horizon = h,
      model = "xgboost_reg",
      rmse = ifelse(
        subset == "test_seen",
        rmse_seen,
        rmse_unseen
      )
    )]
    
  } else {
    cat("Using CLASSIFICATION for horizon", h, "\n")
    
    train_weights <- compute_class_weights(train_d$aqi_class)
    
    dtrain <- xgb.DMatrix(
      data = as.matrix(train_d[, ..predictor_columns]),
      label = train_d$aqi_class_int,
      weight = train_weights
    )
    dvalid <- xgb.DMatrix(
      data = as.matrix(validation_d[, ..predictor_columns]),
      label = validation_d$aqi_class_int
    )
    
    if (h %in% c("1h", "3h")) {
      params <- list(
        objective = "multi:softprob", num_class = n_classes_total,
        eval_metric = "mlogloss", max_depth = 6, eta = 0.1,
        subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 1
      )
      nrounds <- 300; early_stop <- 20
    } else if (h == "6h") {
      params <- list(
        objective = "multi:softprob", num_class = n_classes_total,
        eval_metric = "mlogloss", max_depth = 6, eta = 0.08,
        subsample = 0.75, colsample_bytree = 0.8, min_child_weight = 2
      )
      nrounds <- 500; early_stop <- 30
    } else {  # 12h
      params <- list(
        objective = "multi:softprob", num_class = n_classes_total,
        eval_metric = "mlogloss", max_depth = 8, eta = 0.02,
        subsample = 0.6, colsample_bytree = 0.6,
        min_child_weight = 5, gamma = 0.1
      )
      nrounds <- 1500; early_stop <- 50
    }
    
    xgb_model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      evals = list(train = dtrain, valid = dvalid),
      early_stopping_rounds = early_stop,
      verbose = 1,
      print_every_n = 50
    )
    
    # predict without reshape (use as.matrix and max.col)
    xgb_pred_seen_prob <- predict(xgb_model, as.matrix(test_seen_d[, ..predictor_columns]))
    xgb_pred_unseen_prob <- predict(xgb_model, as.matrix(test_unseen_d[, ..predictor_columns]))
    
    # For multi:softprob, the output is a matrix with columns = classes
    # We need to reshape it to (n_samples, n_classes)
    # If the output is a vector, we need to reshape manually.
    if (is.vector(xgb_pred_seen_prob)) {
      n_classes <- n_classes_total
      xgb_pred_seen_prob <- matrix(xgb_pred_seen_prob, ncol = n_classes, byrow = TRUE)
      xgb_pred_unseen_prob <- matrix(xgb_pred_unseen_prob, ncol = n_classes, byrow = TRUE)
    }
    
    xgb_pred_seen_int   <- max.col(xgb_pred_seen_prob) - 1L
    xgb_pred_unseen_int <- max.col(xgb_pred_unseen_prob) - 1L
    
    eval_xgb_seen   <- evaluate_predictions(test_seen_d$aqi_class_int, xgb_pred_seen_int, "test_seen")
    eval_xgb_unseen <- evaluate_predictions(test_unseen_d$aqi_class_int, xgb_pred_unseen_int, "test_unseen")
    
    xgb_metrics <- rbind(eval_xgb_seen$summary, eval_xgb_unseen$summary)
    xgb_metrics[, `:=`(horizon = h, model = "xgboost", rmse = NA_real_)]
  }
  
  trained_models[[h]] <- xgb_model
  
  # Capture evaluation logs (only for classification)
  if (!use_regression) {
    eval_log_h <- data.table::copy(as.data.table(xgb_model$evals_log))
    iter_col_candidates <- intersect(c("iter", "iteration"), names(eval_log_h))
    if (length(iter_col_candidates) == 0) {
      eval_log_h[, iter := seq_len(.N)]
    } else if (iter_col_candidates[1] != "iter") {
      setnames(eval_log_h, iter_col_candidates[1], "iter")
    }
    eval_log_h[, horizon := h]
    eval_log_h[, best_iteration := xgb_model$best_iteration]
    all_evals_log[[h]] <- eval_log_h
  } else {
    # Placeholder for regression (no mlogloss curve)
    all_evals_log[[h]] <- data.table(iter = 1, horizon = h, best_iteration = 1)
  }
  
  all_results[[h]] <- rbindlist(list(baseline_metrics, xgb_metrics), fill = TRUE)
  
  per_class_h <- rbind(eval_xgb_seen$per_class, eval_xgb_unseen$per_class)
  per_class_h[, `:=`(horizon = h, model = ifelse(use_regression, "xgboost_reg", "xgboost"))]
  all_per_class[[h]] <- per_class_h
  
  cat("\nResults for horizon", h, ":\n")
  print(all_results[[h]])
  
  # ---- Confusion matrix (only for classification, skip regression) ----
  if (!use_regression) {
    conf_dt <- data.table(
      actual    = factor(CONFIG$category_levels[test_seen_d$aqi_class_int + 1],
                         levels = CONFIG$category_levels),
      predicted = factor(CONFIG$category_levels[xgb_pred_seen_int + 1],
                         levels = CONFIG$category_levels)
    )[, .N, by = .(actual, predicted)]
    conf_dt <- conf_dt[CJ(actual = CONFIG$category_levels, predicted = CONFIG$category_levels),
                       on = c("actual", "predicted")]
    conf_dt[is.na(N), N := 0]
    conf_dt[, actual    := factor(actual,    levels = CONFIG$category_levels)]
    conf_dt[, predicted := factor(predicted, levels = rev(CONFIG$category_levels))]
    conf_dt[, row_pct := round(100 * N / sum(N), 1), by = actual]
    
    fwrite(conf_dt, file.path(CONFIG$output_dir, paste0("confusion_matrix_", h, ".csv")))
    
    p_conf <- ggplot(conf_dt, aes(x = actual, y = predicted, fill = row_pct)) +
      geom_tile(color = "white", linewidth = 0.6) +
      geom_text(aes(label = paste0(N, "\n(", row_pct, "%)")), size = 3,
                color = ifelse(conf_dt$row_pct > 50, "white", "grey15")) +
      scale_fill_viridis_c(name = "Row %", option = "magma", direction = -1) +
      scale_x_discrete(guide = guide_axis(angle = 30)) +
      labs(title = paste0("Confusion Matrix -- XGBoost, ", h, " horizon"),
           subtitle = "Seen-district test set, row-normalized",
           x = "Actual class", y = "Predicted class") +
      theme_paper()
    save_plot(p_conf, paste0("confusion_matrix_", h), width = 7.5, height = 6.5)
  }
  
  # ---- Native XGBoost gain importance ----
  imp_h <- as.data.table(xgb.importance(model = xgb_model, feature_names = predictor_columns))
  imp_h[, horizon := h]
  all_importance[[h]] <- imp_h
  
  # ---- SHAP (only for classification models) ----
  if (!use_regression) {
    cat("\nComputing SHAP for horizon", h, "...\n")
    shap_sample <- test_seen_d
    if (nrow(shap_sample) > CONFIG$shap_sample_size) {
      set.seed(CONFIG$seed)
      shap_sample <- shap_sample[sample(.N, CONFIG$shap_sample_size)]
    }
    shap_result <- tryCatch({
      contrib <- predict(xgb_model, newdata = as.matrix(shap_sample[, ..predictor_columns]),
                         predcontrib = TRUE)
      cat("DEBUG: dim(contrib) =", paste(dim(contrib), collapse = " x "), "\n")
      res <- extract_mean_abs_shap(
        xgb_model,
        as.matrix(shap_sample[, ..predictor_columns]),
        predictor_columns
      )
      res[, horizon := h]
      res
    }, error = function(e) {
      cat("SHAP error:", e$message, "\n")
      NULL
    })
    if (!is.null(shap_result)) {
      all_shap[[h]] <- shap_result
      cat("SHAP computed successfully for horizon", h, "\n")
    }
  } else {
    cat("Skipping SHAP for 24h (regression).\n")
  }
}

# ---- 11. HORIZON COMPARISON TABLES + PLOTS (with numeric labels) ----

cat("\n==============================================================\n")
cat("HORIZON COMPARISON -- ALL RESULTS\n")
cat("==============================================================\n")

results_table <- rbindlist(all_results, fill = TRUE)
results_table[, horizon := factor(horizon, levels = CONFIG$horizons)]
fwrite(results_table, file.path(CONFIG$output_dir, "horizon_comparison_results.csv"))

per_class_table <- rbindlist(all_per_class, fill = TRUE)
per_class_table[, horizon := factor(horizon, levels = CONFIG$horizons)]
fwrite(per_class_table, file.path(CONFIG$output_dir, "per_class_metrics.csv"))

# Macro-F1 plot
p_results <- ggplot(
  results_table[model %in% c("xgboost", "xgboost_reg")],
  aes(x = horizon, y = macro_f1, color = subset, group = interaction(subset, model), linetype = model)
) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.3f", macro_f1)),
            vjust = -1.2, size = 3.5, show.legend = FALSE) +
  scale_color_viridis_d(option = "plasma", end = 0.7) +
  scale_linetype_manual(values = c("xgboost" = "solid", "xgboost_reg" = "dashed")) +
  scale_y_continuous(limits = c(0, 1.05)) +
  labs(title = "Macro-F1 by Forecast Horizon (XGBoost)",
       subtitle = "Seen vs. unseen districts (dashed = regression-based 24h)",
       x = "Forecast Horizon", y = "Macro-F1", color = "Test subset", linetype = "Model") +
  theme_paper()
save_plot(p_results, "horizon_comparison_macrof1")

# All metrics plot (including RMSE)
plot_data <- melt(results_table[model %in% c("xgboost", "xgboost_reg")],
                  id.vars = c("subset", "horizon", "model", "n"),
                  measure.vars = c("accuracy", "macro_f1", "qwk", "rmse"),
                  variable.name = "metric", value.name = "value")
p_results_all_metrics <- ggplot(plot_data,
                                aes(x = horizon, y = value, color = subset,
                                    group = interaction(subset, model), linetype = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.8) +
  geom_text(aes(label = sprintf("%.3f", value)),
            vjust = -0.8, size = 3, show.legend = FALSE) +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  scale_color_viridis_d(option = "plasma", end = 0.7) +
  scale_linetype_manual(values = c("xgboost" = "solid", "xgboost_reg" = "dashed")) +
  labs(title = "XGBoost Performance Across Forecast Horizons",
       x = "Forecast Horizon", y = "Score", color = "Test subset", linetype = "Model") +
  theme_paper()
save_plot(p_results_all_metrics, "horizon_comparison_all_metrics", width = 14, height = 5)

cat("\nHorizon comparison tables and plots saved.\n")

# ---- 12. TRAINING / VALIDATION LOSS CURVES (classification only) ----

evals_log_table <- rbindlist(all_evals_log, fill = TRUE)
evals_log_table[, horizon := factor(horizon, levels = CONFIG$horizons)]
fwrite(evals_log_table, file.path(CONFIG$output_dir, "training_validation_curves.csv"))

metric_cols <- grep("mlogloss|logloss", names(evals_log_table), value = TRUE, ignore.case = TRUE)
if (length(metric_cols) == 0) {
  warning("No mlogloss columns found – cannot plot loss curves (24h is regression).")
} else {
  evals_long <- data.table::melt(
    evals_log_table,
    id.vars = setdiff(names(evals_log_table), metric_cols),
    measure.vars = metric_cols,
    variable.name = "set", value.name = "mlogloss"
  )
  if (!"set" %in% names(evals_long)) setnames(evals_long, "variable", "set")
  evals_long[, set := fifelse(grepl("train", set, ignore.case = TRUE), "train", "valid")]
  
  p_curves <- ggplot(evals_long, aes(x = iter, y = mlogloss, color = set)) +
    geom_line(linewidth = 0.9) +
    geom_vline(aes(xintercept = best_iteration), linetype = "dashed",
               color = "grey40", linewidth = 0.5) +
    facet_wrap(~horizon, scales = "free_y", nrow = 1) +
    scale_color_manual(values = c(train = "#2C3E50", valid = "#E67E22")) +
    labs(title = "Training vs. Validation Multiclass Log-Loss",
         subtitle = "Dashed line = best iteration (24h not shown – regression)",
         x = "Boosting round", y = "mlogloss", color = "Set") +
    theme_paper()
  save_plot(p_curves, "training_validation_curves", width = 14, height = 4.5)
  
  # Overfitting gap
  if (nrow(evals_long) > 0) {
    train_col <- metric_cols[grepl("train", metric_cols, ignore.case = TRUE)][1]
    valid_col <- metric_cols[grepl("valid|validation", metric_cols, ignore.case = TRUE)][1]
    if (!is.na(train_col) && !is.na(valid_col)) {
      gap_dt <- evals_log_table[iter == best_iteration,
                                .(horizon, train = get(train_col), valid = get(valid_col))]
      gap_dt[, gap := valid - train]
      p_gap <- ggplot(gap_dt, aes(x = horizon, y = gap, fill = horizon)) +
        geom_col(width = 0.6, show.legend = FALSE) +
        geom_text(aes(label = sprintf("%.3f", gap)), vjust = -0.5, size = 3.5) +
        scale_fill_viridis_d(option = "mako", end = 0.85) +
        labs(title = "Train-Validation Log-Loss Gap at Best Iteration",
             subtitle = "Larger gap = more overfitting (24h not shown)",
             x = "Forecast Horizon", y = "valid_mlogloss - train_mlogloss") +
        theme_paper()
      save_plot(p_gap, "overfitting_gap_by_horizon", width = 7, height = 5)
    }
  }
}
cat("\nTraining/validation curves + overfitting-gap plot saved.\n")

# ---- 13. SHAP IMPORTANCE HEATMAP + CATEGORY DOMINANCE + DIRECTIONAL ----

shap_table <- rbindlist(all_shap, fill = TRUE)

if (nrow(shap_table) == 0 || ncol(shap_table) == 0) {
  cat("WARNING: No SHAP data available. Skipping SHAP plots.\n")
} else {
  fwrite(shap_table, file.path(CONFIG$output_dir, "shap_mean_abs_by_horizon.csv"))
  
  feature_order <- shap_table[, .(total = sum(mean_abs_shap)), by = feature][order(-total)]$feature
  shap_table[, feature := factor(feature, levels = rev(feature_order))]
  shap_table[, horizon := factor(horizon, levels = CONFIG$horizons)]
  
  # Heatmap
  p_shap_heat <- ggplot(shap_table, aes(x = horizon, y = feature, fill = mean_abs_shap)) +
    geom_tile(color = "white", linewidth = 0.4) +
    scale_fill_viridis_c(option = "inferno", name = "Mean |SHAP|") +
    labs(title = "SHAP Feature Importance Across Forecast Horizons",
         subtitle = "Rows ordered by total importance (24h not shown)",
         x = "Forecast Horizon", y = NULL) +
    theme_paper() + theme(panel.grid = element_blank())
  save_plot(p_shap_heat, "shap_importance_heatmap", width = 8, height = 9)
  
  # Lollipop
  shap_top <- shap_table[, .SD[order(-mean_abs_shap)][1:min(CONFIG$shap_top_n, .N)], by = horizon]
  shap_top[, feature := reorder(feature, mean_abs_shap)]
  p_shap_lollipop <- ggplot(shap_top, aes(x = mean_abs_shap, y = feature, color = horizon)) +
    geom_segment(aes(x = 0, xend = mean_abs_shap, yend = feature), linewidth = 0.8) +
    geom_point(size = 2.8) +
    facet_wrap(~horizon, scales = "free", nrow = 1) +
    scale_color_viridis_d(option = "plasma", end = 0.8, guide = "none") +
    labs(title = paste0("Top ", CONFIG$shap_top_n, " Drivers by Horizon"),
         subtitle = "Mean absolute SHAP (24h not shown)",
         x = "Mean |SHAP|", y = NULL) +
    theme_paper()
  save_plot(p_shap_lollipop, "shap_top_features_by_horizon", width = 16, height = 5)
  
  # ---- Category Dominance ----
  pollutant_features <- c("pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                          "sulphur_dioxide", "ozone")
  meteorology_features <- c("temperature_2m", "relative_humidity_2m", "dew_point_2m",
                            "surface_pressure", "precipitation", "cloud_cover",
                            "wind_speed_10m", "wind_direction_10m")
  lagged_features <- c("us_aqi_lag_6h", "us_aqi_lag_24h", "pm2_5_lag_6h",
                       "pm2_5_lag_24h", "us_aqi_roll_24h", "us_aqi_change_6h",
                       "pm2_5_change_6h", "us_aqi_lag_48h", "pm2_5_lag_48h",
                       "us_aqi_lag_72h", "us_aqi_sd_24h", "us_aqi_min_24h",
                       "us_aqi_max_24h", "us_aqi_range_24h", "us_aqi_accel_6h")
  temporal_features <- c("month_bd_fixed", "hour_bd_fixed")
  spatial_features  <- c("latitude", "longitude")
  
  shap_table[, category := "Other"]
  shap_table[feature %in% pollutant_features,  category := "Pollutants (Raw)"]
  shap_table[feature %in% lagged_features,      category := "Lags & Trends"]
  shap_table[feature %in% meteorology_features, category := "Meteorology"]
  shap_table[feature %in% temporal_features,    category := "Temporal"]
  shap_table[feature %in% spatial_features,     category := "Spatial"]
  
  category_dominance <- shap_table[, .(total_importance = sum(mean_abs_shap)),
                                   by = .(horizon, category)]
  category_dominance[, pct := 100 * total_importance / sum(total_importance), by = horizon]
  category_dominance[, horizon := factor(horizon, levels = CONFIG$horizons)]
  
  p_category <- ggplot(category_dominance, aes(x = horizon, y = total_importance, fill = category)) +
    geom_col(width = 0.7, color = "white", linewidth = 0.3) +
    geom_text(aes(label = paste0(round(pct, 1), "%")),
              position = position_stack(vjust = 0.5), size = 3.2, color = "white") +
    scale_fill_viridis_d(option = "turbo", name = "Feature Category") +
    labs(title = "SHAP Importance Composition by Horizon",
         subtitle = "Pollutants dominate short-term; weather gains influence over time",
         x = "Forecast Horizon", y = "Total Mean |SHAP|") +
    theme_paper() + theme(legend.position = "bottom")
  save_plot(p_category, "shap_category_dominance", width = 9, height = 6)
  
  # ---- Directional SHAP for Very Unhealthy ----
  all_shap_directional <- list()
  for (h in CONFIG$horizons) {
    if (!h %in% names(all_shap)) next
    hd <- build_horizon_dataset(dt_model, h)
    test_seen_d <- hd[split_time == "test" & district_role == "seen"]
    shap_sample <- test_seen_d
    if (nrow(shap_sample) > CONFIG$shap_sample_size) {
      set.seed(CONFIG$seed)
      shap_sample <- shap_sample[sample(.N, CONFIG$shap_sample_size)]
    }
    xgb_model <- trained_models[[h]]
    dir_result <- tryCatch({
      contrib <- predict(xgb_model, newdata = as.matrix(shap_sample[, ..predictor_columns]),
                         predcontrib = TRUE)
      if (length(dim(contrib)) == 3) {
        class_slice <- contrib[, 5, ]  # class 4 = Very Unhealthy
        mean_shap <- colMeans(class_slice[, seq_len(length(predictor_columns)), drop = FALSE])
      } else {
        stop("Unexpected format")
      }
      res <- data.table(feature = predictor_columns, mean_shap = as.numeric(mean_shap))
      res[, horizon := h]
      res
    }, error = function(e) NULL)
    if (!is.null(dir_result)) all_shap_directional[[h]] <- dir_result
  }
  shap_directional <- rbindlist(all_shap_directional, fill = TRUE)
  if (nrow(shap_directional) > 0) {
    fwrite(shap_directional, file.path(CONFIG$output_dir, "shap_directional_very_unhealthy.csv"))
    top_pos <- shap_directional[order(-mean_shap)][1:15]
    top_pos[, feature := reorder(feature, mean_shap)]
    p_dir_pos <- ggplot(top_pos, aes(x = mean_shap, y = feature, fill = horizon)) +
      geom_col(orientation = "y", width = 0.7, color = "white", linewidth = 0.2) +
      scale_fill_viridis_d(option = "plasma", end = 0.8, name = "Horizon") +
      labs(title = "Top Drivers Pushing Towards 'Very Unhealthy'",
           subtitle = "Positive SHAP = increases risk", x = "Mean SHAP", y = NULL) +
      theme_paper()
    save_plot(p_dir_pos, "shap_drivers_towards_very_unhealthy", width = 9, height = 6)
    
    top_neg <- shap_directional[order(mean_shap)][1:15]
    top_neg[, feature := reorder(feature, -mean_shap)]
    p_dir_neg <- ggplot(top_neg, aes(x = mean_shap, y = feature, fill = horizon)) +
      geom_col(orientation = "y", width = 0.7, color = "white", linewidth = 0.2) +
      scale_fill_viridis_d(option = "plasma", end = 0.8, name = "Horizon") +
      labs(title = "Top Drivers Pushing AWAY from 'Very Unhealthy'",
           subtitle = "Negative SHAP = protective factors", x = "Mean SHAP", y = NULL) +
      theme_paper()
    save_plot(p_dir_neg, "shap_drivers_away_from_very_unhealthy", width = 9, height = 6)
    cat("Directional SHAP plots saved.\n")
  }
}

# ========================================================================
# 13.5 — BEESWARM PLOTS (POLLUTANTS vs WEATHER, SIGNED SHAP)
# ========================================================================
# Purpose:
#   Two separate beeswarm plots:
#     1. Pollutants only – shows which pollutant increases risk.
#     2. Weather only – shows which weather is protective (blue) or risk (red).
#   Uses SHAP contributions for the "Very Unhealthy" class (class index 4).
#   Caches SHAP data to avoid retraining every run.
# ========================================================================

cat("\n==============================================================\n")
cat("BEESWARM PLOTS — POLLUTANTS & WEATHER (VERY UNHEALTHY CLASS)\n")
cat("==============================================================\n")

# ---- 1. Define raw predictor sets ----
pollutant_features <- c(
  "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
  "sulphur_dioxide", "ozone"
)

weather_features <- c(
  "temperature_2m", "relative_humidity_2m", "dew_point_2m",
  "surface_pressure", "precipitation", "cloud_cover",
  "wind_speed_10m", "wind_direction_10m"
)

raw_predictor_columns <- c(pollutant_features, weather_features)

# ---- 2. Check if SHAP data already exists ----
shap_cache_path <- file.path(CONFIG$output_dir, "shap_raw_values_very_unhealthy.csv")

if (file.exists(shap_cache_path)) {
  cat("\nLoading cached SHAP data from:\n", shap_cache_path, "\n")
  beeswarm_data <- fread(shap_cache_path)
} else {
  cat("\nNo cached SHAP data found. Training raw-only models...\n")
  
  # ---- Train raw-only models ----
  trained_models_raw <- list()
  
  for (h in CONFIG$horizons) {
    if (h == "24h") next
    
    class_col <- paste0("aqi_class_", h)
    cols_needed <- c(raw_predictor_columns, class_col, "split_time", "district_role")
    d <- dt_model[!is.na(get(class_col)), ..cols_needed]
    d <- d[complete.cases(d[, ..raw_predictor_columns])]
    setnames(d, class_col, "aqi_class")
    d[, aqi_class_int := as.integer(aqi_class) - 1L]
    
    train_d <- d[split_time == "train"]
    validation_d <- d[split_time == "validation"]
    
    train_weights <- compute_class_weights(train_d$aqi_class)
    
    dtrain <- xgb.DMatrix(
      data = as.matrix(train_d[, ..raw_predictor_columns]),
      label = train_d$aqi_class_int,
      weight = train_weights
    )
    dvalid <- xgb.DMatrix(
      data = as.matrix(validation_d[, ..raw_predictor_columns]),
      label = validation_d$aqi_class_int
    )
    
    xgb_raw <- xgb.train(
      params = list(
        objective = "multi:softprob",
        num_class = n_classes_total,
        eval_metric = "mlogloss",
        max_depth = 6,
        eta = 0.05,
        subsample = 0.7,
        colsample_bytree = 0.7,
        min_child_weight = 3
      ),
      data = dtrain,
      nrounds = 500,
      evals = list(train = dtrain, valid = dvalid),
      early_stopping_rounds = 30,
      verbose = 0
    )
    trained_models_raw[[h]] <- xgb_raw
    cat("Trained raw model for", h, "\n")
  }
  
  # ---- Extract SIGNED SHAP for "Very Unhealthy" class ----
  cat("\nExtracting signed SHAP for Very Unhealthy class...\n")
  all_beeswarm_data <- list()
  
  for (h in CONFIG$horizons) {
    if (h == "24h") next
    
    class_col <- paste0("aqi_class_", h)
    cols_needed <- c(raw_predictor_columns, class_col, "split_time", "district_role")
    d <- dt_model[!is.na(get(class_col)), ..cols_needed]
    d <- d[complete.cases(d[, ..raw_predictor_columns])]
    setnames(d, class_col, "aqi_class")
    d[, aqi_class_int := as.integer(aqi_class) - 1L]
    
    test_seen_d <- d[split_time == "test" & district_role == "seen"]
    
    shap_sample <- test_seen_d
    if (nrow(shap_sample) > CONFIG$shap_sample_size) {
      set.seed(CONFIG$seed)
      shap_sample <- shap_sample[sample(.N, CONFIG$shap_sample_size)]
    }
    
    contrib <- predict(
      trained_models_raw[[h]],
      newdata = as.matrix(shap_sample[, ..raw_predictor_columns]),
      predcontrib = TRUE
    )
    
    # Extract SHAP for class 4 (Very Unhealthy)
    if (length(dim(contrib)) == 3) {
      class_slice <- contrib[, 5, ]  # index 4 -> 5th column (0-based)
      n_feat <- length(raw_predictor_columns)
      signed_shap <- class_slice[, seq_len(n_feat), drop = FALSE]
    } else {
      signed_shap <- contrib[, seq_len(length(raw_predictor_columns)), drop = FALSE]
    }
    
    dt_shap <- as.data.table(signed_shap)
    setnames(dt_shap, raw_predictor_columns)
    dt_shap[, horizon := h]
    dt_shap[, sample_id := .I]
    
    dt_long <- melt(
      dt_shap,
      id.vars = c("horizon", "sample_id"),
      variable.name = "feature",
      value.name = "shap_value"
    )
    
    all_beeswarm_data[[h]] <- dt_long
  }
  
  beeswarm_data <- rbindlist(all_beeswarm_data, fill = TRUE)
  
  # ---- Save cached SHAP data ----
  fwrite(beeswarm_data, shap_cache_path)
  cat("\nCached SHAP data saved to:\n", shap_cache_path, "\n")
}

# ---- Function to generate a beeswarm plot for a given feature set ----
plot_beeswarm <- function(data, feature_list, title, subtitle, filename) {
  
  plot_data <- data[feature %in% feature_list]
  
  # Order features by overall absolute importance
  feature_order <- plot_data[, .(mean_abs = mean(abs(shap_value))), by = feature][
    order(-mean_abs)]$feature
  plot_data[, feature := factor(feature, levels = rev(feature_order))]
  
  max_abs <- max(abs(plot_data$shap_value), na.rm = TRUE)
  
  p <- ggplot(plot_data, aes(x = shap_value, y = feature, color = shap_value)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
    geom_jitter(width = 0, height = 0.2, size = 1.0, alpha = 0.7) +
    scale_color_gradient2(
      low = "#2C3E50", mid = "white", high = "#E74C3C",
      midpoint = 0,
      limits = c(-max_abs, max_abs),
      name = "SHAP Value"
    ) +
    facet_wrap(~horizon, nrow = 1) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "SHAP Contribution (Very Unhealthy class)",
      y = NULL
    ) +
    theme_paper() +
    theme(
      legend.position = "bottom",
      legend.key.width = unit(1.8, "cm"),
      axis.text.y = element_text(face = "bold", size = 10),
      strip.text = element_text(face = "bold", size = 11)
    )
  
  return(p)
}

# ---- Generate and save Pollutant Beeswarm ----
p_pollutants <- plot_beeswarm(
  data = beeswarm_data,
  feature_list = pollutant_features,
  title = "Pollutant Impact on 'Very Unhealthy' AQI",
  subtitle = "Positive (red) = increases risk; Negative (blue) = decreases risk",
  filename = "shap_pollutants_beeswarm"
)
save_plot(p_pollutants, "shap_pollutants_beeswarm", width = 14, height = 5)
cat("Pollutant beeswarm saved.\n")

# ---- Generate and save Weather Beeswarm ----
p_weather <- plot_beeswarm(
  data = beeswarm_data,
  feature_list = weather_features,
  title = "Weather Impact on 'Very Unhealthy' AQI",
  subtitle = "Positive (red) = worsens AQI; Negative (blue) = improves AQI (dispersion/rain)",
  filename = "shap_weather_beeswarm"
)
save_plot(p_weather, "shap_weather_beeswarm", width = 14, height = 5)
cat("Weather beeswarm saved.\n")

cat("\nBoth beeswarm plots generated successfully.\n")

# ---- 14. PER-CLASS F1 HEATMAP ----

f1_heat_dt <- per_class_table[subset == "test_seen"]
f1_heat_dt[, class := factor(class, levels = CONFIG$category_levels)]
p_f1_heat <- ggplot(f1_heat_dt, aes(x = horizon, y = class, fill = f1)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", f1)),
            color = ifelse(f1_heat_dt$f1 > 0.6, "white", "grey15"), size = 3.6) +
  scale_fill_viridis_c(option = "viridis", name = "F1", limits = c(0, 1)) +
  labs(title = "Per-Class F1 Across Horizons (XGBoost)",
       subtitle = "Seen-district test set (24h uses regression → class mapping)",
       x = "Forecast Horizon", y = NULL) +
  theme_paper() + theme(panel.grid = element_blank())
save_plot(p_f1_heat, "per_class_f1_heatmap", width = 8, height = 5)

# ---- 15. GAIN vs SHAP RANK COMPARISON ----

importance_table <- rbindlist(all_importance, fill = TRUE)
fwrite(importance_table, file.path(CONFIG$output_dir, "gain_importance_by_horizon.csv"))

if (exists("shap_table") && nrow(shap_table) > 0) {
  rank_compare <- merge(
    importance_table[, .(feature = Feature, horizon, gain_rank = frank(-Gain)), ],
    shap_table[, .(feature, horizon, shap_rank = frank(-mean_abs_shap))],
    by = c("feature", "horizon")
  )
  p_rank <- ggplot(rank_compare, aes(x = gain_rank, y = shap_rank, color = horizon)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
    geom_point(size = 2.2, alpha = 0.85) +
    facet_wrap(~horizon, nrow = 1) +
    scale_color_viridis_d(option = "plasma", end = 0.8, guide = "none") +
    labs(title = "Gain-Importance vs. SHAP-Importance Ranking",
         subtitle = "Points on diagonal = agreement (24h not shown)",
         x = "Rank by Gain", y = "Rank by Mean |SHAP|") +
    theme_paper()
  save_plot(p_rank, "gain_vs_shap_rank_comparison", width = 15, height = 4)
} else {
  cat("Skipping gain-vs-SHAP rank comparison.\n")
}

# ---- 16. CLASS SUPPORT / DISTRIBUTION ----

support_dt <- per_class_table[subset == "test_seen", .(class, support, horizon)]
support_dt[, class := factor(class, levels = CONFIG$category_levels)]
support_dt[, pct := round(100 * support / sum(support), 1), by = horizon]
p_support <- ggplot(support_dt, aes(x = horizon, y = pct, fill = class)) +
  geom_col(position = "stack", width = 0.65, color = "white", linewidth = 0.3) +
  scale_fill_viridis_d(option = "turbo", name = "AQI class") +
  labs(title = "Target Class Distribution by Forecast Horizon",
       subtitle = "Seen-district test set",
       x = "Forecast Horizon", y = "Share of test rows (%)") +
  theme_paper()
save_plot(p_support, "class_distribution_by_horizon", width = 8, height = 5)

# ---- 17. SEEN vs UNSEEN DUMBBELL CHART ----

dumbbell_dt <- melt(
  results_table[model %in% c("xgboost", "xgboost_reg")],
  id.vars = c("subset", "horizon", "model"),
  measure.vars = c("macro_f1", "qwk"),
  variable.name = "metric", value.name = "value"
)
dumbbell_wide <- dcast(dumbbell_dt, horizon + metric ~ subset, value.var = "value")
p_dumbbell <- ggplot(dumbbell_wide) +
  geom_segment(aes(x = test_unseen, xend = test_seen, y = horizon, yend = horizon),
               color = "grey70", linewidth = 1) +
  geom_point(aes(x = test_seen,   y = horizon, color = "Seen districts"), size = 3.2) +
  geom_point(aes(x = test_unseen, y = horizon, color = "Unseen districts"), size = 3.2) +
  facet_wrap(~metric, nrow = 1, labeller = as_labeller(c(macro_f1 = "Macro-F1", qwk = "QWK"))) +
  scale_color_manual(name = NULL, values = c("Seen districts" = "#1B9E77",
                                             "Unseen districts" = "#D95F02")) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(title = "Spatial Generalization: Seen vs. Unseen Districts",
       subtitle = "XGBoost (24h regression marked as dashed)",
       x = "Score", y = "Forecast Horizon") +
  theme_paper()
save_plot(p_dumbbell, "seen_vs_unseen_generalization_dumbbell", width = 10, height = 5)

cat("\nAll figures saved.\n")

# ---- FINAL MESSAGE ----

cat("\n==============================================================\n")
cat("PIPELINE COMPLETE.\n")
cat("All outputs (tables + figures) saved to:\n")
cat("  ", CONFIG$output_dir, "\n")
cat("Figures saved to:\n")
cat("  ", CONFIG$plot_dir, "\n")
cat("Processed dataset saved as:\n")
cat("  ", model_dataset_path, "\n")
cat("==============================================================\n")












##############################################################################
# Multi-Horizon AQI Early-Warning System — SINGLE Unified Performance Plot
# Run in RStudio. Requires: ggplot2, dplyr, tidyr, scales
##############################################################################

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "scales")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ---- 1. Output path (EDIT if needed) ----------------------------------------
out_dir <- "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs/figures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_file <- file.path(out_dir, "AQI_model_performance.png")

# ---- 2. Data ------------------------------------------------------------
df <- tribble(
  ~Horizon, ~Model,           ~Subset,   ~Accuracy, ~MacroF1, ~QWK,   ~RMSE,
  "1h",     "Classification", "Seen",    0.990,     0.985,    0.995,  NA,
  "1h",     "Classification", "Unseen",  0.989,     0.984,    0.994,  NA,
  "3h",     "Classification", "Seen",    0.979,     0.966,    0.989,  NA,
  "3h",     "Classification", "Unseen",  0.978,     0.965,    0.988,  NA,
  "6h",     "Classification", "Seen",    0.951,     0.927,    0.975,  NA,
  "6h",     "Classification", "Unseen",  0.952,     0.924,    0.974,  NA,
  "12h",    "Classification", "Seen",    0.890,     0.839,    0.943,  NA,
  "12h",    "Classification", "Unseen",  0.890,     0.838,    0.939,  NA,
  "24h",    "Regression→Map", "Seen",    0.795,     0.712,    0.889,  14.43,
  "24h",    "Regression→Map", "Unseen",  0.797,     0.717,    0.880,  14.50
)

horizon_levels <- c("1h", "3h", "6h", "12h", "24h")
df <- df %>% mutate(Horizon = factor(Horizon, levels = horizon_levels))

# Long format: one row per Horizon x Subset x Metric
df_long <- df %>%
  select(Horizon, Subset, Accuracy, MacroF1, QWK) %>%
  pivot_longer(cols = c(Accuracy, MacroF1, QWK),
               names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = recode(Metric,
                         Accuracy = "Accuracy",
                         MacroF1  = "Macro-F1",
                         QWK      = "Quadratic Weighted Kappa (QWK)"),
         Metric = factor(Metric, levels = c("Accuracy", "Macro-F1",
                                            "Quadratic Weighted Kappa (QWK)")),
         SubsetLabel = ifelse(Subset == "Seen", "Seen districts", "Unseen districts"))

# ---- 3. Colour + linetype encoding (single panel, fully interpretable) -----
pal_metric <- c("Accuracy" = "#264653",
                "Macro-F1" = "#2A9D8F",
                "Quadratic Weighted Kappa (QWK)" = "#E76F51")

# ---- 4. Build ONE unified plot ----------------------------------------------
p <- ggplot(df_long, aes(x = Horizon, y = Score, color = Metric,
                         linetype = SubsetLabel, group = interaction(Metric, SubsetLabel))) +
  geom_line(linewidth = 1.15, alpha = 0.95) +
  geom_point(aes(shape = SubsetLabel), size = 3.4, stroke = 1) +
  geom_text(
    data = df_long %>% filter(SubsetLabel == "Seen districts"),
    aes(label = percent(Score, accuracy = 0.1)),
    vjust = -1.2, size = 3.1, fontface = "bold", show.legend = FALSE
  ) +
  scale_color_manual(values = pal_metric, name = "Metric") +
  scale_linetype_manual(values = c("Seen districts" = "solid",
                                   "Unseen districts" = "dashed"),
                        name = "District subset") +
  scale_shape_manual(values = c("Seen districts" = 16,
                                "Unseen districts" = 17),
                     name = "District subset") +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.65, 1.02),
                     expand = expansion(mult = c(0.02, 0.08))) +
  labs(
    title = "Multi-Horizon AQI Early-Warning System \u2014 Model Performance",
    subtitle = "Accuracy, Macro-F1 & QWK across 1h\u201324h horizons | Solid = Seen districts, Dashed = Unseen districts\n(near-overlapping lines confirm strong spatial generalisation)",
    x = "Forecast Horizon",
    y = "Score",
    caption = "24h horizon uses Regression\u2192Map approach (RMSE: Seen = 14.43, Unseen = 14.50 AQI points), not shown on this % scale.\nClassification used for 1h\u201312h horizons."
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 17, hjust = 0.5),
    plot.subtitle    = element_text(size = 10.5, color = "grey30", hjust = 0.5,
                                    margin = margin(b = 12)),
    plot.caption     = element_text(size = 8.7, color = "grey50", hjust = 0.5,
                                    margin = margin(t = 10)),
    axis.title       = element_text(face = "bold", size = 11),
    axis.text        = element_text(size = 10.5),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position  = "bottom",
    legend.box       = "vertical",
    legend.title     = element_text(face = "bold", size = 9.5),
    legend.text      = element_text(size = 9),
    plot.margin      = margin(15, 20, 10, 15)
  ) +
  guides(color = guide_legend(order = 1, override.aes = list(linetype = "solid")),
         linetype = guide_legend(order = 2),
         shape = guide_legend(order = 2))

# ---- 5. Save as ONE image file -----------------------------------------------
ggsave(out_file, p, width = 10.5, height = 7.5, dpi = 320, bg = "white")

cat("Saved single plot to:\n", out_file, "\n")

# Print to RStudio Plots pane
print(p)



































##############################################################################
# Multi-Horizon AQI Early-Warning System — Unified Heatmap Performance Plot
# ONE single image, ZERO overlapping labels, unique tile-matrix design.
# Run in RStudio. Requires: ggplot2, dplyr, tidyr, scales, patchwork
##############################################################################

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "scales", "patchwork")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ---- 1. Output path (EDIT if needed) ----------------------------------------
out_dir <- "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs/figures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_file <- file.path(out_dir, "AQI_model_performance1.png")

# ---- 2. Data ------------------------------------------------------------
df <- tribble(
  ~Horizon, ~Model,           ~Subset,   ~Accuracy, ~MacroF1, ~QWK,   ~RMSE,
  "1h",     "Classification", "Seen",    0.990,     0.985,    0.995,  NA,
  "1h",     "Classification", "Unseen",  0.989,     0.984,    0.994,  NA,
  "3h",     "Classification", "Seen",    0.979,     0.966,    0.989,  NA,
  "3h",     "Classification", "Unseen",  0.978,     0.965,    0.988,  NA,
  "6h",     "Classification", "Seen",    0.951,     0.927,    0.975,  NA,
  "6h",     "Classification", "Unseen",  0.952,     0.924,    0.974,  NA,
  "12h",    "Classification", "Seen",    0.890,     0.839,    0.943,  NA,
  "12h",    "Classification", "Unseen",  0.890,     0.838,    0.939,  NA,
  "24h",    "Regression→Map", "Seen",    0.795,     0.712,    0.889,  14.43,
  "24h",    "Regression→Map", "Unseen",  0.797,     0.717,    0.880,  14.50
)

horizon_levels <- c("1h", "3h", "6h", "12h", "24h")
df <- df %>% mutate(Horizon = factor(Horizon, levels = horizon_levels))

# ---- 3. Long format: one row per Horizon x Subset x Metric ------------------
df_long <- df %>%
  select(Horizon, Subset, Accuracy, MacroF1, QWK) %>%
  pivot_longer(cols = c(Accuracy, MacroF1, QWK),
               names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = recode(Metric,
                         Accuracy = "Accuracy",
                         MacroF1  = "Macro-F1",
                         QWK      = "QWK"),
         Metric = factor(Metric, levels = c("QWK", "Macro-F1", "Accuracy")),
         SubsetLabel = factor(ifelse(Subset == "Seen", "Seen districts",
                                     "Unseen districts"),
                              levels = c("Seen districts", "Unseen districts")))

# ---- 4. MAIN PANEL: tile-matrix heatmap (no overlap, each cell = 1 value) --
main_panel <- ggplot(df_long, aes(x = Horizon, y = Metric, fill = Score)) +
  geom_tile(color = "white", linewidth = 1.4, width = 0.92, height = 0.82) +
  geom_text(aes(label = percent(Score, accuracy = 0.1),
                color = after_scale(ifelse(Score > 0.90, "white", "grey10"))),
            fontface = "bold", size = 4.1) +
  facet_wrap(~SubsetLabel, ncol = 2) +
  scale_fill_gradientn(
    colors = c("#D73027", "#FEE08B", "#1A9850"),
    values = rescale(c(0.65, 0.85, 1.0)),
    limits = c(0.65, 1.0),
    labels = percent_format(accuracy = 1),
    name = "Score",
    guide = guide_colorbar(barwidth = 14, barheight = 0.6, title.position = "top")
  ) +
  scale_color_identity() +
  labs(
    title = "Multi-Horizon AQI Early-Warning System \u2014 Model Performance Matrix",
    subtitle = "Accuracy, Macro-F1 & QWK across forecast horizons (1h\u201312h: classification) | Colour = score, darker green is better",
    x = "Forecast Horizon (lead time)",
    y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title        = element_text(face = "bold", size = 16.5, hjust = 0.5),
    plot.subtitle     = element_text(size = 10.3, color = "grey30", hjust = 0.5,
                                     margin = margin(b = 10)),
    strip.text        = element_text(face = "bold", size = 12, color = "grey15"),
    strip.background  = element_rect(fill = "grey93", color = NA),
    axis.title.x      = element_text(face = "bold", size = 10.5, margin = margin(t = 8)),
    axis.text         = element_text(size = 10.5, face = "bold"),
    panel.grid        = element_blank(),
    legend.position   = "bottom",
    legend.title      = element_text(face = "bold", size = 9.5, hjust = 0.5),
    legend.text       = element_text(size = 8.5),
    plot.margin       = margin(15, 15, 5, 15)
  )

# ---- 5. SIDE PANEL: 24h RMSE (separate scale, shown as its own small chart) -
df_rmse <- df %>% filter(Horizon == "24h") %>%
  mutate(SubsetLabel = factor(ifelse(Subset == "Seen", "Seen", "Unseen"),
                              levels = c("Seen", "Unseen")))

rmse_panel <- ggplot(df_rmse, aes(x = SubsetLabel, y = RMSE, fill = SubsetLabel)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.2f", RMSE)), vjust = -0.6,
            fontface = "bold", size = 3.6) +
  scale_fill_manual(values = c("Seen" = "#1A9850", "Unseen" = "#D73027")) +
  scale_y_continuous(limits = c(0, max(df_rmse$RMSE) * 1.3), expand = c(0, 0)) +
  labs(title = "24h Horizon \u2014 Regression\u2192Map RMSE",
       x = NULL, y = "RMSE (AQI points)") +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 10.5, hjust = 0.5),
    axis.title.y  = element_text(size = 9, face = "bold"),
    axis.text     = element_text(size = 9.5, face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# ---- 6. Combine into ONE final image (patchwork inset) ----------------------
final_plot <- main_panel + inset_element(rmse_panel, left = 0.72, bottom = 0.02,
                                         right = 0.995, top = 0.30,
                                         align_to = "full")

# ---- 7. Save as ONE image file -----------------------------------------------
ggsave(out_file, final_plot, width = 11, height = 7.2, dpi = 320, bg = "white")

cat("Saved single plot to:\n", out_file, "\n")

# Print to RStudio Plots pane
print(final_plot)



















##############################################################################
# Multi-Horizon AQI Early-Warning System — Modern Radial Performance Plot
# ONE single image | Dark, stylish, circular "radar-ring" design
# Run in RStudio. Requires: ggplot2, dplyr, tidyr, scales
##############################################################################

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "scales")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ---- 1. Output path (EDIT if needed) ----------------------------------------
out_dir <- "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs/figures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_file <- file.path(out_dir, "AQI_model_performance2.png")

# ---- 2. Data ------------------------------------------------------------
df <- tribble(
  ~Horizon, ~Model,           ~Subset,   ~Accuracy, ~MacroF1, ~QWK,   ~RMSE,
  "1h",     "Classification", "Seen",    0.990,     0.985,    0.995,  NA,
  "1h",     "Classification", "Unseen",  0.989,     0.984,    0.994,  NA,
  "3h",     "Classification", "Seen",    0.979,     0.966,    0.989,  NA,
  "3h",     "Classification", "Unseen",  0.978,     0.965,    0.988,  NA,
  "6h",     "Classification", "Seen",    0.951,     0.927,    0.975,  NA,
  "6h",     "Classification", "Unseen",  0.952,     0.924,    0.974,  NA,
  "12h",    "Classification", "Seen",    0.890,     0.839,    0.943,  NA,
  "12h",    "Classification", "Unseen",  0.890,     0.838,    0.939,  NA,
  "24h",    "Regression→Map", "Seen",    0.795,     0.712,    0.889,  14.43,
  "24h",    "Regression→Map", "Unseen",  0.797,     0.717,    0.880,  14.50
)

horizon_levels <- c("1h", "3h", "6h", "12h", "24h")
df <- df %>% mutate(Horizon = factor(Horizon, levels = horizon_levels))

# ---- 3. Long format -----------------------------------------------------
df_long <- df %>%
  select(Horizon, Subset, Accuracy, MacroF1, QWK) %>%
  pivot_longer(cols = c(Accuracy, MacroF1, QWK),
               names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = recode(Metric, Accuracy = "Accuracy",
                         MacroF1 = "Macro-F1", QWK = "QWK"),
         # Ring order: QWK innermost, Accuracy outermost
         Metric = factor(Metric, levels = c("QWK", "Macro-F1", "Accuracy")),
         RingNum = as.numeric(Metric),                # 1,2,3
         SubsetLabel = factor(ifelse(Subset == "Seen", "Seen districts",
                                     "Unseen districts"),
                              levels = c("Seen districts", "Unseen districts")))

# Bar height baseline offset so rings sit at radius 1-2-3 (donut, not pie)
df_long <- df_long %>%
  mutate(ymin = RingNum - 0.42, ymax = RingNum - 0.42 + Score * 0.80)

# ---- 4. Modern dark theme ----------------------------------------------------
bg_col     <- "#0E1117"
panel_col  <- "#0E1117"
grid_col   <- "#2A2F3A"
text_col   <- "#E8EAED"
subtext_col<- "#9AA0A8"

theme_radial <- theme_minimal(base_size = 13) +
  theme(
    plot.background   = element_rect(fill = bg_col, color = NA),
    panel.background  = element_rect(fill = panel_col, color = NA),
    panel.grid        = element_blank(),
    axis.text         = element_blank(),
    axis.title        = element_blank(),
    axis.ticks        = element_blank(),
    strip.text        = element_text(color = text_col, face = "bold", size = 13,
                                     margin = margin(b = 6)),
    strip.background  = element_blank(),
    plot.title        = element_text(color = text_col, face = "bold", size = 18,
                                     hjust = 0.5, margin = margin(t = 6, b = 2)),
    plot.subtitle     = element_text(color = subtext_col, size = 10.5, hjust = 0.5,
                                     margin = margin(b = 10)),
    plot.caption      = element_text(color = subtext_col, size = 8.3, hjust = 0.5,
                                     margin = margin(t = 10)),
    legend.position   = "bottom",
    legend.background = element_rect(fill = bg_col, color = NA),
    legend.title      = element_text(color = text_col, face = "bold", size = 9),
    legend.text       = element_text(color = subtext_col, size = 8.5),
    plot.margin       = margin(10, 10, 10, 10)
  )

# ---- 5. Neon gradient palette -------------------------------------------------
neon_grad <- c("#3A0CA3", "#4361EE", "#4CC9F0", "#F72585")
neon_grad <- c("#F72585", "#7209B7", "#3A0CA3", "#4CC9F0")  # low->high reorder below
grad_colors <- c("#FF4365", "#FCA652", "#4CC9F0", "#06D6A0")

# ---- 6. Build the radial ring chart ------------------------------------------
p <- ggplot(df_long) +
  geom_rect(aes(xmin = as.numeric(Horizon) - 0.44, xmax = as.numeric(Horizon) + 0.44,
                ymin = RingNum - 0.42, ymax = ymax, fill = Score),
            color = bg_col, linewidth = 1.1) +
  # faint full-ring track behind bars for context
  geom_rect(aes(xmin = as.numeric(Horizon) - 0.44, xmax = as.numeric(Horizon) + 0.44,
                ymin = RingNum - 0.42, ymax = RingNum - 0.42 + 0.80),
            fill = NA, color = grid_col, linewidth = 0.4, linetype = "dotted") +
  geom_text(aes(x = as.numeric(Horizon), y = ymax + 0.10,
                label = percent(Score, accuracy = 0.1)),
            color = text_col, size = 3.0, fontface = "bold") +
  # ring labels (QWK/Macro-F1/Accuracy) placed near center
  geom_text(data = df_long %>% distinct(Metric, RingNum),
            aes(x = 0.35, y = RingNum - 0.05, label = Metric),
            color = subtext_col, size = 3.1, fontface = "bold", hjust = 1) +
  # horizon labels around the outer rim
  geom_text(data = df_long %>% distinct(Horizon),
            aes(x = as.numeric(Horizon), y = 3.75, label = Horizon),
            color = text_col, size = 4.3, fontface = "bold") +
  facet_wrap(~SubsetLabel, ncol = 2) +
  coord_polar(theta = "x", start = -0.15, clip = "off") +
  scale_x_continuous(limits = c(0, 5.5)) +
  scale_y_continuous(limits = c(-0.5, 4.0)) +
  scale_fill_gradientn(
    colors = grad_colors,
    values = rescale(c(0.65, 0.80, 0.92, 1.0)),
    limits = c(0.65, 1.0),
    labels = percent_format(accuracy = 1),
    name = "Score",
    guide = guide_colorbar(barwidth = 16, barheight = 0.55, title.position = "top",
                           title.hjust = 0.5)
  ) +
  labs(
    title = "AQI Early-Warning System \u2014 Radial Performance Rings",
    subtitle = "Each ring = one metric (QWK \u2192 Macro-F1 \u2192 Accuracy, inner to outer) \u00b7 Each wedge = one forecast horizon \u00b7 Bar length & colour = score",
    caption = paste0(
      "24h horizon uses Regression\u2192Map (not shown in %): RMSE Seen = 14.43, Unseen = 14.50 AQI points.  ",
      "Nearly identical rings across panels confirm strong spatial generalisation to unseen districts."
    )
  ) +
  theme_radial

# ---- 7. Save as ONE image file -----------------------------------------------
ggsave(out_file, p, width = 11.5, height = 7.5, dpi = 320, bg = bg_col)

cat("Saved single plot to:\n", out_file, "\n")

# Print to RStudio Plots pane
print(p)















##############################################################################
# Multi-Horizon AQI Early-Warning System — Modern Radial Performance Plot
# ONE single image | Clean, stylish, circular "radar-ring" design (WHITE bg)
# Run in RStudio. Requires: ggplot2, dplyr, tidyr, scales
##############################################################################

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "scales")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ---- 1. Output path (EDIT if needed) ----------------------------------------
out_dir <- "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs/figures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_file <- file.path(out_dir, "AQI_model_performance2.1.png")

# ---- 2. Data ------------------------------------------------------------
df <- tribble(
  ~Horizon, ~Model,           ~Subset,   ~Accuracy, ~MacroF1, ~QWK,   ~RMSE,
  "1h",     "Classification", "Seen",    0.990,     0.985,    0.995,  NA,
  "1h",     "Classification", "Unseen",  0.989,     0.984,    0.994,  NA,
  "3h",     "Classification", "Seen",    0.979,     0.966,    0.989,  NA,
  "3h",     "Classification", "Unseen",  0.978,     0.965,    0.988,  NA,
  "6h",     "Classification", "Seen",    0.951,     0.927,    0.975,  NA,
  "6h",     "Classification", "Unseen",  0.952,     0.924,    0.974,  NA,
  "12h",    "Classification", "Seen",    0.890,     0.839,    0.943,  NA,
  "12h",    "Classification", "Unseen",  0.890,     0.838,    0.939,  NA,
  "24h",    "Regression→Map", "Seen",    0.795,     0.712,    0.889,  14.43,
  "24h",    "Regression→Map", "Unseen",  0.797,     0.717,    0.880,  14.50
)

horizon_levels <- c("1h", "3h", "6h", "12h", "24h")
df <- df %>% mutate(Horizon = factor(Horizon, levels = horizon_levels))

# ---- 3. Long format -----------------------------------------------------
df_long <- df %>%
  select(Horizon, Subset, Accuracy, MacroF1, QWK) %>%
  pivot_longer(cols = c(Accuracy, MacroF1, QWK),
               names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = recode(Metric, Accuracy = "Accuracy",
                         MacroF1 = "Macro-F1", QWK = "QWK"),
         # Ring order: QWK innermost, Accuracy outermost
         Metric = factor(Metric, levels = c("QWK", "Macro-F1", "Accuracy")),
         RingNum = as.numeric(Metric),                # 1,2,3
         SubsetLabel = factor(ifelse(Subset == "Seen", "Seen districts",
                                     "Unseen districts"),
                              levels = c("Seen districts", "Unseen districts")))

# Bar height baseline offset so rings sit at radius 1-2-3 (donut, not pie)
df_long <- df_long %>%
  mutate(ymin = RingNum - 0.42, ymax = RingNum - 0.42 + Score * 0.80)

# ---- 4. Clean WHITE theme (same structure, colours flipped for white bg) ----
bg_col     <- "#FFFFFF"   # was "#0E1117"
panel_col  <- "#FFFFFF"   # was "#0E1117"
grid_col   <- "#D9DCE1"   # was "#2A2F3A" (dark grid on dark bg -> light grid on white)
text_col   <- "#1F2430"   # was "#E8EAED" (light text on dark -> dark text on white)
subtext_col<- "#6B7280"   # was "#9AA0A8" (kept mid-grey, darkened slightly for contrast)

theme_radial <- theme_minimal(base_size = 13) +
  theme(
    plot.background   = element_rect(fill = bg_col, color = NA),
    panel.background  = element_rect(fill = panel_col, color = NA),
    panel.grid        = element_blank(),
    axis.text         = element_blank(),
    axis.title        = element_blank(),
    axis.ticks        = element_blank(),
    strip.text        = element_text(color = text_col, face = "bold", size = 13,
                                     margin = margin(b = 6)),
    strip.background  = element_blank(),
    plot.title        = element_text(color = text_col, face = "bold", size = 18,
                                     hjust = 0.5, margin = margin(t = 6, b = 2)),
    plot.subtitle     = element_text(color = subtext_col, size = 10.5, hjust = 0.5,
                                     margin = margin(b = 10)),
    plot.caption      = element_text(color = subtext_col, size = 8.3, hjust = 0.5,
                                     margin = margin(t = 10)),
    legend.position   = "bottom",
    legend.background = element_rect(fill = bg_col, color = NA),
    legend.title      = element_text(color = text_col, face = "bold", size = 9),
    legend.text       = element_text(color = subtext_col, size = 8.5),
    plot.margin       = margin(10, 10, 10, 10)
  )

# ---- 5. Same vivid gradient palette (reads well on both dark and white) -----
grad_colors <- c("#FF4365", "#FCA652", "#06AED5", "#06D6A0")

# ---- 6. Build the radial ring chart (structure identical to original) ------
p <- ggplot(df_long) +
  geom_rect(aes(xmin = as.numeric(Horizon) - 0.44, xmax = as.numeric(Horizon) + 0.44,
                ymin = RingNum - 0.42, ymax = ymax, fill = Score),
            color = bg_col, linewidth = 1.1) +
  # faint full-ring track behind bars for context
  geom_rect(aes(xmin = as.numeric(Horizon) - 0.44, xmax = as.numeric(Horizon) + 0.44,
                ymin = RingNum - 0.42, ymax = RingNum - 0.42 + 0.80),
            fill = NA, color = grid_col, linewidth = 0.4, linetype = "dotted") +
  geom_text(aes(x = as.numeric(Horizon), y = ymax + 0.10,
                label = percent(Score, accuracy = 0.1)),
            color = text_col, size = 3.0, fontface = "bold") +
  # ring labels (QWK/Macro-F1/Accuracy) placed near center
  geom_text(data = df_long %>% distinct(Metric, RingNum),
            aes(x = 0.35, y = RingNum - 0.05, label = Metric),
            color = subtext_col, size = 3.1, fontface = "bold", hjust = 1) +
  # horizon labels around the outer rim
  geom_text(data = df_long %>% distinct(Horizon),
            aes(x = as.numeric(Horizon), y = 3.75, label = Horizon),
            color = text_col, size = 4.3, fontface = "bold") +
  facet_wrap(~SubsetLabel, ncol = 2) +
  coord_polar(theta = "x", start = -0.15, clip = "off") +
  scale_x_continuous(limits = c(0, 5.5)) +
  scale_y_continuous(limits = c(-0.5, 4.0)) +
  scale_fill_gradientn(
    colors = grad_colors,
    values = rescale(c(0.65, 0.80, 0.92, 1.0)),
    limits = c(0.65, 1.0),
    labels = percent_format(accuracy = 1),
    name = "Score",
    guide = guide_colorbar(barwidth = 16, barheight = 0.55, title.position = "top",
                           title.hjust = 0.5)
  ) +
  labs(
    title = "AQI Early-Warning System \u2014 Radial Performance Rings",
    subtitle = "Each ring = one metric (QWK \u2192 Macro-F1 \u2192 Accuracy, inner to outer) \u00b7 Each wedge = one forecast horizon \u00b7 Bar length & colour = score",
    caption = paste0(
      "24h horizon uses Regression\u2192Map (not shown in %): RMSE Seen = 14.43, Unseen = 14.50 AQI points.  ",
      "Nearly identical rings across panels confirm strong spatial generalisation to unseen districts."
    )
  ) +
  theme_radial

# ---- 7. Save as ONE image file -----------------------------------------------
ggsave(out_file, p, width = 11.5, height = 7.5, dpi = 320, bg = bg_col)

cat("Saved single plot to:\n", out_file, "\n")

# Print to RStudio Plots pane
print(p)































##############################################################################
# Multi-Horizon AQI Early-Warning System — Modern Radial Performance Plot
# FINAL VERSION 2: bigger donuts, exact polar geometry, guaranteed no overlap
# Run in RStudio. Requires: ggplot2, dplyr, tidyr, scales, patchwork
##############################################################################

# ---- 0. Packages -----------------------------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "scales", "patchwork")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ---- 1. Output path (EDIT if needed) ----------------------------------------
out_dir <- "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs/figures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_file <- file.path(out_dir, "AQI_model_performance3.png")

# ---- 2. Data ------------------------------------------------------------
df <- tribble(
  ~Horizon, ~Model,           ~Subset,   ~Accuracy, ~MacroF1, ~QWK,   ~RMSE,
  "1h",     "Classification", "Seen",    0.990,     0.985,    0.995,  NA,
  "1h",     "Classification", "Unseen",  0.989,     0.984,    0.994,  NA,
  "3h",     "Classification", "Seen",    0.979,     0.966,    0.989,  NA,
  "3h",     "Classification", "Unseen",  0.978,     0.965,    0.988,  NA,
  "6h",     "Classification", "Seen",    0.951,     0.927,    0.975,  NA,
  "6h",     "Classification", "Unseen",  0.952,     0.924,    0.974,  NA,
  "12h",    "Classification", "Seen",    0.890,     0.839,    0.943,  NA,
  "12h",    "Classification", "Unseen",  0.890,     0.838,    0.939,  NA,
  "24h",    "Regression→Map", "Seen",    0.795,     0.712,    0.889,  14.43,
  "24h",    "Regression→Map", "Unseen",  0.797,     0.717,    0.880,  14.50
)

horizon_levels <- c("1h", "3h", "6h", "12h", "24h")
df <- df %>% mutate(Horizon = factor(Horizon, levels = horizon_levels))

# ---- 3. Long format -----------------------------------------------------
df_long <- df %>%
  select(Horizon, Subset, Accuracy, MacroF1, QWK) %>%
  pivot_longer(cols = c(Accuracy, MacroF1, QWK),
               names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = recode(Metric, Accuracy = "Accuracy",
                         MacroF1 = "Macro-F1", QWK = "QWK"),
         Metric = factor(Metric, levels = c("QWK", "Macro-F1", "Accuracy")),
         HorizonNum = as.numeric(Horizon),             # 1..5, exact category slot
         SubsetLabel = factor(ifelse(Subset == "Seen", "Seen districts",
                                     "Unseen districts"),
                              levels = c("Seen districts", "Unseen districts")))

# ---- 4. EXACT ring geometry (hand-specified, no formula drift) --------------
# Each ring has a fixed start radius and a fixed max height -> fully predictable
ring_info <- tibble(
  Metric     = factor(c("QWK", "Macro-F1", "Accuracy"),
                      levels = c("QWK", "Macro-F1", "Accuracy")),
  ring_base  = c(0.35, 1.55, 2.75),   # inner edge of each ring band
  ring_h     = c(1.05, 1.05, 1.05)    # max bar height (score = 1.0) per ring
)

df_long <- df_long %>%
  left_join(ring_info, by = "Metric") %>%
  mutate(ymin = ring_base,
         ymax = ring_base + Score * ring_h)

WEDGE_HALF <- 0.46           # bar half-width in x-units (gap = 1 - 2*WEDGE_HALF)
HORIZON_LABEL_Y <- 4.10      # radius for the outer 1h/3h/6h/12h/24h labels
Y_MAX <- 4.35                # tight top limit (just past the horizon labels)
Y_MIN <- -0.35               # small centre hole

# ---- 5. Clean, modern WHITE theme --------------------------------------------
bg_col      <- "#FFFFFF"
grid_col    <- "#D8DCE2"
text_col    <- "#1F2430"
subtext_col <- "#6B7280"
label_col   <- "#2B2F38"

theme_radial <- theme_minimal(base_size = 13) +
  theme(
    plot.background   = element_rect(fill = bg_col, color = NA),
    panel.background  = element_rect(fill = bg_col, color = NA),
    panel.grid        = element_blank(),
    axis.text         = element_blank(),
    axis.title        = element_blank(),
    axis.ticks        = element_blank(),
    strip.text        = element_text(color = text_col, face = "bold", size = 13.5,
                                     margin = margin(b = 0)),
    strip.background  = element_blank(),
    legend.position   = "bottom",
    legend.background = element_rect(fill = bg_col, color = NA),
    legend.title      = element_text(color = text_col, face = "bold", size = 9),
    legend.text       = element_text(color = subtext_col, size = 8.3),
    legend.margin     = margin(0, 0, 0, 0),
    legend.box.spacing = unit(0, "pt"),
    panel.spacing     = unit(0.3, "lines"),
    plot.margin       = margin(0, 10, 0, 10)
  )

# ---- 6. Modern vivid gradient (works well on white) --------------------------
grad_colors <- c("#EF476F", "#FFB238", "#06AED5", "#0B7A75")

# ---- 7. MAIN radial chart (donuts only — no in-plot ring-name text) --------
p_main <- ggplot(df_long) +
  # faint full-ring track (reference max) behind each bar
  geom_rect(aes(xmin = HorizonNum - WEDGE_HALF, xmax = HorizonNum + WEDGE_HALF,
                ymin = ring_base, ymax = ring_base + ring_h),
            fill = "#F2F3F5", color = grid_col, linewidth = 0.3, linetype = "dotted") +
  # actual score bars
  geom_rect(aes(xmin = HorizonNum - WEDGE_HALF, xmax = HorizonNum + WEDGE_HALF,
                ymin = ymin, ymax = ymax, fill = Score),
            color = bg_col, linewidth = 1.3) +
  # score value labels — placed INSIDE each bar's own segment (small white chip
  # for guaranteed contrast against every bar colour), centred on the bar
  geom_label(aes(x = HorizonNum, y = (ymin + ymax) / 2,
                 label = percent(Score, accuracy = 0.1)),
             fill = "white", color = label_col, label.size = 0,
             label.padding = unit(0.14, "lines"), label.r = unit(0.09, "lines"),
             alpha = 0.88, size = 2.85, fontface = "bold") +
  # horizon labels around the rim, one fixed radius, well clear of value labels
  geom_text(data = df_long %>% distinct(HorizonNum, Horizon),
            aes(x = HorizonNum, y = HORIZON_LABEL_Y, label = Horizon),
            color = text_col, size = 4.6, fontface = "bold") +
  facet_wrap(~SubsetLabel, ncol = 2) +
  coord_polar(theta = "x", start = -0.15, clip = "off") +
  scale_x_continuous(limits = c(0.5, 5.5), expand = c(0, 0)) +
  scale_y_continuous(limits = c(Y_MIN, Y_MAX), expand = c(0, 0)) +
  scale_fill_gradientn(
    colors = grad_colors,
    values = rescale(c(0.65, 0.80, 0.92, 1.0)),
    limits = c(0.65, 1.0),
    labels = percent_format(accuracy = 1),
    name = "Score",
    guide = guide_colorbar(barwidth = 13, barheight = 0.42, title.position = "top",
                           title.hjust = 0.5)
  ) +
  theme_radial

# ---- 8. Small standalone RING KEY (fully separate from the polar plot, ----
#         so it is geometrically impossible for it to overlap anything)
key_df <- ring_info %>%
  mutate(y = c(1, 2, 3),                # QWK (innermost) first, then outward
         Label = paste0(Metric, ifelse(Metric == "QWK", "  (innermost ring)",
                                       ifelse(Metric == "Accuracy", "  (outermost ring)", ""))))

key_colors <- c("QWK" = "#0B7A75", "Macro-F1" = "#FFB238", "Accuracy" = "#EF476F")

# Single horizontal row (not stacked) — far more compact than 3 stacked rows
p_key <- ggplot(key_df, aes(x = y, y = 0)) +
  geom_point(aes(color = Metric), size = 4.5) +
  geom_text(aes(label = Label), hjust = 0, nudge_x = 0.09, size = 3.55,
            color = text_col, fontface = "bold") +
  scale_color_manual(values = key_colors, guide = "none") +
  scale_x_continuous(limits = c(0.7, 4.6)) +
  scale_y_continuous(limits = c(-0.5, 0.5)) +
  coord_cartesian(clip = "off") +
  theme_void() +
  theme(plot.margin = margin(0, 0, 0, 0))

# ---- 9. Compose title + key + donuts + caption as ONE figure ----------------
final_plot <-
  (p_key) / p_main +
  plot_layout(heights = c(0.05, 1)) +
  plot_annotation(
    title = "AQI Early-Warning System \u2014 Radial Performance Rings",
    subtitle = "Each ring = one metric, inner to outer (see key)  \u00b7  Each wedge = one forecast horizon  \u00b7  Bar length & colour = score",
    caption = paste0(
      "24h horizon uses Regression\u2192Map (not shown in %): RMSE Seen = 14.43, Unseen = 14.50 AQI points.   ",
      "Nearly identical rings across panels confirm strong spatial generalisation to unseen districts."
    ),
    theme = theme(
      plot.background = element_rect(fill = bg_col, color = NA),
      plot.title    = element_text(color = text_col, face = "bold", size = 18,
                                   hjust = 0.5, margin = margin(t = 0, b = 1)),
      plot.subtitle = element_text(color = subtext_col, size = 10.5, hjust = 0.5,
                                   margin = margin(b = 2)),
      plot.caption  = element_text(color = subtext_col, size = 8.5, hjust = 0.5,
                                   margin = margin(t = 0)),
      plot.margin   = margin(4, 20, 2, 20)
    )
  ) &
  theme(plot.background = element_rect(fill = bg_col, color = NA))

# ---- 10. Save as ONE image file (compact — minimal outer whitespace) --------
ggsave(out_file, final_plot, width = 12.5, height = 6.3, dpi = 320, bg = bg_col)

cat("Saved single plot to:\n", out_file, "\n")

# Print to RStudio Plots pane
print(final_plot)


















# ============================================================
# AQI Forecasting Performance Across Horizons
# Correct Final Results
# Accuracy, Macro-F1, QWK + 24-h RMSE
# ============================================================

library(ggplot2)
library(tidyr)
library(dplyr)
library(grid)

# ------------------------------------------------------------
# 1. Correct final results from model output
# ------------------------------------------------------------

results <- data.frame(
  Horizon = factor(
    c("1h", "3h", "6h", "12h", "24h"),
    levels = c("1h", "3h", "6h", "12h", "24h")
  ),
  
  # -------------------------
  # Seen districts
  # -------------------------
  Seen_Accuracy = c(
    0.9898,
    0.9787,
    0.9511,
    0.8899,
    0.7954
  ),
  
  Seen_MacroF1 = c(
    0.9852,
    0.9660,
    0.9269,
    0.8385,
    0.7118
  ),
  
  Seen_QWK = c(
    0.9948,
    0.9892,
    0.9752,
    0.9434,
    0.8888
  ),
  
  # -------------------------
  # Unseen districts
  # -------------------------
  Unseen_Accuracy = c(
    0.9892,
    0.9776,
    0.9521,
    0.8901,
    0.7965
  ),
  
  Unseen_MacroF1 = c(
    0.9841,
    0.9649,
    0.9237,
    0.8381,
    0.7172
  ),
  
  Unseen_QWK = c(
    0.9941,
    0.9877,
    0.9739,
    0.9386,
    0.8802
  )
)

# ------------------------------------------------------------
# 2. Convert to percentage
# ------------------------------------------------------------

results <- results %>%
  mutate(
    across(
      c(
        Seen_Accuracy,
        Seen_MacroF1,
        Seen_QWK,
        Unseen_Accuracy,
        Unseen_MacroF1,
        Unseen_QWK
      ),
      ~ .x * 100
    )
  )

# ------------------------------------------------------------
# 3. Seen-district data
# ------------------------------------------------------------

seen <- results %>%
  select(
    Horizon,
    Accuracy = Seen_Accuracy,
    Macro_F1 = Seen_MacroF1,
    QWK = Seen_QWK
  ) %>%
  pivot_longer(
    cols = c(
      Accuracy,
      Macro_F1,
      QWK
    ),
    names_to = "Metric",
    values_to = "Performance"
  ) %>%
  mutate(
    Districts = "(a) Seen districts"
  )

# ------------------------------------------------------------
# 4. Unseen-district data
# ------------------------------------------------------------

unseen <- results %>%
  select(
    Horizon,
    Accuracy = Unseen_Accuracy,
    Macro_F1 = Unseen_MacroF1,
    QWK = Unseen_QWK
  ) %>%
  pivot_longer(
    cols = c(
      Accuracy,
      Macro_F1,
      QWK
    ),
    names_to = "Metric",
    values_to = "Performance"
  ) %>%
  mutate(
    Districts = "(b) Unseen districts"
  )

# ------------------------------------------------------------
# 5. Combine
# ------------------------------------------------------------

plot_data <- bind_rows(
  seen,
  unseen
)

plot_data$Metric <- factor(
  plot_data$Metric,
  levels = c(
    "Accuracy",
    "Macro_F1",
    "QWK"
  ),
  labels = c(
    "Accuracy",
    "Macro-F1",
    "QWK"
  )
)

plot_data$Districts <- factor(
  plot_data$Districts,
  levels = c(
    "(a) Seen districts",
    "(b) Unseen districts"
  )
)

# ------------------------------------------------------------
# 6. Correct 24-h RMSE values
# ------------------------------------------------------------

rmse_seen <- 14.42561
rmse_unseen <- 14.50059

rmse_note <- paste0(
  "24-h regression RMSE: Seen = ",
  sprintf("%.2f", rmse_seen),
  " AQI points  |  Unseen = ",
  sprintf("%.2f", rmse_unseen),
  " AQI points"
)

# ------------------------------------------------------------
# 7. Create plot
# ------------------------------------------------------------

p <- ggplot(
  plot_data,
  aes(
    x = Horizon,
    y = Performance,
    group = Metric,
    color = Metric,
    linetype = Metric,
    shape = Metric
  )
) +
  
  geom_line(
    linewidth = 0.9
  ) +
  
  geom_point(
    size = 3,
    stroke = 0.8
  ) +
  
  facet_wrap(
    ~Districts,
    ncol = 1
  ) +
  
  scale_y_continuous(
    limits = c(65, 100),
    breaks = seq(
      65,
      100,
      5
    ),
    expand = expansion(
      mult = c(0.01, 0.02)
    )
  ) +
  
  scale_color_manual(
    values = c(
      "Accuracy" = "#1F77B4",
      "Macro-F1" = "#FF7F0E",
      "QWK" = "#2CA02C"
    )
  ) +
  
  scale_linetype_manual(
    values = c(
      "Accuracy" = "solid",
      "Macro-F1" = "dashed",
      "QWK" = "dotted"
    )
  ) +
  
  scale_shape_manual(
    values = c(
      "Accuracy" = 16,
      "Macro-F1" = 15,
      "QWK" = 17
    )
  ) +
  
  labs(
    title = "Performance Across Forecast Horizons",
    subtitle = rmse_note,
    x = "Forecast Horizon",
    y = "Performance (%)",
    color = NULL,
    linetype = NULL,
    shape = NULL
  ) +
  
  theme_minimal(
    base_size = 13
  ) +
  
  theme(
    
    plot.title = element_text(
      face = "bold",
      size = 18,
      hjust = 0.5
    ),
    
    plot.subtitle = element_text(
      size = 10.5,
      hjust = 0.5,
      color = "grey35",
      margin = margin(
        b = 8
      )
    ),
    
    strip.text = element_text(
      face = "bold",
      size = 14,
      hjust = 0
    ),
    
    strip.background = element_blank(),
    
    axis.title.x = element_text(
      face = "bold",
      size = 13
    ),
    
    axis.title.y = element_text(
      face = "bold",
      size = 13
    ),
    
    axis.text = element_text(
      size = 11,
      color = "black"
    ),
    
    legend.position = "bottom",
    
    legend.direction = "horizontal",
    
    legend.text = element_text(
      size = 11
    ),
    
    legend.key.width = unit(
      1.3,
      "cm"
    ),
    
    panel.grid.major = element_line(
      color = "grey80",
      linewidth = 0.4
    ),
    
    panel.grid.minor = element_blank(),
    
    panel.spacing = unit(
      0.55,
      "cm"
    ),
    
    plot.margin = margin(
      12,
      15,
      10,
      15
    )
  )

# ------------------------------------------------------------
# 8. Display
# ------------------------------------------------------------

print(p)

# ------------------------------------------------------------
# 9. Output directory
# ------------------------------------------------------------

output_dir <- paste0(
  "F:/Semester 9/Data Science/Final Research/",
  "Main Project Work/08_ML_Pipeline_Outputs/figures"
)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}

# ------------------------------------------------------------
# 10. Save PNG
# ------------------------------------------------------------

png_path <- file.path(
  output_dir,
  "performance_across_forecast_horizons.png"
)

ggsave(
  filename = png_path,
  plot = p,
  width = 8,
  height = 8.5,
  dpi = 600,
  bg = "white"
)

# ------------------------------------------------------------
# 11. Save PDF
# ------------------------------------------------------------

pdf_path <- file.path(
  output_dir,
  "performance_across_forecast_horizons.pdf"
)

ggsave(
  filename = pdf_path,
  plot = p,
  width = 8,
  height = 8.5,
  device = cairo_pdf
)

# ------------------------------------------------------------
# 12. Confirm output
# ------------------------------------------------------------

cat(
  "\n============================================\n",
  "FIGURE SAVED SUCCESSFULLY\n",
  "============================================\n\n",
  "PNG:\n",
  png_path,
  "\n\nPDF:\n",
  pdf_path,
  "\n"
)







































##############################################################################
# UNIQUE BUBBLE-BEESWARM SHAP PLOT
# Horizon order: 1h -> 3h -> 6h -> 12h
# Uses existing SHAP cache only - NO MODEL TRAINING
##############################################################################

# ---------------------------------------------------------------------------
# 1. Packages
# ---------------------------------------------------------------------------

packages <- c(
  "data.table",
  "ggplot2",
  "ggbeeswarm",
  "scales"
)

missing_pkgs <- packages[
  !packages %in% rownames(installed.packages())
]

if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs)
}

library(data.table)
library(ggplot2)
library(ggbeeswarm)
library(scales)


# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------

output_dir <- "F:/Semester 9/Data Science/Final Research/Main Project Work/08_ML_Pipeline_Outputs"

figure_dir <- file.path(
  output_dir,
  "figures"
)

shap_file <- file.path(
  output_dir,
  "shap_raw_values_very_unhealthy.csv"
)

if (!dir.exists(figure_dir)) {
  dir.create(
    figure_dir,
    recursive = TRUE
  )
}


# ---------------------------------------------------------------------------
# 3. Load existing SHAP data
# ---------------------------------------------------------------------------

if (!file.exists(shap_file)) {
  stop(
    "SHAP cache not found:\n",
    shap_file
  )
}

cat("Loading SHAP cache...\n")

shap_data <- fread(shap_file)

cat(
  "Loaded ",
  format(nrow(shap_data), big.mark = ","),
  " rows.\n",
  sep = ""
)


# ---------------------------------------------------------------------------
# 4. Correct horizon order
# ---------------------------------------------------------------------------

shap_data[, horizon := factor(
  horizon,
  levels = c(
    "1h",
    "3h",
    "6h",
    "12h"
  )
)]

# Remove anything outside the classification horizons
shap_data <- shap_data[
  !is.na(horizon)
]


# ---------------------------------------------------------------------------
# 5. Pollutants
# ---------------------------------------------------------------------------

pollutants <- c(
  "pm2_5",
  "ozone",
  "carbon_monoxide",
  "sulphur_dioxide",
  "pm10",
  "nitrogen_dioxide"
)

shap_data <- shap_data[
  feature %in% pollutants
]


# ---------------------------------------------------------------------------
# 6. Human-readable pollutant names
# ---------------------------------------------------------------------------

pollutant_labels <- c(
  pm2_5 = expression(PM[2.5]),
  ozone = expression(O[3]),
  carbon_monoxide = "CO",
  sulphur_dioxide = expression(SO[2]),
  pm10 = expression(PM[10]),
  nitrogen_dioxide = expression(NO[2])
)


# ---------------------------------------------------------------------------
# 7. Feature ordering
#    Based on overall mean absolute SHAP
# ---------------------------------------------------------------------------

feature_order <- shap_data[
  ,
  .(
    importance = mean(
      abs(shap_value),
      na.rm = TRUE
    )
  ),
  by = feature
][
  order(-importance)
]$feature

shap_data[, feature := factor(
  feature,
  levels = rev(feature_order)
)]


# ---------------------------------------------------------------------------
# 8. Bubble size
# ---------------------------------------------------------------------------

# Absolute SHAP represents the magnitude of the contribution.
shap_data[, abs_shap := abs(shap_value)]

# Prevent extremely large points from dominating the figure
size_cap <- quantile(
  shap_data$abs_shap,
  0.98,
  na.rm = TRUE
)

shap_data[
  abs_shap > size_cap,
  abs_shap := size_cap
]


# ---------------------------------------------------------------------------
# 9. Publication-style theme
# ---------------------------------------------------------------------------

theme_unique_shap <- theme_minimal(
  base_size = 12
) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18,
      hjust = 0
    ),
    
    plot.subtitle = element_text(
      size = 11,
      color = "grey35",
      hjust = 0
    ),
    
    axis.title.x = element_text(
      face = "bold",
      size = 12
    ),
    
    axis.text.x = element_text(
      size = 10
    ),
    
    axis.text.y = element_text(
      face = "bold",
      size = 11,
      color = "grey15"
    ),
    
    panel.grid.major.y = element_line(
      color = "grey88",
      linewidth = 0.35
    ),
    
    panel.grid.minor = element_blank(),
    
    panel.grid.major.x = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    
    strip.background = element_rect(
      fill = "grey15",
      color = NA
    ),
    
    strip.text = element_text(
      face = "bold",
      color = "white",
      size = 12
    ),
    
    legend.position = "bottom",
    
    legend.title = element_text(
      face = "bold"
    ),
    
    plot.margin = margin(
      10, 15, 10, 10
    )
  )


# ---------------------------------------------------------------------------
# 10. Create bubble-beeswarm plot
# ---------------------------------------------------------------------------

p <- ggplot(
  shap_data,
  aes(
    x = shap_value,
    y = feature
  )
) +
  
  # Zero reference line
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.6,
    color = "grey45"
  ) +
  
  # Bubble beeswarm
  geom_quasirandom(
    aes(
      color = shap_value,
      size = abs_shap
    ),
    width = 0,
    groupOnX = FALSE,
    varwidth = FALSE,
    alpha = 0.55
  ) +
  
  # Diverging SHAP color
  scale_color_gradient2(
    low = "#2166AC",
    mid = "#F7F7F7",
    high = "#D73027",
    midpoint = 0,
    name = "SHAP value"
  ) +
  
  # Bubble size legend
  scale_size_continuous(
    range = c(
      0.8,
      7
    ),
    name = "|SHAP|"
  ) +
  
  # Separate horizons
  facet_wrap(
    ~horizon,
    nrow = 1,
    drop = TRUE
  ) +
  
  labs(
    title = "Pollutant Impact on 'Very Unhealthy' AQI",
    subtitle = paste(
      "Bubble position shows SHAP contribution;",
      "color shows direction and bubble size shows magnitude"
    ),
    x = "SHAP value (Very Unhealthy class)",
    y = NULL
  ) +
  
  theme_unique_shap


# ---------------------------------------------------------------------------
# 11. Save high-resolution PNG
# ---------------------------------------------------------------------------

png_file <- file.path(
  figure_dir,
  "shap_pollutants_bubble_beeswarm.png"
)

ggsave(
  filename = png_file,
  plot = p,
  width = 14,
  height = 6.5,
  units = "in",
  dpi = 400,
  bg = "white"
)


# ---------------------------------------------------------------------------
# 12. Save PDF for Overleaf
# ---------------------------------------------------------------------------

pdf_file <- file.path(
  figure_dir,
  "shap_pollutants_bubble_beeswarm.pdf"
)

ggsave(
  filename = pdf_file,
  plot = p,
  width = 14,
  height = 6.5,
  units = "in",
  bg = "white"
)


# ---------------------------------------------------------------------------
# 13. Display
# ---------------------------------------------------------------------------

print(p)


# ---------------------------------------------------------------------------
# 14. Confirmation
# ---------------------------------------------------------------------------

cat("\n")
cat("==============================================================\n")
cat("BUBBLE-BEESWARM SHAP PLOT COMPLETE\n")
cat("==============================================================\n")
cat("Horizon order: 1h -> 3h -> 6h -> 12h\n")
cat("24h: excluded (regression horizon)\n")
cat("\nPNG:\n")
cat(png_file, "\n")
cat("\nPDF:\n")
cat(pdf_file, "\n")
cat("==============================================================\n")



























# =============================================================================
# FINAL SHAP POLLUTANT BEESWARM
# VERY UNHEALTHY CLASS
# HORIZON ORDER: 1h → 3h → 6h → 12h
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PACKAGES
# -----------------------------------------------------------------------------

required_packages <- c(
  "data.table",
  "ggplot2"
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(data.table)
library(ggplot2)


# -----------------------------------------------------------------------------
# 2. FILE PATHS
# -----------------------------------------------------------------------------

shap_file <- paste0(
  "F:/Semester 9/Data Science/Final Research/",
  "Main Project Work/08_ML_Pipeline_Outputs/",
  "shap_raw_values_very_unhealthy.csv"
)

output_dir <- paste0(
  "F:/Semester 9/Data Science/Final Research/",
  "Main Project Work/08_ML_Pipeline_Outputs/figures"
)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


# -----------------------------------------------------------------------------
# 3. LOAD EXISTING SHAP VALUES
# -----------------------------------------------------------------------------

cat("\nLoading SHAP file...\n")

shap <- fread(shap_file)

cat("Rows:", nrow(shap), "\n")
cat("Columns:", paste(names(shap), collapse = ", "), "\n")


# -----------------------------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# -----------------------------------------------------------------------------

required_columns <- c(
  "feature",
  "shap_value",
  "horizon"
)

missing_columns <- setdiff(
  required_columns,
  names(shap)
)

if (length(missing_columns) > 0) {
  
  stop(
    paste(
      "Missing columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
  
}


# -----------------------------------------------------------------------------
# 5. SIX POLLUTANTS ONLY
# -----------------------------------------------------------------------------

pollutants <- c(
  "pm2_5",
  "ozone",
  "carbon_monoxide",
  "sulphur_dioxide",
  "pm10",
  "nitrogen_dioxide"
)

shap_pollutants <- shap[
  feature %in% pollutants
]


# -----------------------------------------------------------------------------
# 6. CHECK DATA
# -----------------------------------------------------------------------------

cat("\nPollutants found:\n")
print(unique(shap_pollutants$feature))

cat("\nHorizons found:\n")
print(unique(shap_pollutants$horizon))


# -----------------------------------------------------------------------------
# 7. FORCE EXACT HORIZON ORDER
# -----------------------------------------------------------------------------

shap_pollutants[
  ,
  horizon := factor(
    horizon,
    levels = c(
      "1h",
      "3h",
      "6h",
      "12h"
    )
  )
]


# -----------------------------------------------------------------------------
# 8. FORCE EXACT POLLUTANT ORDER
#
# TOP → BOTTOM:
#
# PM2.5
# O3
# CO
# SO2
# PM10
# NO2
# -----------------------------------------------------------------------------

shap_pollutants[
  ,
  feature := factor(
    feature,
    levels = rev(
      pollutants
    )
  )
]


# -----------------------------------------------------------------------------
# 9. CREATE DISPLAY LABELS
# -----------------------------------------------------------------------------

shap_pollutants[
  ,
  feature_label := factor(
    feature,
    levels = rev(pollutants),
    labels = rev(
      c(
        "PM2.5",
        "O3",
        "CO",
        "SO2",
        "PM10",
        "NO2"
      )
    )
  )
]


# -----------------------------------------------------------------------------
# 10. SYMMETRIC X-AXIS
#
# Keeps zero exactly in the middle.
# -----------------------------------------------------------------------------

max_shap <- max(
  abs(shap_pollutants$shap_value),
  na.rm = TRUE
)

axis_limit <- ceiling(
  max_shap * 2
) / 2

cat(
  "\nSHAP axis:",
  -axis_limit,
  "to",
  axis_limit,
  "\n"
)


# -----------------------------------------------------------------------------
# 11. PAPER THEME
# -----------------------------------------------------------------------------

theme_paper <- function(base_size = 12) {
  
  theme_minimal(
    base_size = base_size
  ) +
    
    theme(
      
      plot.title = element_text(
        face = "bold",
        size = base_size + 2,
        hjust = 0
      ),
      
      plot.subtitle = element_text(
        size = base_size - 1,
        color = "grey35"
      ),
      
      axis.title.x = element_text(
        face = "bold",
        size = base_size
      ),
      
      axis.text.y = element_text(
        face = "bold",
        size = base_size - 1
      ),
      
      axis.text.x = element_text(
        size = base_size - 1
      ),
      
      panel.grid.minor = element_blank(),
      
      panel.grid.major = element_line(
        color = "grey90",
        linewidth = 0.3
      ),
      
      strip.background = element_rect(
        fill = "grey15",
        color = NA
      ),
      
      strip.text = element_text(
        face = "bold",
        color = "white",
        size = base_size
      ),
      
      legend.position = "bottom",
      
      legend.title = element_text(
        face = "bold"
      ),
      
      panel.spacing = unit(
        0.4,
        "lines"
      )
    )
}


# -----------------------------------------------------------------------------
# 12. DRAW BEESWARM
# -----------------------------------------------------------------------------

p_pollutants <- ggplot(
  shap_pollutants,
  aes(
    x = shap_value,
    y = feature_label,
    color = shap_value
  )
) +
  
  # Zero reference
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50",
    linewidth = 0.6
  ) +
  
  # Beeswarm-like distribution
  geom_jitter(
    width = 0,
    height = 0.20,
    size = 1.0,
    alpha = 0.70
  ) +
  
  # BLUE → WHITE → RED
  scale_color_gradient2(
    low = "#2C3E50",
    mid = "white",
    high = "#E74C3C",
    midpoint = 0,
    limits = c(
      -axis_limit,
      axis_limit
    ),
    name = "SHAP Value"
  ) +
  
  # IMPORTANT:
  # This is what guarantees:
  # 1h → 3h → 6h → 12h
  facet_wrap(
    ~horizon,
    nrow = 1,
    drop = FALSE
  ) +
  
  scale_x_continuous(
    limits = c(
      -axis_limit,
      axis_limit
    ),
    expand = expansion(
      mult = 0.03
    )
  ) +
  
  labs(
    
    title = "Pollutant Impact on 'Very Unhealthy' AQI",
    
    subtitle =
      "Positive (red) = increases risk; Negative (blue) = decreases risk",
    
    x = "SHAP Contribution (Very Unhealthy class)",
    
    y = NULL
    
  ) +
  
  theme_paper(
    base_size = 12
  ) +
  
  theme(
    
    legend.position = "bottom",
    
    legend.key.width = unit(
      1.8,
      "cm"
    ),
    
    axis.text.y = element_text(
      face = "bold",
      size = 10
    ),
    
    strip.text = element_text(
      face = "bold",
      color = "white",
      size = 11
    ),
    
    strip.background = element_rect(
      fill = "grey15",
      color = NA
    ),
    
    panel.grid.major.y = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    
    panel.grid.major.x = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    
    panel.grid.minor = element_blank()
  )


# -----------------------------------------------------------------------------
# 13. PRINT PLOT
# -----------------------------------------------------------------------------

print(p_pollutants)


# -----------------------------------------------------------------------------
# 14. SAVE PNG
# -----------------------------------------------------------------------------

png_file <- file.path(
  output_dir,
  "shap_pollutants_beeswarm_final.png"
)

ggsave(
  filename = png_file,
  plot = p_pollutants,
  width = 14,
  height = 5,
  dpi = 320,
  bg = "white"
)


# -----------------------------------------------------------------------------
# 15. SAVE PDF
# -----------------------------------------------------------------------------

pdf_file <- file.path(
  output_dir,
  "shap_pollutants_beeswarm_final.pdf"
)

ggsave(
  filename = pdf_file,
  plot = p_pollutants,
  width = 14,
  height = 5,
  bg = "white"
)


# -----------------------------------------------------------------------------
# 16. FINISHED
# -----------------------------------------------------------------------------

cat("\n==============================================================\n")
cat("FINAL SHAP BEESWARM COMPLETE\n")
cat("==============================================================\n")

cat("\nHorizon order:\n")
cat("1h → 3h → 6h → 12h\n")

cat("\nPollutant order:\n")
cat("PM2.5 → O3 → CO → SO2 → PM10 → NO2\n")

cat("\nCurrent AQI:\n")
cat("NOT INCLUDED\n")

cat("\nPNG saved to:\n")
cat(png_file, "\n")

cat("\nPDF saved to:\n")
cat(pdf_file, "\n")






































# ----------------------------------------------------------------------------
# 6. FINAL SQUARE-CELL CONFUSION MATRIX
# ----------------------------------------------------------------------------

p <- ggplot(
  cm,
  aes(
    x = Actual,
    y = Predicted,
    fill = pct
  )
) +
  
  geom_tile(
    color = "white",
    linewidth = 1.2
  ) +
  
  geom_text(
    aes(label = label),
    size = 4.2,
    color = "black",
    lineheight = 0.9
  ) +
  
  scale_fill_gradient(
    low = "#F4F8FC",
    high = "#245A8D",
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100),
    name = "Column %"
  ) +
  
  labs(
    title = "Confusion Matrix — XGBoost, 1h Horizon",
    subtitle = "Seen-district test set, column-normalized",
    x = "Actual class",
    y = "Predicted class"
  ) +
  
  # FORCE PERFECT SQUARE CELLS
  coord_fixed(ratio = 1) +
  
  theme_minimal(base_size = 13) +
  
  theme(
    plot.title = element_text(
      size = 19,
      face = "bold",
      hjust = 0.5
    ),
    
    plot.subtitle = element_text(
      size = 12,
      hjust = 0.5,
      color = "grey35"
    ),
    
    axis.title = element_text(
      size = 13,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      angle = 30,
      hjust = 1,
      size = 10.5
    ),
    
    axis.text.y = element_text(
      size = 10.5
    ),
    
    panel.grid = element_blank(),
    
    legend.title = element_text(
      face = "bold",
      size = 11
    ),
    
    legend.text = element_text(
      size = 10
    ),
    
    plot.margin = margin(
      10, 15, 10, 10
    )
  )

print(p)

# ----------------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------------

ggsave(
  "confusion_matrix_1h_final.png",
  p,
  width = 9,
  height = 9,
  dpi = 400,
  bg = "white"
)