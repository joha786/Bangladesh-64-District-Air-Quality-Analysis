# ==============================================================================
# 06_Modeling_Dataset_EDA.R — REVISED WITH KEY DIAGNOSTIC PLOTS
# ==============================================================================
#
# PROJECT:
# Multi-Horizon Early Warning of Future AQI Categories
# Across 64 Districts of Bangladesh
#
# PURPOSE:
#   Exploratory analysis of the FINAL modeling dataset created by Step 05.
#
# THIS STEP:
#   - DOES NOT modify the Step 05 dataset
#   - DOES NOT create new target categories
#   - DOES NOT categorize predictor variables
#   - DOES NOT train models
#   - DOES NOT perform train/test splitting
#   - DOES NOT perform normalization/standardization
#   - DOES NOT use ARM
#
# TARGETS:
#   +1h, +3h, +6h, +12h, +24h
#
# INPUT:
#   Bangladesh_64_District_Hourly_2023_2026_MULTI_HORIZON_AQI_TARGETS.csv
#
# OUTPUTS:
#   Tables: 09 CSV files (missingness, summary, target distributions, etc.)
#   Figures: 06 PNG files (target distribution, coverage heatmap, histograms,
#            boxplots, correlation heatmap, target transition matrix)
# ==============================================================================


# ---- 0. PACKAGES ------------------------------------------------------------
required_packages <- c("data.table", "ggplot2", "gridExtra", "corrplot")
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) install.packages(missing_packages)

library(data.table)
library(ggplot2)
library(gridExtra)
library(corrplot)


# ---- 1. PATHS ---------------------------------------------------------------
BASE_DIR <- "F:/Semester 9/Data Science/Final Research/Main Project Work"

INPUT_FILE <- file.path(
  BASE_DIR,
  "05_Future_AQI_Target_Construction",
  "outputs",
  "Bangladesh_64_District_Hourly_2023_2026_MULTI_HORIZON_AQI_TARGETS.csv"
)

OUTPUT_DIR <- file.path(BASE_DIR, "06_Modeling_Dataset_EDA")
TABLE_DIR  <- file.path(OUTPUT_DIR, "tables")
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
REPORT_DIR <- file.path(OUTPUT_DIR, "reports")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(REPORT_DIR, recursive = TRUE, showWarnings = FALSE)


# ---- 2. INPUT CHECK ---------------------------------------------------------
if (!file.exists(INPUT_FILE)) {
  stop("Step 05 output not found:\n", INPUT_FILE)
}


# ---- 3. LOAD DATA -----------------------------------------------------------
cat("\n============================================================\n")
cat("06 — MODELING DATASET EDA (REVISED)\n")
cat("============================================================\n")

cat("\nLoading Step 05 dataset...\n")
dt <- fread(INPUT_FILE, showProgress = TRUE)
cat("Rows: ", format(nrow(dt), big.mark = ","), "\n")
cat("Columns: ", ncol(dt), "\n")


# ---- 4. REQUIRED COLUMNS ----------------------------------------------------
predictor_columns <- c(
  "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
  "sulphur_dioxide", "ozone",
  "temperature_2m", "relative_humidity_2m", "dew_point_2m",
  "surface_pressure", "precipitation", "cloud_cover",
  "wind_speed_10m", "wind_direction_10m"
)

target_category_columns <- paste0("aqi_category_t_plus_", c(1,3,6,12,24), "h")
target_numeric_columns  <- paste0("aqi_t_plus_", c(1,3,6,12,24), "h")
target_datetime_columns <- paste0("target_datetime_", c(1,3,6,12,24), "h")

required_columns <- c(
  "datetime", "datetime_utc", "district", "district_pcode", "district_key",
  "us_aqi",
  predictor_columns,
  target_category_columns,
  target_numeric_columns,
  target_datetime_columns
)

missing_cols <- setdiff(required_columns, names(dt))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))
cat("Required-column validation: PASS\n")


# ---- 5. BASIC DATASET VALIDATION --------------------------------------------
cat("\n============================================================\n")
cat("5. BASIC DATASET VALIDATION\n")
cat("============================================================\n")

district_count <- uniqueN(dt$district_pcode)
if (district_count != 64) stop("Expected 64 districts, found ", district_count)
cat("Districts: ", district_count, "\n")

duplicate_count <- dt[, .N, by = .(district_pcode, datetime)][N > 1, .N]
if (duplicate_count > 0) stop("Duplicate district-time keys detected.")
cat("Duplicate keys: 0\n")


# ---- 6. DATETIME VALIDATION (Step 05 already validated) --------------------
cat("\n============================================================\n")
cat("6. DATETIME VALIDATION\n")
cat("============================================================\n")

if (anyNA(dt$datetime)) stop("Missing Bangladesh datetime values.")
if (anyNA(dt$datetime_utc)) stop("Missing UTC datetime values.")
setorder(dt, district_pcode, datetime)
order_errors <- dt[, sum(datetime[-.N] >= datetime[-1]), by = district_pcode][, sum(V1)]
if (order_errors > 0) stop("Chronological ordering errors detected.")
cat("Datetime integrity: PASS\n")


# ---- 7. PREDICTOR MISSINGNESS -----------------------------------------------
cat("\n============================================================\n")
cat("7. PREDICTOR MISSINGNESS\n")
cat("============================================================\n")

missing_summary <- data.table(
  variable = predictor_columns,
  missing_count = sapply(predictor_columns, function(x) sum(is.na(dt[[x]])))
)
missing_summary[, missing_percent := 100 * missing_count / nrow(dt)]
fwrite(missing_summary, file.path(TABLE_DIR, "01_predictor_missingness.csv"))
print(missing_summary)


# ---- 8. NUMERICAL PREDICTOR SUMMARY -----------------------------------------
cat("\n============================================================\n")
cat("8. NUMERICAL PREDICTOR SUMMARY\n")
cat("============================================================\n")

numeric_summary <- rbindlist(lapply(predictor_columns, function(x) {
  vals <- dt[[x]]
  data.table(
    variable = x,
    count = sum(!is.na(vals)),
    missing = sum(is.na(vals)),
    mean = mean(vals, na.rm = TRUE),
    median = median(vals, na.rm = TRUE),
    sd = sd(vals, na.rm = TRUE),
    min = min(vals, na.rm = TRUE),
    max = max(vals, na.rm = TRUE),
    q25 = quantile(vals, 0.25, na.rm = TRUE),
    q75 = quantile(vals, 0.75, na.rm = TRUE)
  )
}))
fwrite(numeric_summary, file.path(TABLE_DIR, "02_predictor_numeric_summary.csv"))
print(numeric_summary)


# ---- 9. TARGET CLASS DISTRIBUTION -------------------------------------------
cat("\n============================================================\n")
cat("9. TARGET CLASS DISTRIBUTION\n")
cat("============================================================\n")

allowed_categories <- c("Good", "Moderate", "Unhealthy for Sensitive Groups",
                        "Unhealthy", "Very Unhealthy", "Hazardous")
horizons <- c(1, 3, 6, 12, 24)

target_distribution <- rbindlist(lapply(horizons, function(h) {
  col <- paste0("aqi_category_t_plus_", h, "h")
  tmp <- dt[!is.na(get(col)), .(observations = .N), by = .(category = get(col))]
  tmp[, `:=`(horizon = paste0(h, "h"), percentage = 100 * observations / sum(observations))]
  tmp
}))
target_distribution[, category := factor(category, levels = allowed_categories)]
setorder(target_distribution, horizon, category)
fwrite(target_distribution, file.path(TABLE_DIR, "03_target_class_distribution.csv"))
print(target_distribution)


# ---- 10. TARGET CLASS IMBALANCE ---------------------------------------------
imbalance_summary <- rbindlist(lapply(horizons, function(h) {
  col <- paste0("aqi_category_t_plus_", h, "h")
  tmp <- dt[!is.na(get(col)), .(observations = .N), by = .(category = get(col))]
  tmp[, `:=`(horizon = paste0(h, "h"),
             prevalence = observations / sum(observations),
             inverse_frequency_weight = 1 / prevalence)]
  tmp
}))
imbalance_summary[, category := factor(category, levels = allowed_categories)]
setorder(imbalance_summary, horizon, category)
fwrite(imbalance_summary, file.path(TABLE_DIR, "04_target_class_imbalance.csv"))
print(imbalance_summary)


# ---- 11. TARGET DISTRIBUTION PLOT -------------------------------------------
p_target <- ggplot(target_distribution, aes(x = horizon, y = percentage, fill = category)) +
  geom_col(position = "stack") +
  labs(title = "Future AQI Class Distribution by Forecast Horizon",
       x = "Forecast Horizon", y = "Percentage", fill = "AQI Category") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIGURE_DIR, "01_future_aqi_class_distribution.png"), p_target,
       width = 10, height = 6, dpi = 300)


# ---- 12. DISTRICT × HORIZON TARGET COVERAGE ---------------------------------
district_coverage <- rbindlist(lapply(horizons, function(h) {
  exists_col <- paste0("target_exists_", h, "h")
  tmp <- dt[, .(source_rows = .N, valid_targets = sum(get(exists_col), na.rm = TRUE)),
            by = .(district_pcode, district)]
  tmp[, `:=`(horizon = paste0(h, "h"), coverage_percent = 100 * valid_targets / source_rows)]
  tmp
}))
fwrite(district_coverage, file.path(TABLE_DIR, "05_district_horizon_target_coverage.csv"))

coverage_summary <- district_coverage[, .(
  mean = mean(coverage_percent), median = median(coverage_percent),
  min = min(coverage_percent), max = max(coverage_percent),
  sd = sd(coverage_percent)
), by = horizon]
fwrite(coverage_summary, file.path(TABLE_DIR, "06_district_horizon_coverage_summary.csv"))
print(coverage_summary)


# ---- 13. COVERAGE HEATMAP ---------------------------------------------------
p_coverage <- ggplot(district_coverage, aes(x = horizon, y = district, fill = coverage_percent)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Target Coverage Across Districts and Horizons",
       x = "Horizon", y = "District", fill = "Coverage (%)") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 5), plot.title = element_text(face = "bold"))
ggsave(file.path(FIGURE_DIR, "02_district_horizon_target_coverage.png"), p_coverage,
       width = 10, height = 14, dpi = 300)


# ---- 14. TARGET TRANSITIONS (descriptive) -----------------------------------
transition_pairs <- list(c(1,3), c(3,6), c(6,12), c(12,24))
transition_summary <- rbindlist(lapply(transition_pairs, function(pair) {
  h1 <- pair[1]; h2 <- pair[2]
  col1 <- paste0("aqi_category_t_plus_", h1, "h")
  col2 <- paste0("aqi_category_t_plus_", h2, "h")
  tmp <- dt[!is.na(get(col1)) & !is.na(get(col2)),
            .(observations = .N), by = .(from = get(col1), to = get(col2))]
  tmp[, `:=`(from_horizon = paste0(h1, "h"), to_horizon = paste0(h2, "h"))]
  tmp
}))
fwrite(transition_summary, file.path(TABLE_DIR, "07_target_horizon_transitions.csv"))


# ---- 15. OVERALL TIME RANGE -------------------------------------------------
time_range <- data.table(
  first_datetime = min(dt$datetime),
  last_datetime = max(dt$datetime),
  first_utc = min(dt$datetime_utc),
  last_utc = max(dt$datetime_utc),
  total_rows = nrow(dt),
  districts = uniqueN(dt$district_pcode)
)
fwrite(time_range, file.path(TABLE_DIR, "08_dataset_time_range.csv"))


# ---- 16. PREDICTOR CORRELATIONS (table) ------------------------------------
cor_mat <- cor(dt[, ..predictor_columns], use = "pairwise.complete.obs")
cor_dt <- as.data.table(cor_mat, keep.rownames = "variable")
fwrite(cor_dt, file.path(TABLE_DIR, "09_predictor_correlation_matrix.csv"))


# ---- 17. DISTRIBUTION PLOTS (HISTOGRAMS) -----------------------------------
# Reshape to long format for faceted plotting
predictor_long <- melt(dt, id.vars = NULL, measure.vars = predictor_columns,
                       variable.name = "variable", value.name = "value")
p_hist <- ggplot(predictor_long, aes(x = value)) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white", alpha = 0.7) +
  facet_wrap(~ variable, scales = "free", ncol = 4) +
  labs(title = "Distributions of Numeric Predictors") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIGURE_DIR, "03_predictor_histograms.png"), p_hist,
       width = 14, height = 10, dpi = 300)


# ---- 18. BOXPLOTS (OUTLIER DETECTION) --------------------------------------
p_box <- ggplot(predictor_long, aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.size = 0.5) +
  coord_flip() +
  labs(title = "Boxplots of Predictors (red points = outliers)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIGURE_DIR, "04_predictor_boxplots.png"), p_box,
       width = 12, height = 8, dpi = 300)


# ---- 19. CORRELATION HEATMAP (FIGURE) --------------------------------------
png(file.path(FIGURE_DIR, "05_predictor_correlation_heatmap.png"),
    width = 1200, height = 1200, res = 150)
corrplot(cor_mat, method = "color", type = "upper",
         tl.cex = 0.7, number.cex = 0.5, addCoef.col = "black",
         tl.col = "black", title = "Predictor Correlation Matrix",
         mar = c(0,0,2,0))
dev.off()


# ---- 20. TARGET TRANSITION HEATMAP (for each pair) -------------------------
# Only plot the most interesting pair (12h → 24h) as example
pair_12_24 <- transition_summary[from_horizon == "12h" & to_horizon == "24h"]
pair_12_24[, from := factor(from, levels = allowed_categories)]
pair_12_24[, to := factor(to, levels = allowed_categories)]
p_trans <- ggplot(pair_12_24, aes(x = from, y = to, fill = observations)) +
  geom_tile() +
  geom_text(aes(label = observations), color = "white", size = 3) +
  scale_fill_viridis_c() +
  labs(title = "Transition Matrix: 12h → 24h (counts)",
       x = "Class at 12h", y = "Class at 24h") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file.path(FIGURE_DIR, "06_transition_12h_to_24h.png"), p_trans,
       width = 8, height = 6, dpi = 300)


# ---- 21. FINAL REPORT ------------------------------------------------------
final_summary <- data.table(
  rows = nrow(dt),
  columns = ncol(dt),
  districts = uniqueN(dt$district_pcode),
  predictor_count = length(predictor_columns),
  horizon_count = length(horizons),
  target_class_count = length(allowed_categories),
  datetime_format = "Bangladesh local time (+06:00)",
  target_type = "Six-class future overall AQI",
  predictor_type = "Numerical meteorological and pollutant variables",
  arm_used = FALSE,
  sub_aqi_categories_used = FALSE
)
fwrite(final_summary, file.path(REPORT_DIR, "final_eda_summary.csv"))


# ---- 22. FINAL CONSOLE MESSAGE ---------------------------------------------
cat("\n============================================================\n")
cat("06 — MODELING DATASET EDA — COMPLETE\n")
cat("============================================================\n")
cat("\nRows:        ", format(nrow(dt), big.mark = ","))
cat("\nDistricts:   ", uniqueN(dt$district_pcode))
cat("\nPredictors:  ", length(predictor_columns))
cat("\nHorizons:    ", paste(horizons, collapse = ", "))
cat("\nClasses:     ", length(allowed_categories))
cat("\n\nFigures saved to:  ", FIGURE_DIR)
cat("\nTables saved to:    ", TABLE_DIR)
cat("\nReports saved to:   ", REPORT_DIR)
cat("\n\nARM: NOT USED\n")
cat("Sub-AQI categories: NOT USED\n")
cat("Predictor categorization: NOT USED\n")
cat("Model training: NOT performed\n")
cat("Train/test splitting: NOT performed\n")
cat("============================================================\n")