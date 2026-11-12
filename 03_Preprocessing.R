# ==============================================================================
# 03_Preprocessing.R — FINAL WITH DIAGNOSTIC PLOTS & DROPPED UNNECESSARY COLUMNS
# ==============================================================================
#
# PURPOSE:
#   Validate, clean, and convert the raw merged dataset to a consistent
#   preprocessed format with correct timezone handling. Drop redundant columns
#   (sub‑AQI, temporary) before saving. Generate diagnostic plots
#   (boxplots, histograms, missingness) to check for outliers and data quality.
#
# INPUT:
#   F:/Semester 9/Data Science/Final Research/Main Project Work/
#   02_Data_Download/Bangladesh_64_District_Hourly_2023_2026_RAW.csv
#
# OUTPUT:
#   1. Preprocessed CSV: 03_Preprocessing/outputs/
#      Bangladesh_64_District_Hourly_2023_2026_PREPROCESSED.csv
#   2. Quality reports: 03_Preprocessing/quality_reports/
#   3. Diagnostic figures: 03_Preprocessing/figures/
# ==============================================================================

# ---- 0. PACKAGES ------------------------------------------------------------
required_packages <- c("data.table", "ggplot2", "gridExtra", "scales", "corrplot")
missing_pkgs <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_pkgs) > 0) install.packages(missing_pkgs)

library(data.table)
library(ggplot2)
library(gridExtra)
library(scales)
library(corrplot)

# ---- 1. PATHS ---------------------------------------------------------------
BASE_DIR <- "F:/Semester 9/Data Science/Final Research/Main Project Work"

INPUT_FILE <- file.path(
  BASE_DIR,
  "02_Data_Download",
  "Bangladesh_64_District_Hourly_2023_2026_RAW.csv"
)

PREPROCESS_DIR <- file.path(BASE_DIR, "03_Preprocessing")
OUTPUT_DIR     <- file.path(PREPROCESS_DIR, "outputs")
REPORT_DIR     <- file.path(PREPROCESS_DIR, "quality_reports")
FIG_DIR        <- file.path(PREPROCESS_DIR, "figures")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(REPORT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR,    recursive = TRUE, showWarnings = FALSE)

# ---- 2. INPUT CHECK ---------------------------------------------------------
if (!file.exists(INPUT_FILE)) {
  stop("RAW DATASET NOT FOUND:\n", INPUT_FILE)
}

cat("\n============================================================\n")
cat("03 PREPROCESSING — WITH DIAGNOSTIC PLOTS\n")
cat("============================================================\n")

# ---- 3. LOAD DATA -----------------------------------------------------------
cat("\nLoading raw dataset...\n")
dt <- fread(INPUT_FILE, colClasses = list(character = "datetime"), showProgress = TRUE)
cat("Rows: ", format(nrow(dt), big.mark = ","), "\n")
cat("Columns: ", ncol(dt), "\n")

# ---- 4. REQUIRED COLUMNS ----------------------------------------------------
# We'll keep a minimal set for the final dataset; validation still checks all.
required_columns <- c(
  "datetime", "district", "district_pcode", "district_key",
  "latitude", "longitude",
  "temperature_2m", "relative_humidity_2m", "dew_point_2m",
  "surface_pressure", "precipitation", "cloud_cover",
  "wind_speed_10m", "wind_direction_10m",
  "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
  "sulphur_dioxide", "ozone",
  "us_aqi",
  "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide",
  "us_aqi_ozone", "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide"
)

missing_cols <- setdiff(required_columns, names(dt))
if (length(missing_cols) > 0) {
  stop("Missing columns: ", paste(missing_cols, collapse = ", "))
}
cat("Required columns: PASS\n")

# ---- 5. RAW DATETIME FORMAT ------------------------------------------------
valid_format <- grepl("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$", dt$datetime)
invalid_count <- sum(!valid_format)
if (invalid_count > 0) {
  fwrite(dt[!valid_format, .(datetime)], file.path(REPORT_DIR, "invalid_datetime_format.csv"))
  stop("Invalid ISO-8601 UTC timestamps found.")
}
cat("ISO-8601 format: PASS\n")

# ---- 6. PARSE UTC TIMESTAMP ------------------------------------------------
dt[, datetime_utc := as.POSIXct(datetime, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")]
if (anyNA(dt$datetime_utc)) {
  fwrite(dt[is.na(datetime_utc), .(datetime)], file.path(REPORT_DIR, "invalid_datetime_records.csv"))
  stop("UTC timestamp conversion failed.")
}
cat("UTC timestamp conversion: PASS\n")

# ---- 7. CONVERT UTC → BANGLADESH LOCAL TIME --------------------------------
dt[, datetime := as.POSIXct(
  format(datetime_utc, tz = "Asia/Dhaka", format = "%Y-%m-%d %H:%M:%S"),
  format = "%Y-%m-%d %H:%M:%S",
  tz = "Asia/Dhaka"
)]
if (anyNA(dt$datetime)) stop("Bangladesh datetime conversion failed.")

# ---- 8. TIMEZONE VALIDATION ------------------------------------------------
utc_clock <- format(dt$datetime_utc, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
bd_clock  <- format(dt$datetime, tz = "Asia/Dhaka", format = "%Y-%m-%d %H:%M:%S")
expected_bd_clock <- format(
  as.POSIXct(utc_clock, format = "%Y-%m-%d %H:%M:%S", tz = "UTC") + 6*60*60,
  tz = "UTC", format = "%Y-%m-%d %H:%M:%S"
)
if (any(bd_clock != expected_bd_clock)) stop("Timezone conversion failed.")
cat("Timezone conversion: PASS — Bangladesh time = UTC + 6 hours.\n")

# ---- 9. EXPECTED DATE RANGE ------------------------------------------------
actual_start <- min(dt$datetime, na.rm = TRUE)
actual_end   <- max(dt$datetime, na.rm = TRUE)
expected_start <- as.POSIXct("2023-01-01 00:00:00", tz = "Asia/Dhaka")
expected_end   <- as.POSIXct("2026-08-20 23:00:00", tz = "Asia/Dhaka")
if (actual_start != expected_start || actual_end != expected_end) {
  stop("Date range does not match expected.")
}
cat("Date range: PASS\n")

# ---- 10. DISTRICT VALIDATION -----------------------------------------------
district_count <- uniqueN(dt$district_key)
if (district_count != 64) stop("Expected 64 districts, found ", district_count)
cat("District validation: PASS\n")

# ---- 11. DUPLICATE CHECK ---------------------------------------------------
dup <- dt[, .N, by = .(district_key, datetime)][N > 1]
dup_count <- if (nrow(dup) == 0) 0 else sum(dup$N - 1)
if (dup_count > 0) {
  fwrite(dup, file.path(REPORT_DIR, "duplicate_district_datetime.csv"))
  stop("Duplicate district-datetime records detected.")
}
cat("Duplicate check: PASS\n")

# ---- 12. SORT --------------------------------------------------------------
setorder(dt, district_key, datetime)

# ---- 13. HOURLY CONTINUITY ------------------------------------------------
continuity <- dt[, {
  hrs <- as.numeric(diff(datetime), units = "hours")
  .(rows = .N,
    min_interval = if (length(hrs) > 0) min(hrs) else NA_real_,
    max_interval = if (length(hrs) > 0) max(hrs) else NA_real_,
    non_hourly = if (length(hrs) > 0) sum(hrs != 1) else 0)
}, by = .(district_key, district)]
continuity[, status := fifelse(non_hourly == 0, "PASS", "CHECK")]
fwrite(continuity, file.path(REPORT_DIR, "hourly_continuity_report.csv"))
if (any(continuity$non_hourly > 0)) stop("Hourly continuity check failed.")
cat("Hourly continuity: PASS\n")

# ---- 14. EXPECTED ROW COUNT ------------------------------------------------
expected_days <- 365 + 366 + 365 + 232   # 2023–2026-08-20
expected_hours <- expected_days * 24
expected_total <- expected_hours * 64
if (nrow(dt) != expected_total) stop("Total row count does not match expected coverage.")
cat("Row count: PASS\n")

# ---- 15. DISTRICT COVERAGE -------------------------------------------------
coverage <- dt[, .(
  rows = .N,
  unique_hours = uniqueN(datetime),
  start = min(datetime),
  end = max(datetime)
), by = .(district_key, district)]
coverage[, status := fifelse(rows == expected_hours & unique_hours == expected_hours, "PASS", "CHECK")]
fwrite(coverage, file.path(REPORT_DIR, "district_coverage.csv"))
if (any(coverage$status != "PASS")) stop("District coverage validation failed.")
cat("District coverage: PASS\n")

# ---- 16. BANGLADESH CALENDAR VALIDATION (for report only) -----------------
dt[, year_bd := as.integer(format(datetime, "%Y", tz = "Asia/Dhaka"))]
dt[, month_bd := as.integer(format(datetime, "%m", tz = "Asia/Dhaka"))]
dt[, hour_bd := as.integer(format(datetime, "%H", tz = "Asia/Dhaka"))]
calendar <- dt[, .(rows = .N, unique_hours = uniqueN(datetime),
                   start = min(datetime), end = max(datetime)),
               by = .(year_bd, month_bd)]
setorder(calendar, year_bd, month_bd)
fwrite(calendar, file.path(REPORT_DIR, "bangladesh_calendar_coverage.csv"))

# ---- 17. MISSING VALUE REPORT ---------------------------------------------
missing <- data.table(
  variable = names(dt),
  missing_count = sapply(dt, function(x) sum(is.na(x)))
)
missing[, missing_percent := 100 * missing_count / nrow(dt)]
fwrite(missing, file.path(REPORT_DIR, "missing_value_report.csv"))

# ---- 18. RANGE CHECKS ------------------------------------------------------
range_checks <- list(
  temperature_2m = c(-60, 60),
  relative_humidity_2m = c(0, 100),
  surface_pressure = c(800, 1100),
  precipitation = c(0, 1000),
  cloud_cover = c(0, 100),
  wind_speed_10m = c(0, 100),
  wind_direction_10m = c(0, 360),
  pm10 = c(0, Inf), pm2_5 = c(0, Inf),
  carbon_monoxide = c(0, Inf), nitrogen_dioxide = c(0, Inf),
  sulphur_dioxide = c(0, Inf), ozone = c(0, Inf),
  us_aqi = c(0, Inf),
  us_aqi_pm2_5 = c(0, Inf), us_aqi_pm10 = c(0, Inf),
  us_aqi_nitrogen_dioxide = c(0, Inf),
  us_aqi_ozone = c(0, Inf),
  us_aqi_sulphur_dioxide = c(0, Inf),
  us_aqi_carbon_monoxide = c(0, Inf)
)

range_report <- rbindlist(lapply(names(range_checks), function(v) {
  x <- dt[[v]]
  lim <- range_checks[[v]]
  below <- sum(!is.na(x) & x < lim[1])
  above <- sum(!is.na(x) & x > lim[2])
  data.table(
    variable = v,
    observed_min = if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE),
    observed_max = if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE),
    below_min_count = below,
    above_max_count = above,
    status = if (below == 0 && above == 0) "PASS" else "CHECK"
  )
}), fill = TRUE)
fwrite(range_report, file.path(REPORT_DIR, "range_check_report.csv"))

# ============================================================================
# 19. DIAGNOSTIC PLOTS
# ============================================================================

cat("\nGenerating diagnostic plots...\n")

# ---- 19a. Missingness bar plot (excluding temporary columns) ---------------
missing_plot <- missing[variable != "datetime" & variable != "datetime_utc" & 
                          !grepl("^us_aqi_", variable) & # exclude sub-AQI
                          !variable %in% c("year_bd", "month_bd", "hour_bd")]
p_missing <- ggplot(missing_plot, aes(x = reorder(variable, missing_percent), y = missing_percent)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Missing Values by Variable (excluding sub-AQI)", x = NULL, y = "Missing (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIG_DIR, "missingness_barplot.png"), p_missing, width = 10, height = 8, dpi = 300)

# ---- 19b. Boxplots for key variables (excluding sub-AQI) -------------------
key_vars <- c("temperature_2m", "relative_humidity_2m", "surface_pressure",
              "wind_speed_10m", "pm2_5", "pm10", "ozone", "us_aqi")
boxplot_list <- lapply(key_vars, function(v) {
  dt_plot <- dt[!is.na(get(v)), .(value = get(v))]
  ggplot(dt_plot, aes(y = value)) +
    geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.size = 0.5) +
    labs(title = v, y = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
})
p_boxplots <- do.call(grid.arrange, c(boxplot_list, ncol = 4, top = "Boxplots of Key Variables (red dots = outliers)"))
ggsave(file.path(FIG_DIR, "boxplots_key_vars.png"), p_boxplots, width = 14, height = 10, dpi = 300)

# ---- 19c. Histograms for key variables -------------------------------------
hist_list <- lapply(key_vars, function(v) {
  dt_plot <- dt[!is.na(get(v)), .(value = get(v))]
  ggplot(dt_plot, aes(x = value)) +
    geom_histogram(bins = 80, fill = "steelblue", color = "white", alpha = 0.8) +
    labs(title = v, x = NULL, y = NULL) +
    theme_minimal()
})
p_hists <- do.call(grid.arrange, c(hist_list, ncol = 4, top = "Distributions of Key Variables"))
ggsave(file.path(FIG_DIR, "histograms_key_vars.png"), p_hists, width = 14, height = 10, dpi = 300)

# ---- 19d. District-wise mean AQI boxplot (spatial summary) -----------------
aqi_district <- dt[, .(mean_aqi = mean(us_aqi, na.rm = TRUE)), by = district]
aqi_district <- aqi_district[!is.na(mean_aqi)]
p_aqi_dist <- ggplot(aqi_district, aes(x = reorder(district, mean_aqi), y = mean_aqi)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(title = "Mean Overall AQI by District", x = NULL, y = "Mean AQI") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIG_DIR, "mean_aqi_by_district.png"), p_aqi_dist, width = 12, height = 10, dpi = 300)

# ---- 19e. Time series of daily mean AQI (overview) ------------------------
dt[, date := as.Date(datetime)]
daily_aqi <- dt[, .(daily_mean = mean(us_aqi, na.rm = TRUE)), by = date]
daily_aqi <- daily_aqi[!is.na(daily_mean)]
p_ts <- ggplot(daily_aqi, aes(x = date, y = daily_mean)) +
  geom_line(color = "darkblue", alpha = 0.7) +
  geom_smooth(method = "loess", span = 0.1, se = FALSE, color = "red") +
  labs(title = "Daily Mean Overall AQI (Bangladesh)", x = "Date", y = "Daily Mean AQI") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIG_DIR, "daily_aqi_timeseries.png"), p_ts, width = 14, height = 6, dpi = 300)

# ---- 19f. Correlation heatmap (excluding sub-AQI and temporary) -----------
# Select only necessary numeric columns: weather, raw pollutants, overall AQI
cor_vars <- c("temperature_2m", "relative_humidity_2m", "dew_point_2m",
              "surface_pressure", "precipitation", "cloud_cover",
              "wind_speed_10m", "wind_direction_10m",
              "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
              "sulphur_dioxide", "ozone", "us_aqi")
cor_dt <- dt[, ..cor_vars]
cor_mat <- cor(cor_dt, use = "pairwise.complete.obs")
png(file.path(FIG_DIR, "correlation_heatmap.png"), width = 1200, height = 1200, res = 150)
corrplot(cor_mat, method = "color", type = "upper", tl.cex = 0.7, number.cex = 0.5,
         addCoef.col = "black", tl.col = "black", title = "Correlation Matrix (Weather, Pollutants, AQI)", mar = c(0,0,2,0))
dev.off()

cat("Diagnostic plots saved to:", FIG_DIR, "\n")

# ============================================================================
# 20. DROP UNNECESSARY COLUMNS BEFORE SAVING
# ============================================================================
# Keep only columns needed downstream:
# - Weather
# - Raw pollutants
# - Overall AQI (us_aqi)
# - Spatial (lat/lon, district info)
# - Datetime (both local and UTC)
# Drop sub-AQI, temporary validation columns, and any other derived fields.
cols_to_keep <- c(
  "datetime", "datetime_utc",
  "district", "district_pcode", "district_key",
  "latitude", "longitude",
  "temperature_2m", "relative_humidity_2m", "dew_point_2m",
  "surface_pressure", "precipitation", "cloud_cover",
  "wind_speed_10m", "wind_direction_10m",
  "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
  "sulphur_dioxide", "ozone",
  "us_aqi"
)
dt_final <- dt[, ..cols_to_keep]

# ---- 21. CREATE EXPLICIT CSV‑SAFE DATETIME STRINGS -------------------------
dt_final[, datetime_utc := format(datetime_utc, tz = "UTC", format = "%Y-%m-%dT%H:%M:%SZ")]
dt_final[, datetime := format(datetime, tz = "Asia/Dhaka", format = "%Y-%m-%dT%H:%M:%S+06:00")]

# ---- 22. SAVE FINAL DATASET -----------------------------------------------
OUTPUT_FILE <- file.path(OUTPUT_DIR, "Bangladesh_64_District_Hourly_2023_2026_PREPROCESSED.csv")
fwrite(dt_final, OUTPUT_FILE)
cat("\nSaved:", OUTPUT_FILE, "\n")
cat("Kept columns:", ncol(dt_final), "\n")
cat("Dropped columns:", ncol(dt) - ncol(dt_final), "\n")

# ---- 23. FINAL SUMMARY ----------------------------------------------------
final_summary <- data.table(
  total_rows = nrow(dt_final),
  total_columns = ncol(dt_final),
  districts = uniqueN(dt_final$district_key),
  duplicate_records = dup_count,
  expected_rows = expected_total,
  start_datetime_bd = min(dt_final$datetime),
  end_datetime_bd = max(dt_final$datetime),
  start_datetime_utc = min(dt_final$datetime_utc),
  end_datetime_utc = max(dt_final$datetime_utc)
)
fwrite(final_summary, file.path(REPORT_DIR, "preprocessing_summary.csv"))

# ---- 24. FINAL CONSOLE MESSAGE --------------------------------------------
cat("\n============================================================\n")
cat("03 PREPROCESSING COMPLETE — SUCCESS\n")
cat("============================================================\n")
cat("\nRows:         ", format(nrow(dt_final), big.mark = ","))
cat("\nColumns:      ", ncol(dt_final))
cat("\nDistricts:    ", uniqueN(dt_final$district_key))
cat("\nOutput file:\n  ", OUTPUT_FILE)
cat("\nFigures saved to:\n  ", FIG_DIR)
cat("\nQuality reports saved to:\n  ", REPORT_DIR)
cat("\n\nIMPORTANT:\n")
cat("  - Sub‑AQI columns (us_aqi_pm2_5, etc.) have been dropped.\n")
cat("  - Only required columns for downstream pipeline are kept.\n")
cat("  - datetime     = Bangladesh local time (+06:00)\n")
cat("  - datetime_utc = original UTC time (Z)\n")
cat("All downstream work must use 'datetime' for Bangladesh local time.\n")
cat("============================================================\n")