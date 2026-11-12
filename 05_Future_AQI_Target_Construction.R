# ==============================================================================
# 05_Future_AQI_Target_Construction.R — CORRECTED LEAK‑SAFE VERSION
# ==============================================================================
#
# PROJECT:
# Multi-Horizon Early Warning of Future AQI Categories
# Across 64 Districts of Bangladesh
#
# PURPOSE:
#   Construct future overall-AQI targets at:
#       +1 hour, +3 hours, +6 hours, +12 hours, +24 hours
#
# IMPORTANT LEAKAGE PREVENTION:
#   - Targets are created using EXACT timestamp matching per district.
#   - NO row‑position shifting (which could leak if gaps exist).
#   - NO interpolation.
#   - NO aggregation across districts.
#   - The future columns are NEVER used as predictors in Step 08.
#
# INPUT:
#   Bangladesh_64_District_Hourly_2023_2026_PREPROCESSED.csv (Step 03)
#
# OUTPUT:
#   Bangladesh_64_District_Hourly_2023_2026_MULTI_HORIZON_AQI_TARGETS.csv
#   (with columns: aqi_t_plus_*h, aqi_category_t_plus_*h,
#    target_datetime_*h, target_exists_*h)
# ==============================================================================


# ---- 0. PACKAGES ------------------------------------------------------------
library(data.table)
library(lubridate)


# ---- 1. PATHS ---------------------------------------------------------------
BASE_DIR <- "F:/Semester 9/Data Science/Final Research/Main Project Work"

INPUT_FILE <- file.path(
  BASE_DIR,
  "03_Preprocessing",
  "outputs",
  "Bangladesh_64_District_Hourly_2023_2026_PREPROCESSED.csv"
)

OUTPUT_DIR <- file.path(
  BASE_DIR,
  "05_Future_AQI_Target_Construction",
  "outputs"
)

TABLE_DIR <- file.path(
  BASE_DIR,
  "05_Future_AQI_Target_Construction",
  "tables"
)

REPORT_DIR <- file.path(
  BASE_DIR,
  "05_Future_AQI_Target_Construction",
  "reports"
)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(REPORT_DIR, recursive = TRUE, showWarnings = FALSE)


# ---- 2. INPUT CHECK ---------------------------------------------------------
if (!file.exists(INPUT_FILE)) {
  stop("Preprocessed dataset not found:\n", INPUT_FILE)
}


# ---- 3. LOAD DATA -----------------------------------------------------------
cat("\n============================================================\n")
cat("05 — FUTURE AQI TARGET CONSTRUCTION (CORRECTED)\n")
cat("============================================================\n")

cat("\nLoading preprocessed dataset...\n")
dt <- fread(
  INPUT_FILE,
  colClasses = list(character = c("datetime", "datetime_utc")),
  showProgress = TRUE
)
cat("Rows: ", format(nrow(dt), big.mark = ","), "\n")
cat("Columns: ", ncol(dt), "\n")


# ---- 4. REQUIRED COLUMNS ----------------------------------------------------
required_cols <- c(
  "datetime", "datetime_utc",
  "district", "district_pcode", "district_key",
  "us_aqi"
)
missing <- setdiff(required_cols, names(dt))
if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))
cat("Required columns: PASS\n")


# ---- 5. PARSE BANGLADESH LOCAL TIME -----------------------------------------
dt[, datetime_bd := as.POSIXct(
  sub("\\+06:00$", "", datetime),
  format = "%Y-%m-%dT%H:%M:%S",
  tz = "Asia/Dhaka"
)]
if (anyNA(dt$datetime_bd)) stop("NA values in parsed Bangladesh datetime.")

# Ensure the data is sorted per district
setorder(dt, district_pcode, datetime_bd)


# ---- 6. CREATE FUTURE TARGET FUNCTION (LEAK‑SAFE, CORRECTED) ---------------
#
# This function performs an EXACT timestamp lookup within the same district.
# It uses a lookup table with the ACTUAL timestamps as keys, so the join
# correctly retrieves us_aqi from the FUTURE row.
#
# =============================================================================

create_horizon_target <- function(source_dt, horizon_hours) {
  
  cat("\n------------------------------------------------------------\n")
  cat("Creating +", horizon_hours, "hour target...\n")
  
  # ---- Build lookup table: actual datetime -> us_aqi (per district) ----
  lookup <- source_dt[, .(
    district_pcode,
    datetime_bd,
    future_us_aqi = us_aqi
  )]
  
  # ---- Build result table: source rows with target timestamps ----
  result <- source_dt[, .(
    district_pcode,
    source_datetime = datetime_bd,
    target_datetime = datetime_bd + hours(horizon_hours)
  )]
  
  # ---- EXACT JOIN: match district + target datetime to lookup datetime ----
  # This correctly pulls us_aqi from the FUTURE row (if it exists).
  result[
    lookup,
    on = .(district_pcode, target_datetime = datetime_bd),
    target_us_aqi := i.future_us_aqi
  ]
  
  # ---- Flag whether the target exists ----
  result[, target_exists := !is.na(target_us_aqi)]
  
  # ---- Validation: ensure target datetime is exactly source + horizon ----
  result[, expected_datetime := source_datetime + hours(horizon_hours)]
  result[, mapping_error_seconds := as.numeric(
    difftime(target_datetime, expected_datetime, units = "secs")
  )]
  if (any(abs(result$mapping_error_seconds) > 1e-9, na.rm = TRUE)) {
    stop("Target datetime mapping error for +", horizon_hours, "h.")
  }
  
  # ---- Summary ----
  total <- nrow(result)
  matched <- sum(result$target_exists)
  cat("Source rows:      ", format(total, big.mark = ","), "\n")
  cat("Future targets:   ", format(matched, big.mark = ","), "\n")
  cat("Missing targets:  ", format(total - matched, big.mark = ","), "\n")
  cat("Coverage:         ", round(100 * matched / total, 4), "%\n")
  
  # Return only the necessary columns
  result[, .(target_datetime, target_us_aqi, target_exists)]
}


# ---- 7. CREATE ALL HORIZONS -------------------------------------------------
horizons <- c(1, 3, 6, 12, 24)
horizon_results <- list()

for (h in horizons) {
  horizon_results[[paste0("h", h)]] <- create_horizon_target(dt, h)
}


# ---- 8. ATTACH FUTURE TARGETS TO SOURCE DATA --------------------------------
cat("\n============================================================\n")
cat("8. ATTACHING FUTURE TARGETS\n")
cat("============================================================\n")

for (h in horizons) {
  res <- horizon_results[[paste0("h", h)]]
  
  # Numeric future AQI
  dt[, paste0("aqi_t_plus_", h, "h") := res$target_us_aqi]
  
  # Target existence flag
  dt[, paste0("target_exists_", h, "h") := res$target_exists]
  
  # Target datetime as a clean string (for reference)
  dt[, paste0("target_datetime_", h, "h") :=
       format(res$target_datetime, "%Y-%m-%dT%H:%M:%S+06:00")]
}

cat("Future targets attached successfully.\n")


# ---- 9. CREATE CATEGORICAL TARGETS (EPA CLASSES) ---------------------------
aqi_category <- function(x) {
  fifelse(
    is.na(x), NA_character_,
    fifelse(x <= 50, "Good",
            fifelse(x <= 100, "Moderate",
                    fifelse(x <= 150, "Unhealthy for Sensitive Groups",
                            fifelse(x <= 200, "Unhealthy",
                                    fifelse(x <= 300, "Very Unhealthy", "Hazardous")))))
  )
}

for (h in horizons) {
  numeric_col <- paste0("aqi_t_plus_", h, "h")
  category_col <- paste0("aqi_category_t_plus_", h, "h")
  dt[, (category_col) := aqi_category(get(numeric_col))]
}

cat("Categorical future AQI targets created.\n")


# ---- 10. VALIDATION: TARGET COVERAGE SUMMARY --------------------------------
coverage_summary <- rbindlist(lapply(horizons, function(h) {
  exists_col <- paste0("target_exists_", h, "h")
  dt[, .(
    observations = .N,
    valid_targets = sum(get(exists_col), na.rm = TRUE)
  )][, horizon := paste0(h, "h")]
}))
coverage_summary[, coverage_percent := 100 * valid_targets / observations]
fwrite(coverage_summary, file.path(TABLE_DIR, "01_target_coverage_summary.csv"))
cat("\nTarget coverage summary:\n")
print(coverage_summary)


# ---- 11. TARGET CLASS DISTRIBUTIONS -----------------------------------------
target_dist <- rbindlist(lapply(horizons, function(h) {
  category_col <- paste0("aqi_category_t_plus_", h, "h")
  tmp <- dt[!is.na(get(category_col)), .(observations = .N), by = .(category = get(category_col))]
  tmp[, horizon := paste0(h, "h")]
  tmp[, percentage := 100 * observations / sum(observations)]
  tmp
}))
setorder(target_dist, horizon, category)
fwrite(target_dist, file.path(TABLE_DIR, "02_target_class_distribution.csv"))
cat("\nTarget class distribution:\n")
print(target_dist)


# ---- 12. DISTRICT × HORIZON COVERAGE ---------------------------------------
district_coverage <- rbindlist(lapply(horizons, function(h) {
  exists_col <- paste0("target_exists_", h, "h")
  dt[, .(
    source_rows = .N,
    valid_targets = sum(get(exists_col), na.rm = TRUE)
  ), by = .(district_pcode, district)][, horizon := paste0(h, "h")]
}))
district_coverage[, coverage_percent := 100 * valid_targets / source_rows]
fwrite(district_coverage, file.path(TABLE_DIR, "03_district_horizon_coverage.csv"))


# ---- 13. SAVE FINAL DATASET ------------------------------------------------
OUTPUT_FILE <- file.path(
  OUTPUT_DIR,
  "Bangladesh_64_District_Hourly_2023_2026_MULTI_HORIZON_AQI_TARGETS.csv"
)

fwrite(dt, OUTPUT_FILE)
cat("\nFinal dataset saved to:\n", OUTPUT_FILE, "\n")


# ---- 14. FINAL VALIDATION ---------------------------------------------------
verify <- fread(OUTPUT_FILE, nrows = 5)
expected_new_cols <- c(
  paste0("aqi_t_plus_", horizons, "h"),
  paste0("target_exists_", horizons, "h"),
  paste0("target_datetime_", horizons, "h"),
  paste0("aqi_category_t_plus_", horizons, "h")
)
missing_new <- setdiff(expected_new_cols, names(verify))
if (length(missing_new) > 0) {
  stop("Missing new target columns: ", paste(missing_new, collapse = ", "))
}
cat("Output validation: PASS – all target columns present.\n")


# ---- 15. FINAL CONSOLE MESSAGE ---------------------------------------------
cat("\n============================================================\n")
cat("05 — FUTURE AQI TARGET CONSTRUCTION — COMPLETE\n")
cat("============================================================\n")
cat("\nOutput file:\n  ", OUTPUT_FILE)
cat("\n\nRows:", format(nrow(dt), big.mark = ","))
cat("\nColumns:", ncol(dt))
cat("\nHorizons: +1h, +3h, +6h, +12h, +24h")
cat("\n\nLeakage prevention:\n")
cat("  - Exact timestamp matching (no row shifting)\n")
cat("  - Per‑district matching (no cross‑district leakage)\n")
cat("  - Lookup table uses ACTUAL timestamps, so future targets are correct\n")
cat("  - Future columns are LABELS only (never features)\n")
cat("============================================================\n")