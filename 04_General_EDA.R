# ==============================================================================
# 04_General_EDA.R — ADAPTED FOR PREPROCESSED DATASET (NO SUB‑AQI)
# ==============================================================================
# Project:
# Temporal Association Mining for Early Warning of Future AQI Episodes
# in Bangladesh
#
# PURPOSE:
#   General exploratory data analysis (EDA) of the preprocessed hourly dataset.
#   All sub‑AQI columns have been dropped; only overall AQI (us_aqi) is used.
#
# IMPORTANT TIMEZONE POLICY:
#   datetime       = Bangladesh local time, explicitly stored as +06:00
#   datetime_utc   = original UTC timestamp, explicitly stored as Z
#
#   ALL temporal interpretation in this script uses Bangladesh local time.
#
# THIS SCRIPT DOES NOT:
#   - create ARM categories
#   - create future AQI labels
#   - create horizon transactions
#   - alter the modeling dataset
#   - interpolate observations
#
# Aggregation is used ONLY for descriptive summaries and visualizations.
#
# ==============================================================================


# ==============================================================================
# 0. PACKAGES
# ==============================================================================

required_packages <- c(
  "data.table",
  "ggplot2",
  "lubridate",
  "sf",
  "scales",
  "viridis",
  "corrplot",
  "hexbin"
)

missing_packages <- setdiff(
  required_packages,
  rownames(installed.packages())
)

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

library(data.table)
library(ggplot2)
library(lubridate)
library(sf)
library(scales)
library(viridis)
library(corrplot)
library(hexbin)


# ==============================================================================
# 1. PATHS
# ==============================================================================

BASE_DIR <- "F:/Semester 9/Data Science/Final Research/Main Project Work"

INPUT_FILE <- file.path(
  BASE_DIR,
  "03_Preprocessing",
  "outputs",
  "Bangladesh_64_District_Hourly_2023_2026_PREPROCESSED.csv"
)

SHAPEFILE <- file.path(
  BASE_DIR,
  "bgd_admin_boundaries.shp"
)

SHAPE_LAYER <- "bgd_admin2"

EDA_DIR <- file.path(
  BASE_DIR,
  "04_General_EDA"
)

FIG_DIR <- file.path(
  EDA_DIR,
  "figures"
)

TABLE_DIR <- file.path(
  EDA_DIR,
  "tables"
)

REPORT_DIR <- file.path(
  EDA_DIR,
  "reports"
)

CATALOG_FILE <- file.path(
  EDA_DIR,
  "FIGURE_CATALOG.csv"
)


# ==============================================================================
# 2. OUTPUT FOLDERS
# ==============================================================================

figure_folders <- c(
  "01_dataset_overview",
  "02_missingness",
  "03_pollutant_distributions",
  "04_aqi_distributions",
  "05_weather_distributions",
  "06_temporal_analysis",
  "07_seasonal_analysis",
  "08_seasonal_spatial_maps",
  "09_pollutant_spatial_choropleths",
  "10_aqi_spatial_choropleths",
  "11_weather_spatial_maps",
  "12_pollutant_relationships",
  "13_weather_pollutant_relationships",
  "14_aqi_pollutant_relationships",
  "15_advanced_analysis"
)

for (f in figure_folders) {
  dir.create(
    file.path(FIG_DIR, f),
    recursive = TRUE,
    showWarnings = FALSE
  )
}

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(REPORT_DIR, recursive = TRUE, showWarnings = FALSE)


# ==============================================================================
# 3. INPUT VALIDATION
# ==============================================================================

if (!file.exists(INPUT_FILE)) {
  stop(
    paste0(
      "\nPreprocessed dataset not found:\n",
      INPUT_FILE
    )
  )
}

if (!file.exists(SHAPEFILE)) {
  stop(
    paste0(
      "\nBoundary datasource not found:\n",
      SHAPEFILE
    )
  )
}


# ==============================================================================
# 4. LOAD DATA
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("GENERAL EDA — START (PREPROCESSED DATA, NO SUB‑AQI)\n")
cat("============================================================\n")

cat("Loading preprocessed dataset...\n")

# Force timestamp columns to character to preserve explicit timezone strings.
dt <- fread(
  INPUT_FILE,
  colClasses = list(
    character = c(
      "datetime",
      "datetime_utc"
    )
  ),
  showProgress = TRUE
)

cat(
  "Rows:",
  format(nrow(dt), big.mark = ","),
  "\n"
)

cat(
  "Columns:",
  ncol(dt),
  "\n"
)


# ==============================================================================
# 5. REQUIRED COLUMNS
# ==============================================================================

# Note: sub‑AQI columns (us_aqi_pm2_5, etc.) are NOT required in preprocessed data.
required_columns <- c(
  # Time / geography
  "datetime",
  "datetime_utc",
  "district",
  "district_pcode",
  "district_key",
  "latitude",
  "longitude",
  
  # Weather
  "temperature_2m",
  "relative_humidity_2m",
  "dew_point_2m",
  "surface_pressure",
  "precipitation",
  "cloud_cover",
  "wind_speed_10m",
  "wind_direction_10m",
  
  # Pollutants
  "pm2_5",
  "pm10",
  "carbon_monoxide",
  "nitrogen_dioxide",
  "sulphur_dioxide",
  "ozone",
  
  # Overall AQI (only this; sub‑AQI dropped)
  "us_aqi"
)

missing_columns <- setdiff(
  required_columns,
  names(dt)
)

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

cat(
  "Required-column validation: PASS\n"
)


# ==============================================================================
# 6. DATETIME — CORRECT BANGLADESH TIME HANDLING
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("6. DATETIME VALIDATION\n")
cat("============================================================\n")

# Validate string formats
valid_bd_format <- grepl(
  "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\+06:00$",
  dt$datetime
)

valid_utc_format <- grepl(
  "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$",
  dt$datetime_utc
)

if (any(!valid_bd_format)) {
  stop("Invalid Bangladesh datetime format detected.")
}
if (any(!valid_utc_format)) {
  stop("Invalid UTC datetime format detected.")
}
cat("Datetime string-format validation: PASS\n")

# Parse Bangladesh local datetime
dt[
  ,
  datetime_bd :=
    as.POSIXct(
      sub("\\+06:00$", "", datetime),
      format = "%Y-%m-%dT%H:%M:%S",
      tz = "Asia/Dhaka"
    )
]

# Parse original UTC datetime
dt[
  ,
  datetime_utc_parsed :=
    as.POSIXct(
      datetime_utc,
      format = "%Y-%m-%dT%H:%M:%SZ",
      tz = "UTC"
    )
]

if (anyNA(dt$datetime_bd)) {
  stop("NA values detected in parsed Bangladesh datetime.")
}
if (anyNA(dt$datetime_utc_parsed)) {
  stop("NA values detected in parsed UTC datetime.")
}

# Verify timezone consistency
utc_as_bd <- with_tz(dt$datetime_utc_parsed, tzone = "Asia/Dhaka")
timezone_difference_seconds <- as.numeric(
  difftime(dt$datetime_bd, utc_as_bd, units = "secs")
)
timezone_errors <- sum(abs(timezone_difference_seconds) > 1e-9, na.rm = TRUE)
cat("Timezone consistency errors:", timezone_errors, "\n")
if (timezone_errors > 0) {
  stop("datetime and datetime_utc are inconsistent.")
}
cat("Timezone consistency: PASS\n")

# Extract temporal components (Bangladesh local time)
dt[, year := year(datetime_bd)]
dt[, month_num := month(datetime_bd)]
dt[, month := factor(month.abb[month_num], levels = month.abb)]
dt[, hour := hour(datetime_bd)]
dt[, weekday := factor(
  weekdays(datetime_bd),
  levels = c("Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
)]
dt[, season := fifelse(
  month_num %in% c(12,1,2), "Winter",
  fifelse(month_num %in% c(3,4,5), "Pre-monsoon",
          fifelse(month_num %in% c(6,7,8,9), "Monsoon", "Post-monsoon")
  )
)]
dt[, season := factor(season, levels = c("Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"))]


# ==============================================================================
# 7. VARIABLE DEFINITIONS
# ==============================================================================

pollutants <- c(
  "pm2_5",
  "pm10",
  "carbon_monoxide",
  "nitrogen_dioxide",
  "sulphur_dioxide",
  "ozone"
)

pollutant_labels <- c(
  pm2_5 = "PM2.5",
  pm10 = "PM10",
  carbon_monoxide = "CO",
  nitrogen_dioxide = "NO₂",
  sulphur_dioxide = "SO₂",
  ozone = "O₃"
)

# Only overall AQI (sub‑AQI removed)
aqi_vars <- c("us_aqi")
aqi_labels <- c(us_aqi = "Overall US AQI")

weather_vars <- c(
  "temperature_2m",
  "relative_humidity_2m",
  "dew_point_2m",
  "surface_pressure",
  "precipitation",
  "cloud_cover",
  "wind_speed_10m",
  "wind_direction_10m"
)

weather_labels <- c(
  temperature_2m = "Temperature (°C)",
  relative_humidity_2m = "Relative Humidity (%)",
  dew_point_2m = "Dew Point (°C)",
  surface_pressure = "Surface Pressure",
  precipitation = "Precipitation (mm)",
  cloud_cover = "Cloud Cover (%)",
  wind_speed_10m = "Wind Speed (m/s)",
  wind_direction_10m = "Wind Direction (°)"
)


# ==============================================================================
# 8. FIGURE CATALOG AND HELPERS
# ==============================================================================

figure_catalog <- data.table(
  figure_id = character(),
  figure_name = character(),
  category = character(),
  variable = character(),
  temporal_scale = character(),
  spatial_scale = character(),
  purpose = character(),
  file = character()
)

add_catalog <- function(
    id,
    name,
    category,
    variable,
    temporal,
    spatial,
    purpose,
    file
) {
  figure_catalog <<- rbind(
    figure_catalog,
    data.table(
      figure_id = id,
      figure_name = name,
      category = category,
      variable = variable,
      temporal_scale = temporal,
      spatial_scale = spatial,
      purpose = purpose,
      file = file
    ),
    fill = TRUE
  )
}

save_plot <- function(
    p,
    filename,
    folder,
    width = 10,
    height = 7
) {
  path <- file.path(FIG_DIR, folder, filename)
  ggsave(path, p, width = width, height = height, dpi = 300, bg = "white")
  path
}

theme_paper <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11),
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )


# ==============================================================================
# 9. DATASET OVERVIEW
# ==============================================================================

dataset_summary <- data.table(
  rows = nrow(dt),
  columns = ncol(dt),
  districts = uniqueN(dt$district_pcode),
  start_utc = min(dt$datetime_utc_parsed),
  end_utc = max(dt$datetime_utc_parsed),
  start_bangladesh = min(dt$datetime_bd),
  end_bangladesh = max(dt$datetime_bd)
)
fwrite(dataset_summary, file.path(TABLE_DIR, "01_dataset_summary.csv"))

# Observations by district
district_counts <- dt[, .(observations = .N), by = .(district_pcode, district)]
setorder(district_counts, observations)
p <- ggplot(district_counts, aes(x = reorder(district, observations), y = observations)) +
  geom_col() + coord_flip() +
  scale_y_continuous(labels = comma) +
  labs(title = "Hourly Observations by District", x = "District", y = "Observations") +
  theme_paper
file <- save_plot(p, "EDA_001_observations_by_district.png", "01_dataset_overview", 11, 12)
add_catalog("EDA_001", "Hourly Observations by District", "Dataset Overview", "All", "Full period", "District", "Verify spatial coverage", file)

# Observations by year
year_counts <- dt[, .(observations = .N), by = year]
p <- ggplot(year_counts, aes(x = factor(year), y = observations)) +
  geom_col() + scale_y_continuous(labels = comma) +
  labs(title = "Hourly Observations by Year", x = "Year (Bangladesh local time)", y = "Observations") +
  theme_paper
file <- save_plot(p, "EDA_002_observations_by_year.png", "01_dataset_overview")
add_catalog("EDA_002", "Hourly Observations by Year", "Dataset Overview", "All", "Annual", "National", "Verify temporal coverage", file)


# ==============================================================================
# 10. MISSINGNESS
# ==============================================================================

missing_summary <- data.table(
  variable = names(dt),
  missing_count = sapply(dt, function(x) sum(is.na(x)))
)
missing_summary[, missing_percent := 100 * missing_count / nrow(dt)]
fwrite(missing_summary, file.path(TABLE_DIR, "02_missingness_summary.csv"))

p <- ggplot(missing_summary, aes(x = reorder(variable, missing_percent), y = missing_percent)) +
  geom_col() + coord_flip() +
  labs(title = "Missing Values by Variable", x = "Variable", y = "Missing (%)") +
  theme_paper
file <- save_plot(p, "EDA_003_missingness.png", "02_missingness", 11, 9)
add_catalog("EDA_003", "Missingness by Variable", "Missingness", "All", "Full period", "National", "Verify data completeness", file)


# ==============================================================================
# 11. POLLUTANT DISTRIBUTIONS
# ==============================================================================

for (v in pollutants) {
  label <- pollutant_labels[[v]]
  plot_dt <- dt[, .(value = get(v))][is.finite(value)]
  
  # Histogram + density
  p <- ggplot(plot_dt, aes(x = value)) +
    geom_histogram(aes(y = after_stat(density)), bins = 80) +
    geom_density(linewidth = 0.8) +
    labs(title = paste(label, "Distribution"), x = label, y = "Density") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_hist_density.png"), "03_pollutant_distributions")
  add_catalog(paste0("P_", v, "_01"), paste(label, "Histogram and Density"), "Pollutant Distribution", label, "Full period", "National", "Characterize distribution", file)
  
  # ECDF
  p <- ggplot(plot_dt, aes(x = value)) + stat_ecdf() +
    labs(title = paste(label, "ECDF"), x = label, y = "Cumulative probability") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_ecdf.png"), "03_pollutant_distributions")
  add_catalog(paste0("P_", v, "_02"), paste(label, "ECDF"), "Pollutant Distribution", label, "Full period", "National", "Show cumulative exposure", file)
  
  # District boxplot
  box_dt <- dt[, .(value = get(v)), by = district]
  p <- ggplot(box_dt, aes(x = reorder(district, value, FUN = median), y = value)) +
    geom_boxplot(outlier.size = 0.15) + coord_flip() +
    labs(title = paste(label, "Distribution by District"), x = "District", y = label) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_district_boxplot.png"), "03_pollutant_distributions", 11, 12)
  add_catalog(paste0("P_", v, "_03"), paste(label, "District Boxplots"), "Pollutant Distribution", label, "Full period", "District", "Compare district-level distributions", file)
  
  # Seasonal violin
  seasonal_dt <- dt[, .(value = get(v)), by = season]
  p <- ggplot(seasonal_dt, aes(x = season, y = value)) +
    geom_violin(trim = TRUE) + geom_boxplot(width = 0.12) +
    labs(title = paste(label, "Seasonal Distribution"), x = "Season (Bangladesh local time)", y = label) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_seasonal_distribution.png"), "03_pollutant_distributions")
  add_catalog(paste0("P_", v, "_04"), paste(label, "Seasonal Distribution"), "Pollutant Distribution", label, "Seasonal", "National", "Compare seasonal distributions", file)
}


# ==============================================================================
# 12. AQI DISTRIBUTIONS (only overall AQI)
# ==============================================================================

for (v in aqi_vars) {
  label <- aqi_labels[[v]]
  plot_dt <- dt[, .(value = get(v))][is.finite(value)]
  
  # Histogram
  p <- ggplot(plot_dt, aes(x = value)) +
    geom_histogram(bins = 80) +
    labs(title = paste(label, "Distribution"), x = label, y = "Observations") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_histogram.png"), "04_aqi_distributions")
  add_catalog(paste0("A_", v, "_01"), paste(label, "Histogram"), "AQI Distribution", label, "Full period", "National", "Characterize AQI distribution", file)
  
  # ECDF
  p <- ggplot(plot_dt, aes(x = value)) + stat_ecdf() +
    labs(title = paste(label, "ECDF"), x = label, y = "Cumulative probability") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_ecdf.png"), "04_aqi_distributions")
  add_catalog(paste0("A_", v, "_02"), paste(label, "ECDF"), "AQI Distribution", label, "Full period", "National", "Show cumulative AQI", file)
  
  # District boxplot
  box_dt <- dt[, .(value = get(v)), by = district]
  p <- ggplot(box_dt, aes(x = reorder(district, value, FUN = median), y = value)) +
    geom_boxplot(outlier.size = 0.15) + coord_flip() +
    labs(title = paste(label, "by District"), x = "District", y = label) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_district_boxplot.png"), "04_aqi_distributions", 11, 12)
  add_catalog(paste0("A_", v, "_03"), paste(label, "District Boxplots"), "AQI Distribution", label, "Full period", "District", "Compare district AQI burden", file)
}


# ==============================================================================
# 13. WEATHER DISTRIBUTIONS
# ==============================================================================

for (v in weather_vars) {
  label <- weather_labels[[v]]
  plot_dt <- dt[, .(value = get(v))][is.finite(value)]
  
  # Histogram + density
  p <- ggplot(plot_dt, aes(x = value)) +
    geom_histogram(aes(y = after_stat(density)), bins = 80) +
    geom_density(linewidth = 0.8) +
    labs(title = paste(label, "Distribution"), x = label, y = "Density") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_hist_density.png"), "05_weather_distributions")
  add_catalog(paste0("W_", v, "_01"), paste(label, "Histogram and Density"), "Weather Distribution", label, "Full period", "National", "Characterize meteorological distribution", file)
  
  # Seasonal violin
  seasonal_dt <- dt[, .(value = get(v)), by = season]
  p <- ggplot(seasonal_dt, aes(x = season, y = value)) +
    geom_violin() + geom_boxplot(width = 0.12) +
    labs(title = paste(label, "Seasonal Distribution"), x = "Season (Bangladesh local time)", y = label) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_seasonal.png"), "05_weather_distributions")
  add_catalog(paste0("W_", v, "_02"), paste(label, "Seasonal Distribution"), "Weather Distribution", label, "Seasonal", "National", "Compare meteorological seasons", file)
}


# ==============================================================================
# 14. DESCRIPTIVE AQI CATEGORY PLOT
# ==============================================================================

aqi_category <- function(x) {
  fifelse(
    x <= 50, "Good",
    fifelse(x <= 100, "Moderate",
            fifelse(x <= 150, "Unhealthy for Sensitive Groups",
                    fifelse(x <= 200, "Unhealthy",
                            fifelse(x <= 300, "Very Unhealthy", "Hazardous"))))
  )
}

dt[, overall_aqi_descriptive_category := aqi_category(us_aqi)]
aqi_category_counts <- dt[, .(observations = .N), by = overall_aqi_descriptive_category]
aqi_category_counts[, percentage := 100 * observations / sum(observations)]
fwrite(aqi_category_counts, file.path(TABLE_DIR, "03_overall_aqi_descriptive_categories.csv"))

p <- ggplot(aqi_category_counts, aes(x = reorder(overall_aqi_descriptive_category, observations), y = observations)) +
  geom_col() + coord_flip() + scale_y_continuous(labels = comma) +
  labs(title = "Overall US AQI Severity Distribution", x = "AQI Category", y = "Hourly Observations") +
  theme_paper
file <- save_plot(p, "EDA_010_overall_aqi_severity.png", "04_aqi_distributions", 11, 8)
add_catalog("A_CAT_01", "Overall AQI Severity Distribution", "AQI Descriptive Analysis", "Overall AQI", "Full period", "National", "Describe AQI severity before ARM categorization", file)


# ==============================================================================
# 15. TEMPORAL — DIURNAL
# ==============================================================================

for (v in c(pollutants, "us_aqi")) {
  label <- ifelse(v == "us_aqi", "Overall US AQI", pollutant_labels[[v]])
  hourly <- dt[, .(mean_value = mean(get(v), na.rm = TRUE)), by = hour]
  p <- ggplot(hourly, aes(x = hour, y = mean_value)) +
    geom_line(linewidth = 0.9) + geom_point() +
    scale_x_continuous(breaks = 0:23) +
    labs(title = paste(label, "Mean Diurnal Pattern"), x = "Hour of day (Bangladesh time)", y = paste("Mean", label)) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_diurnal.png"), "06_temporal_analysis")
  add_catalog(paste0("T_DIURNAL_", v), paste(label, "Mean Diurnal Pattern"), "Temporal Analysis", label, "Hourly", "National", "Identify within-day patterns", file)
}


# ==============================================================================
# 16. HOUR × MONTH HEATMAPS
# ==============================================================================

for (v in c(pollutants, "us_aqi")) {
  label <- ifelse(v == "us_aqi", "Overall US AQI", pollutant_labels[[v]])
  hm <- dt[, .(mean_value = mean(get(v), na.rm = TRUE)), by = .(month_num, month, hour)]
  p <- ggplot(hm, aes(x = hour, y = month, fill = mean_value)) +
    geom_tile() + scale_fill_viridis_c() +
    labs(title = paste(label, "Hour × Month Heatmap"), x = "Hour (Bangladesh time)", y = "Month", fill = "Mean") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_hour_month_heatmap.png"), "06_temporal_analysis", 11, 8)
  add_catalog(paste0("T_HEAT_", v), paste(label, "Hour × Month Heatmap"), "Temporal Analysis", label, "Hour × Month", "National", "Identify seasonal-diurnal interaction", file)
}


# ==============================================================================
# 17. DAY-OF-WEEK PATTERNS
# ==============================================================================

for (v in c(pollutants, "us_aqi")) {
  label <- ifelse(v == "us_aqi", "Overall US AQI", pollutant_labels[[v]])
  dow <- dt[, .(mean_value = mean(get(v), na.rm = TRUE)), by = weekday]
  p <- ggplot(dow, aes(x = weekday, y = mean_value)) +
    geom_col() +
    labs(title = paste(label, "by Day of Week"), x = "Day (Bangladesh local time)", y = paste("Mean", label)) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_weekday.png"), "06_temporal_analysis")
  add_catalog(paste0("T_DOW_", v), paste(label, "Day-of-Week Pattern"), "Temporal Analysis", label, "Weekly", "National", "Identify weekly structure", file)
}


# ==============================================================================
# 18. MONTHLY TIME SERIES
# ==============================================================================

monthly_aqi <- dt[, .(
  mean_aqi = mean(us_aqi, na.rm = TRUE),
  median_aqi = median(us_aqi, na.rm = TRUE),
  sd_aqi = sd(us_aqi, na.rm = TRUE)
), by = .(year, month_num)]
monthly_aqi[, date := as.Date(sprintf("%04d-%02d-01", year, month_num))]
fwrite(monthly_aqi, file.path(TABLE_DIR, "04_monthly_overall_aqi_summary.csv"))

p <- ggplot(monthly_aqi, aes(x = date, y = mean_aqi)) +
  geom_line(linewidth = 0.9) + geom_point() +
  labs(title = "Monthly Mean Overall US AQI", x = "Month (Bangladesh local time)", y = "Mean Overall AQI") +
  theme_paper
file <- save_plot(p, "EDA_020_monthly_aqi_timeseries.png", "06_temporal_analysis", 13, 7)
add_catalog("T_MONTH_AQI", "Monthly Overall AQI Time Series", "Temporal Analysis", "Overall AQI", "Monthly", "National", "Identify long-term temporal variation", file)


# ==============================================================================
# 19. YEAR × MONTH AQI HEATMAP
# ==============================================================================

p <- ggplot(monthly_aqi, aes(
  x = factor(month_num, levels = 1:12, labels = month.abb),
  y = factor(year),
  fill = mean_aqi
)) +
  geom_tile() + scale_fill_viridis_c() +
  labs(title = "Overall AQI — Year × Month Heatmap", x = "Month (Bangladesh local time)", y = "Year", fill = "Mean AQI") +
  theme_paper
file <- save_plot(p, "EDA_021_year_month_aqi_heatmap.png", "06_temporal_analysis", 11, 7)
add_catalog("T_YEAR_MONTH_AQI", "Year × Month AQI Heatmap", "Temporal Analysis", "Overall AQI", "Year × Month", "National", "Identify recurring temporal periods", file)


# ==============================================================================
# 20. SEASONAL POLLUTION
# ==============================================================================

seasonal_summary <- dt[, lapply(c(pollutants, "us_aqi"), function(v) mean(get(v), na.rm = TRUE)), by = season]
setnames(seasonal_summary, c("season", pollutants, "us_aqi"))
fwrite(seasonal_summary, file.path(TABLE_DIR, "05_seasonal_mean_summary.csv"))

for (v in c(pollutants, "us_aqi")) {
  label <- ifelse(v == "us_aqi", "Overall US AQI", pollutant_labels[[v]])
  ss <- dt[, .(mean_value = mean(get(v), na.rm = TRUE)), by = season]
  p <- ggplot(ss, aes(x = season, y = mean_value)) + geom_col() +
    labs(title = paste(label, "Seasonal Mean"), x = "Season (Bangladesh local time)", y = paste("Mean", label)) +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_seasonal_mean.png"), "07_seasonal_analysis")
  add_catalog(paste0("S_MEAN_", v), paste(label, "Seasonal Mean"), "Seasonal Analysis", label, "Seasonal", "National", "Compare pollution across seasons", file)
}


# ==============================================================================
# 21. DISTRICT × SEASON HEATMAPS
# ==============================================================================

for (v in c(pollutants, "us_aqi")) {
  label <- ifelse(v == "us_aqi", "Overall US AQI", pollutant_labels[[v]])
  ds <- dt[, .(mean_value = mean(get(v), na.rm = TRUE)), by = .(district, season)]
  p <- ggplot(ds, aes(x = season, y = reorder(district, mean_value, FUN = mean), fill = mean_value)) +
    geom_tile() + scale_fill_viridis_c() +
    labs(title = paste(label, "District × Season Heatmap"), x = "Season (Bangladesh local time)", y = "District", fill = "Mean") +
    theme_paper
  file <- save_plot(p, paste0("EDA_", v, "_district_season_heatmap.png"), "07_seasonal_analysis", 11, 14)
  add_catalog(paste0("S_HEAT_", v), paste(label, "District × Season Heatmap"), "Seasonal Spatial Analysis", label, "Season × District", "District", "Identify spatial-seasonal differences", file)
}


# ==============================================================================
# 22. LOAD OFFICIAL DISTRICT LAYER ONCE
# ==============================================================================

cat("\nLoading official Bangladesh district polygons...\n")
district_sf <- st_read(SHAPEFILE, layer = SHAPE_LAYER, quiet = TRUE)
if (nrow(district_sf) != 64) {
  stop(paste("Expected 64 district polygons but found", nrow(district_sf)))
}
cat("District polygons:", nrow(district_sf), "\n")


# ==============================================================================
# 23. VALIDATE SPATIAL P-CODES
# ==============================================================================

data_pcodes <- sort(unique(dt$district_pcode))
map_pcodes <- sort(unique(district_sf$adm2_pcode))
if (!identical(data_pcodes, map_pcodes)) {
  stop(paste0("District P-code mismatch.\n",
              "Data-only codes: ", paste(setdiff(data_pcodes, map_pcodes), collapse = ", "),
              "\nMap-only codes: ", paste(setdiff(map_pcodes, data_pcodes), collapse = ", ")))
}
cat("District P-code validation: PASS — 64/64 matched.\n")


# ==============================================================================
# 24. DISTRICT SPATIAL SUMMARY
# ==============================================================================

district_summary <- dt[, .(
  observations = .N,
  mean_pm2_5 = mean(pm2_5, na.rm = TRUE),
  mean_pm10 = mean(pm10, na.rm = TRUE),
  mean_co = mean(carbon_monoxide, na.rm = TRUE),
  mean_no2 = mean(nitrogen_dioxide, na.rm = TRUE),
  mean_so2 = mean(sulphur_dioxide, na.rm = TRUE),
  mean_o3 = mean(ozone, na.rm = TRUE),
  mean_aqi = mean(us_aqi, na.rm = TRUE)
), by = .(district_pcode, district)]
fwrite(district_summary, file.path(TABLE_DIR, "06_district_spatial_summary.csv"))


# ==============================================================================
# 25. POLLUTANT CHOROPLETHS
# ==============================================================================

map_data <- merge(district_sf, district_summary, by.x = "adm2_pcode", by.y = "district_pcode", all.x = TRUE)
if (nrow(map_data) != 64) {
  stop(paste("Pollutant spatial join did not produce 64 districts. Rows:", nrow(map_data)))
}

map_pollutants <- c("mean_pm2_5", "mean_pm10", "mean_co", "mean_no2", "mean_so2", "mean_o3")
map_pollutant_labels <- c(
  mean_pm2_5 = "Mean PM2.5", mean_pm10 = "Mean PM10", mean_co = "Mean CO",
  mean_no2 = "Mean NO₂", mean_so2 = "Mean SO₂", mean_o3 = "Mean O₃"
)

for (v in map_pollutants) {
  label <- map_pollutant_labels[[v]]
  p <- ggplot(map_data) +
    geom_sf(aes(fill = .data[[v]]), color = "white", linewidth = 0.25) +
    geom_sf_text(aes(label = adm2_name), size = 2.2, color = "black", fontface = "bold") +
    scale_fill_viridis_c(option = "C", na.value = "grey90") +
    labs(title = paste(label, "Across Bangladesh Districts"),
         subtitle = "District-level spatial distribution", fill = label) +
    theme_void() +
    theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
          plot.subtitle = element_text(size = 11, hjust = 0.5),
          legend.title = element_text(face = "bold"))
  file <- save_plot(p, paste0("MAP_", v, "_district_choropleth_named.png"), "09_pollutant_spatial_choropleths", 12, 10)
  add_catalog(paste0("MAP_POLL_NAMED_", v), paste(label, "District Choropleth with Names"),
              "Spatial Analysis", label, "Full period", "District",
              "District-level pollutant heatmap with district names", file)
}
cat("Pollutant choropleths completed.\n")


# ==============================================================================
# 26. AQI CHOROPLETHS (only overall AQI)
# ==============================================================================

district_aqi_summary <- dt[, .(
  mean_aqi = mean(us_aqi, na.rm = TRUE)
), by = .(district_pcode, district)]
fwrite(district_aqi_summary, file.path(TABLE_DIR, "06_district_aqi_spatial_summary.csv"))

aqi_map_data <- merge(district_sf, district_aqi_summary, by.x = "adm2_pcode", by.y = "district_pcode", all.x = TRUE)
if (nrow(aqi_map_data) != 64) {
  stop(paste("AQI spatial join did not produce 64 districts. Rows:", nrow(aqi_map_data)))
}

p <- ggplot(aqi_map_data) +
  geom_sf(aes(fill = mean_aqi), color = "white", linewidth = 0.25) +
  geom_sf_text(aes(label = adm2_name), size = 2.2, color = "black", fontface = "bold") +
  scale_fill_viridis_c(option = "C", na.value = "grey90") +
  labs(title = "Mean Overall AQI Across Bangladesh Districts",
       subtitle = "District-level spatial distribution", fill = "Mean AQI") +
  theme_void() +
  theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        legend.title = element_text(face = "bold"))
file <- save_plot(p, "MAP_mean_aqi_district_choropleth_named.png", "10_aqi_spatial_choropleths", 12, 10)
add_catalog("MAP_AQI_NAMED_mean_aqi", "Mean Overall AQI District Choropleth with Names",
            "Spatial Analysis", "Overall AQI", "Full period", "District",
            "District-level AQI heatmap with district names", file)
cat("AQI choropleths completed.\n")


# ==============================================================================
# 27. SEASONAL SPATIAL CHOROPLEPTHS
# ==============================================================================

season_spatial <- dt[, .(
  mean_pm2_5 = mean(pm2_5, na.rm = TRUE),
  mean_pm10 = mean(pm10, na.rm = TRUE),
  mean_aqi = mean(us_aqi, na.rm = TRUE)
), by = .(district_pcode, district, season)]

season_map_vars <- c("mean_pm2_5", "mean_pm10", "mean_aqi")
season_map_labels <- c(
  mean_pm2_5 = "Mean PM2.5",
  mean_pm10 = "Mean PM10",
  mean_aqi = "Mean Overall AQI"
)

for (v in season_map_vars) {
  label <- season_map_labels[[v]]
  for (s in levels(dt$season)) {
    tmp <- season_spatial[season == s]
    map_tmp <- merge(district_sf, tmp, by.x = "adm2_pcode", by.y = "district_pcode", all.x = TRUE)
    if (nrow(map_tmp) != 64) {
      stop(paste("Seasonal spatial join failed for:", s, "| Rows:", nrow(map_tmp)))
    }
    p <- ggplot(map_tmp) +
      geom_sf(aes(fill = .data[[v]]), color = "white", linewidth = 0.25) +
      geom_sf_text(aes(label = adm2_name), size = 2.2, color = "black", fontface = "bold") +
      scale_fill_viridis_c(option = "C", na.value = "grey90") +
      labs(title = paste(label, "-", s),
           subtitle = "District-level seasonal spatial distribution", fill = label) +
      theme_void() +
      theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
            plot.subtitle = element_text(size = 11, hjust = 0.5),
            legend.title = element_text(face = "bold"))
    safe_s <- gsub("[^A-Za-z0-9]+", "_", s)
    file <- save_plot(p, paste0("MAP_", v, "_", safe_s, "_district_choropleth_named.png"),
                      "08_seasonal_spatial_maps", 12, 10)
    add_catalog(paste0("MAP_SEASON_NAMED_", v, "_", safe_s),
                paste(label, "Seasonal District Choropleth with Names -", s),
                "Spatial Seasonal Analysis", label, "Seasonal", "District",
                "Seasonal district heatmap with district names", file)
  }
}
cat("Seasonal spatial choropleths completed.\n")


# ==============================================================================
# 28. WEATHER SPATIAL CHOROPLETHS
# ==============================================================================

weather_spatial_summary <- dt[, .(
  mean_temperature = mean(temperature_2m, na.rm = TRUE),
  mean_relative_humidity = mean(relative_humidity_2m, na.rm = TRUE),
  mean_dew_point = mean(dew_point_2m, na.rm = TRUE),
  mean_surface_pressure = mean(surface_pressure, na.rm = TRUE),
  mean_precipitation = mean(precipitation, na.rm = TRUE),
  mean_cloud_cover = mean(cloud_cover, na.rm = TRUE),
  mean_wind_speed = mean(wind_speed_10m, na.rm = TRUE),
  mean_wind_direction = mean(wind_direction_10m, na.rm = TRUE)
), by = .(district_pcode, district)]
fwrite(weather_spatial_summary, file.path(TABLE_DIR, "07_district_weather_spatial_summary.csv"))

weather_map_data <- merge(district_sf, weather_spatial_summary, by.x = "adm2_pcode", by.y = "district_pcode", all.x = TRUE)
if (nrow(weather_map_data) != 64) {
  stop(paste("Weather spatial join did not produce 64 districts. Rows:", nrow(weather_map_data)))
}

map_weather <- c("mean_temperature", "mean_relative_humidity", "mean_dew_point",
                 "mean_surface_pressure", "mean_precipitation", "mean_cloud_cover",
                 "mean_wind_speed", "mean_wind_direction")
map_weather_labels <- c(
  mean_temperature = "Mean Temperature (°C)",
  mean_relative_humidity = "Mean Relative Humidity (%)",
  mean_dew_point = "Mean Dew Point (°C)",
  mean_surface_pressure = "Mean Surface Pressure",
  mean_precipitation = "Mean Precipitation (mm)",
  mean_cloud_cover = "Mean Cloud Cover (%)",
  mean_wind_speed = "Mean Wind Speed (m/s)",
  mean_wind_direction = "Mean Wind Direction (°)"
)

for (v in map_weather) {
  label <- map_weather_labels[[v]]
  p <- ggplot(weather_map_data) +
    geom_sf(aes(fill = .data[[v]]), color = "white", linewidth = 0.25) +
    geom_sf_text(aes(label = adm2_name), size = 2.2, color = "black", fontface = "bold") +
    scale_fill_viridis_c(option = "C", na.value = "grey90") +
    labs(title = paste(label, "Across Bangladesh Districts"),
         subtitle = "District-level spatial distribution", fill = label) +
    theme_void() +
    theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
          plot.subtitle = element_text(size = 11, hjust = 0.5),
          legend.title = element_text(face = "bold"))
  safe_v <- gsub("[^A-Za-z0-9]+", "_", v)
  file <- save_plot(p, paste0("MAP_", safe_v, "_district_choropleth_named.png"), "11_weather_spatial_maps", 12, 10)
  add_catalog(paste0("MAP_WEATHER_NAMED_", safe_v), paste(label, "District Choropleth with Names"),
              "Weather Spatial Analysis", label, "Full period", "District",
              "District-level weather heatmap with district names", file)
}
cat("Weather spatial choropleths completed.\n")


# ==============================================================================
# 29. POLLUTANT CORRELATION
# ==============================================================================

pollutant_cor <- cor(dt[, ..pollutants], use = "pairwise.complete.obs", method = "spearman")
write.csv(pollutant_cor, file.path(TABLE_DIR, "07_pollutant_spearman_correlation.csv"))
png(file.path(FIG_DIR, "12_pollutant_relationships", "EDA_100_pollutant_spearman_correlation.png"),
    width = 2600, height = 2400, res = 300)
corrplot(pollutant_cor, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", number.cex = 0.7)
dev.off()
add_catalog("REL_001", "Pollutant Spearman Correlation Matrix", "Pollutant Relationships",
            "All pollutants", "Full period", "National", "Identify pollutant co-variation",
            file.path(FIG_DIR, "12_pollutant_relationships", "EDA_100_pollutant_spearman_correlation.png"))


# ==============================================================================
# 30. WEATHER CORRELATION
# ==============================================================================

weather_cor <- cor(dt[, ..weather_vars], use = "pairwise.complete.obs", method = "spearman")
write.csv(weather_cor, file.path(TABLE_DIR, "08_weather_spearman_correlation.csv"))
png(file.path(FIG_DIR, "13_weather_pollutant_relationships", "EDA_101_weather_spearman_correlation.png"),
    width = 2600, height = 2600, res = 300)
corrplot(weather_cor, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", number.cex = 0.6)
dev.off()
add_catalog("REL_002", "Weather Spearman Correlation Matrix", "Weather Relationships",
            "Weather variables", "Full period", "National", "Identify meteorological co-variation",
            file.path(FIG_DIR, "13_weather_pollutant_relationships", "EDA_101_weather_spearman_correlation.png"))


# ==============================================================================
# 31. WEATHER + POLLUTANT + AQI CORRELATION
# ==============================================================================

relationship_vars <- c(weather_vars, pollutants, "us_aqi")
relationship_cor <- cor(dt[, ..relationship_vars], use = "pairwise.complete.obs", method = "spearman")
write.csv(relationship_cor, file.path(TABLE_DIR, "09_weather_pollutant_aqi_spearman_correlation.csv"))
png(file.path(FIG_DIR, "13_weather_pollutant_relationships", "EDA_102_weather_pollutant_aqi_correlation.png"),
    width = 3300, height = 3000, res = 300)
corrplot(relationship_cor, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.65, number.cex = 0.4)
dev.off()
add_catalog("REL_003", "Weather-Pollutant-AQI Correlation Matrix", "Weather-Pollutant Relationships",
            "Weather + pollutants + AQI", "Full period", "National",
            "Explore meteorological associations with pollution",
            file.path(FIG_DIR, "13_weather_pollutant_relationships", "EDA_102_weather_pollutant_aqi_correlation.png"))


# ==============================================================================
# 32. WEATHER vs PM2.5 HEXBIN
# ==============================================================================

weather_for_pm25 <- c("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure")
for (w in weather_for_pm25) {
  plot_dt <- dt[is.finite(get(w)) & is.finite(pm2_5), .(x = get(w), y = pm2_5)]
  p <- ggplot(plot_dt, aes(x = x, y = y)) + geom_hex(bins = 45) + scale_fill_viridis_c() +
    labs(title = paste("PM2.5 vs", weather_labels[[w]]), x = weather_labels[[w]], y = "PM2.5", fill = "Count") +
    theme_paper
  file <- save_plot(p, paste0("EDA_PM25_vs_", w, "_hexbin.png"), "13_weather_pollutant_relationships")
  add_catalog(paste0("WP_PM25_", w), paste("PM2.5 vs", weather_labels[[w]], "Hexbin"),
              "Weather-Pollutant Relationships", paste("PM2.5 +", weather_labels[[w]]),
              "Full period", "National", "Visualize dense weather-pollution relationships", file)
}


# ==============================================================================
# 33. POLLUTANT PAIR HEXBIN PLOTS
# ==============================================================================

pairs <- combn(pollutants, 2, simplify = FALSE)
pair_results <- data.table(pollutant_1 = character(), pollutant_2 = character(), spearman_rho = numeric())
for (pair in pairs) {
  v1 <- pair[1]; v2 <- pair[2]
  x <- dt[[v1]]; y <- dt[[v2]]
  keep <- is.finite(x) & is.finite(y)
  rho <- cor(x[keep], y[keep], method = "spearman")
  pair_results <- rbind(pair_results, data.table(pollutant_1 = v1, pollutant_2 = v2, spearman_rho = rho))
  plot_dt <- data.table(x = x, y = y)[is.finite(x) & is.finite(y)]
  p <- ggplot(plot_dt, aes(x = x, y = y)) + geom_hex(bins = 45) + scale_fill_viridis_c() +
    labs(title = paste(pollutant_labels[[v1]], "vs", pollutant_labels[[v2]]),
         subtitle = paste("Spearman ρ =", round(rho, 3)),
         x = pollutant_labels[[v1]], y = pollutant_labels[[v2]], fill = "Count") +
    theme_paper
  file <- save_plot(p, paste0("EDA_PAIR_", v1, "_", v2, ".png"), "12_pollutant_relationships")
  add_catalog(paste0("PAIR_", v1, "_", v2), paste(pollutant_labels[[v1]], "vs", pollutant_labels[[v2]]),
              "Pollutant Relationships", paste(pollutant_labels[[v1]], "+", pollutant_labels[[v2]]),
              "Full period", "National", "Explore pollutant co-variation", file)
}
fwrite(pair_results, file.path(TABLE_DIR, "10_pollutant_pair_spearman.csv"))


# ==============================================================================
# 34. AQI vs WEATHER CORRELATION RANKING
# ==============================================================================

aqi_weather_cor <- rbindlist(lapply(weather_vars, function(w) {
  x <- dt[[w]]; y <- dt$us_aqi
  keep <- is.finite(x) & is.finite(y)
  data.table(variable = w, label = weather_labels[[w]],
             spearman_rho = cor(x[keep], y[keep], method = "spearman"))
}))
setorder(aqi_weather_cor, spearman_rho)
fwrite(aqi_weather_cor, file.path(TABLE_DIR, "11_aqi_weather_correlation_ranking.csv"))
p <- ggplot(aqi_weather_cor, aes(x = reorder(label, spearman_rho), y = spearman_rho)) +
  geom_col() + coord_flip() +
  labs(title = "Weather Variables Associated with Overall AQI", x = "Weather variable", y = "Spearman correlation") +
  theme_paper
file <- save_plot(p, "EDA_120_aqi_weather_correlation_ranking.png", "14_aqi_pollutant_relationships", 11, 8)
add_catalog("AQI_WEATHER_RANK", "AQI-Weather Correlation Ranking", "AQI-Weather Relationships",
            "Overall AQI + weather", "Full period", "National",
            "Rank meteorological associations with AQI", file)


# ==============================================================================
# 35. AQI vs PM2.5 / PM10
# ==============================================================================

for (v in c("pm2_5", "pm10")) {
  plot_dt <- dt[is.finite(get(v)) & is.finite(us_aqi), .(x = get(v), y = us_aqi)]
  p <- ggplot(plot_dt, aes(x = x, y = y)) + geom_hex(bins = 50) + scale_fill_viridis_c() +
    labs(title = paste("Overall AQI vs", pollutant_labels[[v]]),
         x = pollutant_labels[[v]], y = "Overall US AQI", fill = "Count") +
    theme_paper
  file <- save_plot(p, paste0("EDA_AQI_vs_", v, "_hexbin.png"), "14_aqi_pollutant_relationships")
  add_catalog(paste0("AQI_POLL_", v), paste("Overall AQI vs", pollutant_labels[[v]]),
              "AQI-Pollutant Relationships", paste("Overall AQI +", pollutant_labels[[v]]),
              "Full period", "National", "Examine pollutant-AQI relationship", file)
}


# ==============================================================================
# 36. EXTREME AQI FREQUENCY BY DISTRICT
# ==============================================================================

extreme_summary <- dt[, .(
  total_hours = .N,
  aqi_101_plus = sum(us_aqi >= 101, na.rm = TRUE),
  aqi_151_plus = sum(us_aqi >= 151, na.rm = TRUE),
  aqi_201_plus = sum(us_aqi >= 201, na.rm = TRUE),
  aqi_301_plus = sum(us_aqi >= 301, na.rm = TRUE)
), by = .(district_pcode, district)]
extreme_summary[, `:=`(
  pct_101_plus = 100 * aqi_101_plus / total_hours,
  pct_151_plus = 100 * aqi_151_plus / total_hours,
  pct_201_plus = 100 * aqi_201_plus / total_hours,
  pct_301_plus = 100 * aqi_301_plus / total_hours
)]
fwrite(extreme_summary, file.path(TABLE_DIR, "12_district_aqi_extreme_frequency.csv"))
p <- ggplot(extreme_summary, aes(x = reorder(district, pct_101_plus), y = pct_101_plus)) +
  geom_col() + coord_flip() +
  labs(title = "District Frequency of Overall AQI ≥ 101", x = "District", y = "Hours (%)") +
  theme_paper
file <- save_plot(p, "EDA_130_district_aqi_101_plus.png", "15_advanced_analysis", 11, 12)
add_catalog("EXTREME_101", "District Frequency of AQI ≥ 101", "Extreme AQI Analysis",
            "Overall AQI", "Full period", "District", "Identify districts with frequent elevated AQI", file)


# ==============================================================================
# 37. (REMOVED) Pollutant-Specific AQI Dominance – sub‑AQI not available
# ==============================================================================
cat("Pollutant-specific AQI dominance skipped (sub‑AQI columns not present in preprocessed data).\n")


# ==============================================================================
# 38. FINAL EDA OUTPUTS
# ==============================================================================

fwrite(figure_catalog, CATALOG_FILE)
eda_summary <- data.table(
  observations = nrow(dt),
  columns = ncol(dt),
  districts = uniqueN(dt$district_pcode),
  pollutant_variables = length(pollutants),
  aqi_variables = length(aqi_vars),
  weather_variables = length(weather_vars),
  figures_generated = nrow(figure_catalog),
  tables_generated = length(list.files(TABLE_DIR, pattern = "\\.csv$"))
)
fwrite(eda_summary, file.path(REPORT_DIR, "EDA_summary.csv"))


# ==============================================================================
# 39. FINAL DATASET/TIME VALIDATION
# ==============================================================================

if (nrow(dt) <= 0) stop("EDA dataset contains zero rows.")
if (uniqueN(dt$district_pcode) != 64) stop("Final EDA district count is not 64.")
if (anyNA(dt$datetime_bd)) stop("Final EDA datetime_bd contains NA values.")
if (anyNA(dt$datetime_utc_parsed)) stop("Final EDA datetime_utc_parsed contains NA values.")
cat("\n============================================================\n")
cat("FINAL EDA VALIDATION\n")
cat("============================================================\n")
cat("Rows:", format(nrow(dt), big.mark = ","), "\n")
cat("Districts:", uniqueN(dt$district_pcode), "\n")
cat("Timezone consistency errors:", timezone_errors, "\n")
cat("Figure catalog entries:", nrow(figure_catalog), "\n")
cat("Final validation: PASS\n")


# ==============================================================================
# 40. CLEANUP EDA-ONLY COLUMNS
# ==============================================================================

dt[, c("datetime_bd", "datetime_utc_parsed", "year", "month_num", "month",
       "hour", "weekday", "season", "overall_aqi_descriptive_category") := NULL]
rm(utc_as_bd, timezone_difference_seconds, envir = .GlobalEnv)
gc(verbose = FALSE)


# ==============================================================================
# 41. FINAL
# ==============================================================================

cat("\n")
cat("============================================================\n")
cat("GENERAL EDA — COMPLETE\n")
cat("============================================================\n")
cat("Observations:", format(nrow(dt), big.mark = ","), "\n")
cat("Districts:", uniqueN(dt$district_pcode), "\n")
cat("Figures generated:", nrow(figure_catalog), "\n")
cat("Figure catalog:\n", CATALOG_FILE, "\n")
cat("Figures folder:\n", FIG_DIR, "\n")
cat("Tables folder:\n", TABLE_DIR, "\n")
cat("\nTimezone policy: All temporal EDA uses Bangladesh local time (+06:00).\n")
cat("UTC timestamp retained only as reference/audit time.\n")
cat("\nARM categories: NOT CREATED.\n")
cat("Future AQI labels: NOT CREATED.\n")
cat("Modeling dataset: NOT MODIFIED.\n")
cat("Spatial aggregation: DESCRIPTIVE VISUALIZATION ONLY.\n")
cat("============================================================\n")