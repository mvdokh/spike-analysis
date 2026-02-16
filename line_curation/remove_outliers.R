# Outlier Removal Script for Line Curvature Data
# This script identifies statistical outliers in the analog_output.csv file
# and removes corresponding rows from both analog_output.csv and lines_output.csv

# Load required libraries
library(readr)
library(dplyr)

# Set working directory to the line_curation folder
setwd("c:/Users/wanglab/Desktop/REPO/licking-and-spike-analysis/line_curation")

# Read the data files
cat("Reading data files...\n")
analog_data <- read_csv("analog_output.csv")
lines_data <- read_csv("lines_output.csv")

cat(sprintf("Original analog_output.csv: %d rows\n", nrow(analog_data)))
cat(sprintf("Original lines_output.csv: %d rows\n", nrow(lines_data)))

# Display basic statistics of the Data column
cat("\nOriginal Data Statistics:\n")
cat(sprintf("Min: %.4f\n", min(analog_data$Data, na.rm = TRUE)))
cat(sprintf("Max: %.4f\n", max(analog_data$Data, na.rm = TRUE)))
cat(sprintf("Mean: %.4f\n", mean(analog_data$Data, na.rm = TRUE)))
cat(sprintf("Median: %.4f\n", median(analog_data$Data, na.rm = TRUE)))
cat(sprintf("Standard Deviation: %.4f\n", sd(analog_data$Data, na.rm = TRUE)))

# Method 1: IQR-based outlier detection (more conservative)
Q1 <- quantile(analog_data$Data, 0.25, na.rm = TRUE)
Q3 <- quantile(analog_data$Data, 0.75, na.rm = TRUE)
IQR_value <- Q3 - Q1
lower_bound_iqr <- Q1 - 1.5 * IQR_value
upper_bound_iqr <- Q3 + 1.5 * IQR_value

iqr_outliers <- analog_data$Data < lower_bound_iqr | analog_data$Data > upper_bound_iqr

cat(sprintf("\nIQR Method (1.5 * IQR):\n"))
cat(sprintf("Lower bound: %.4f\n", lower_bound_iqr))
cat(sprintf("Upper bound: %.4f\n", upper_bound_iqr))
cat(sprintf("Number of outliers detected: %d\n", sum(iqr_outliers, na.rm = TRUE)))

# Method 2: Modified Z-score method (more sensitive)
median_val <- median(analog_data$Data, na.rm = TRUE)
mad_val <- mad(analog_data$Data, na.rm = TRUE)
modified_z_scores <- 0.6745 * (analog_data$Data - median_val) / mad_val
z_outliers <- abs(modified_z_scores) > 3.5

cat(sprintf("\nModified Z-score Method (threshold = 3.5):\n"))
cat(sprintf("Number of outliers detected: %d\n", sum(z_outliers, na.rm = TRUE)))

# Method 3: Standard Z-score method
z_scores <- abs(scale(analog_data$Data))
standard_z_outliers <- z_scores > 3

cat(sprintf("\nStandard Z-score Method (threshold = 3):\n"))
cat(sprintf("Number of outliers detected: %d\n", sum(standard_z_outliers, na.rm = TRUE)))

# Combine methods - use IQR as primary method (more conservative)
# You can change this to use a different method or combination
outlier_indices <- which(iqr_outliers)

if (length(outlier_indices) > 0) {
  cat(sprintf("\n=== Using IQR Method ===\n"))
  cat(sprintf("Removing %d outlier rows...\n", length(outlier_indices)))
  
  # Get the Time/Frame values that will be removed
  outlier_times <- analog_data$Time[outlier_indices]
  
  cat("Outlier values being removed:\n")
  outlier_summary <- analog_data[outlier_indices, ]
  print(outlier_summary)
  
  # Remove outliers from analog_data
  analog_clean <- analog_data[-outlier_indices, ]
  
  # Remove corresponding rows from lines_data based on matching Frame values
  lines_clean <- lines_data[!lines_data$Frame %in% outlier_times, ]
  
  cat(sprintf("\nCleaned analog_output.csv: %d rows (removed %d)\n", 
              nrow(analog_clean), nrow(analog_data) - nrow(analog_clean)))
  cat(sprintf("Cleaned lines_output.csv: %d rows (removed %d)\n", 
              nrow(lines_clean), nrow(lines_data) - nrow(lines_clean)))
  
  # Display cleaned statistics
  cat("\nCleaned Data Statistics:\n")
  cat(sprintf("Min: %.4f\n", min(analog_clean$Data, na.rm = TRUE)))
  cat(sprintf("Max: %.4f\n", max(analog_clean$Data, na.rm = TRUE)))
  cat(sprintf("Mean: %.4f\n", mean(analog_clean$Data, na.rm = TRUE)))
  cat(sprintf("Median: %.4f\n", median(analog_clean$Data, na.rm = TRUE)))
  cat(sprintf("Standard Deviation: %.4f\n", sd(analog_clean$Data, na.rm = TRUE)))
  
  # Save the cleaned datasets
  write_csv(analog_clean, "analog_output_clean.csv")
  write_csv(lines_clean, "lines_output_clean.csv")
  
  cat("\nCleaned files saved as:\n")
  cat("- analog_output_clean.csv\n")
  cat("- lines_output_clean.csv\n")
  
} else {
  cat("\nNo outliers detected with the current method.\n")
  cat("You may want to adjust the outlier detection parameters.\n")
}

# Optional: Create a visualization of the data distribution
# (requires ggplot2 package)
if (require(ggplot2, quietly = TRUE)) {
  
  # Create histogram of original data
  p1 <- ggplot(analog_data, aes(x = Data)) +
    geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
    geom_vline(xintercept = c(lower_bound_iqr, upper_bound_iqr), 
               color = "red", linetype = "dashed") +
    ggtitle("Distribution of Line Curvature Data\n(Red lines show IQR outlier boundaries)") +
    xlab("Line Curvature") +
    ylab("Frequency")
  
  ggsave("data_distribution.png", p1, width = 10, height = 6, dpi = 300)
  cat("\nData distribution plot saved as 'data_distribution.png'\n")
  
  # Boxplot comparison
  if (exists("analog_clean")) {
    combined_data <- rbind(
      data.frame(Data = analog_data$Data, Dataset = "Original"),
      data.frame(Data = analog_clean$Data, Dataset = "Cleaned")
    )
    
    p2 <- ggplot(combined_data, aes(x = Dataset, y = Data)) +
      geom_boxplot() +
      ggtitle("Comparison of Original vs Cleaned Data") +
      ylab("Line Curvature")
    
    ggsave("boxplot_comparison.png", p2, width = 8, height = 6, dpi = 300)
    cat("Boxplot comparison saved as 'boxplot_comparison.png'\n")
  }
}

cat("\nScript completed successfully!\n")