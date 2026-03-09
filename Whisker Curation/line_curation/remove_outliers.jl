#!/usr/bin/env julia
"""
Outlier Removal Script for Line Curvature Data
This script identifies statistical outliers in the analog_output.csv file
and removes corresponding rows from both analog_output.csv and lines_output.csv
"""

using DataFrames
using CSV
using Statistics
using StatsBase  
using Plots
using Printf

"""
Detect outliers using the Interquartile Range (IQR) method
"""
function detect_outliers_iqr(data::Vector, multiplier::Float64=1.5)
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = (data .< lower_bound) .| (data .> upper_bound)
    return outliers, lower_bound, upper_bound
end

"""
Detect outliers using Modified Z-score with Median Absolute Deviation (MAD)
"""
function detect_outliers_modified_zscore(data::Vector, threshold::Float64=3.5)
    median_val = median(data)
    mad_val = median(abs.(data .- median_val))
    modified_z_scores = 0.6745 * (data .- median_val) / mad_val
    outliers = abs.(modified_z_scores) .> threshold
    return outliers
end

"""
Detect outliers using standard Z-score method
"""
function detect_outliers_zscore(data::Vector, threshold::Float64=3.0)
    z_scores = abs.((data .- mean(data)) / std(data))
    outliers = z_scores .> threshold
    return outliers
end

function main()
    # Set working directory
    cd("c:/Users/wanglab/Desktop/REPO/licking-and-spike-analysis/line_curation")
    
    println("Reading data files...")
    # Read the data files
    try
        analog_data = CSV.read("analog_output.csv", DataFrame)
        lines_data = CSV.read("lines_output.csv", DataFrame)
        
        println("Original analog_output.csv: $(nrow(analog_data)) rows")
        println("Original lines_output.csv: $(nrow(lines_data)) rows")
        
        # Display basic statistics of the Data column
        data_col = analog_data.Data
        println("\nOriginal Data Statistics:")
        @printf "Min: %.4f\n" minimum(data_col)
        @printf "Max: %.4f\n" maximum(data_col)
        @printf "Mean: %.4f\n" mean(data_col)
        @printf "Median: %.4f\n" median(data_col)
        @printf "Standard Deviation: %.4f\n" std(data_col)
        @printf "Range: %.4f\n" (maximum(data_col) - minimum(data_col))
        
        # Method 1: IQR-based outlier detection (more conservative)
        iqr_outliers, lower_bound_iqr, upper_bound_iqr = detect_outliers_iqr(data_col, 1.5)
        
        println("\nIQR Method (1.5 * IQR):")
        @printf "Lower bound: %.4f\n" lower_bound_iqr
        @printf "Upper bound: %.4f\n" upper_bound_iqr
        println("Number of outliers detected: $(sum(iqr_outliers))")
        
        # Method 2: Modified Z-score method
        z_outliers = detect_outliers_modified_zscore(data_col, 3.5)
        println("\nModified Z-score Method (threshold = 3.5):")
        println("Number of outliers detected: $(sum(z_outliers))")
        
        # Method 3: Standard Z-score method
        standard_z_outliers = detect_outliers_zscore(data_col, 3.0)
        println("\nStandard Z-score Method (threshold = 3):")
        println("Number of outliers detected: $(sum(standard_z_outliers))")
        
        # Use IQR method as primary (you can change this)
        outlier_mask = iqr_outliers
        outlier_indices = findall(outlier_mask)
        
        if length(outlier_indices) > 0
            println("\n=== Using IQR Method ===")
            println("Removing $(length(outlier_indices)) outlier rows...")
            
            # Get the Time/Frame values that will be removed
            outlier_times = analog_data.Time[outlier_indices]
            
            println("\nOutlier values being removed:")
            outlier_summary = analog_data[outlier_indices, [:Time, :Data]]
            println(outlier_summary)
            
            # Remove outliers from analog_data
            analog_clean = analog_data[.!outlier_mask, :]
            
            # Remove corresponding rows from lines_data based on matching Frame values
            lines_clean = lines_data[.!(lines_data.Frame .∈ Ref(outlier_times)), :]
            
            println("\nCleaned analog_output.csv: $(nrow(analog_clean)) rows (removed $((nrow(analog_data) - nrow(analog_clean))))")
            println("Cleaned lines_output.csv: $(nrow(lines_clean)) rows (removed $((nrow(lines_data) - nrow(lines_clean))))")
            
            # Display cleaned statistics
            clean_data_col = analog_clean.Data
            println("\nCleaned Data Statistics:")
            @printf "Min: %.4f\n" minimum(clean_data_col)
            @printf "Max: %.4f\n" maximum(clean_data_col)
            @printf "Mean: %.4f\n" mean(clean_data_col)
            @printf "Median: %.4f\n" median(clean_data_col)
            @printf "Standard Deviation: %.4f\n" std(clean_data_col)
            @printf "Range: %.4f\n" (maximum(clean_data_col) - minimum(clean_data_col))
            
            # Save the cleaned datasets with fixed decimal formatting
            # Format the Data column to 10 decimal places without scientific notation
            analog_clean_formatted = copy(analog_clean)
            analog_clean_formatted.Data = [@sprintf("%.10f", x) for x in analog_clean_formatted.Data]
            CSV.write("analog_output_clean.csv", analog_clean_formatted)
            CSV.write("lines_output_clean.csv", lines_clean)
            
            println("\nCleaned files saved as:")
            println("- analog_output_clean.csv")
            println("- lines_output_clean.csv")
            
            # Create visualizations
            try
                # Create subplot layout
                p1 = histogram(data_col, bins=50, alpha=0.7, color=:blue, 
                             title="Distribution of Original Line Curvature Data",
                             xlabel="Line Curvature", ylabel="Frequency",
                             grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
                vline!(p1, [lower_bound_iqr], color=:red, linestyle=:dash, linewidth=2, label="IQR Lower Bound")
                vline!(p1, [upper_bound_iqr], color=:red, linestyle=:dash, linewidth=2, label="IQR Upper Bound")
                
                p2 = histogram(clean_data_col, bins=50, alpha=0.7, color=:green,
                             title="Distribution of Cleaned Line Curvature Data",
                             xlabel="Line Curvature", ylabel="Frequency",
                             grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
                
                # Box plot comparison
                p3 = boxplot([data_col, clean_data_col], labels=["Original" "Cleaned"],
                           title="Boxplot Comparison: Original vs Cleaned Data",
                           ylabel="Line Curvature",
                           grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
                
                # Time series plot showing outliers
                p4 = plot(analog_data.Time, data_col, alpha=0.6, linewidth=0.5, 
                         color=:blue, label="Original Data",
                         title="Time Series with Outliers Highlighted",
                         xlabel="Time/Frame", ylabel="Line Curvature",
                         grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
                scatter!(p4, analog_data.Time[outlier_indices], data_col[outlier_indices], 
                        color=:red, markersize=3, label="Outliers")
                
                # Combine plots
                final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
                savefig(final_plot, "outlier_analysis.png")
                println("\nOutlier analysis plot saved as 'outlier_analysis.png'")
                
            catch e
                println("Warning: Could not create visualization: $e")
            end
            
            # Save outlier summary with proper formatting
            outlier_report = DataFrame(
                Time = outlier_times,
                Data = [@sprintf("%.10f", x) for x in analog_data.Data[outlier_indices]],
                Method = fill("IQR_1.5", length(outlier_indices)),
                Lower_Bound = fill(@sprintf("%.10f", lower_bound_iqr), length(outlier_indices)),
                Upper_Bound = fill(@sprintf("%.10f", upper_bound_iqr), length(outlier_indices))
            )
            CSV.write("outliers_removed.csv", outlier_report)
            println("- outliers_removed.csv (summary of removed data points)")
            
        else
            println("\nNo outliers detected with the current method.")
            println("You may want to adjust the outlier detection parameters.")
        end
        
        println("\nScript completed successfully!")
        
    catch e
        if isa(e, SystemError)
            println("Error reading files: $e")
        else
            rethrow(e)
        end
    end
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end