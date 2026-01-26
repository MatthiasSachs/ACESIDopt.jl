#!/usr/bin/env julia

"""
Subsample Extended XYZ File

This script loads an extended XYZ file, subsamples a specified number of configurations
without replacement, and saves the subsampled data to a new file in the same directory.

Usage:
    julia subsample_xyz.jl <input_file> <n_samples> [--seed <seed>]

Arguments:
    input_file  : Path to the input extended XYZ file
    n_samples   : Number of samples to extract (without replacement)
    --seed      : (Optional) Random seed for reproducibility
"""

using Pkg
Pkg.activate(".")

using ExtXYZ
using Random
using Printf
using ACESIDopt: convert_forces_to_svector

# # Parse command line arguments
# function parse_arguments()
#     if length(ARGS) < 2
#         println("Error: Insufficient arguments")
#         println("\nUsage: julia subsample_xyz.jl <input_file> <n_samples> [--seed <seed>]")
#         println("\nArguments:")
#         println("  input_file  : Path to the input extended XYZ file")
#         println("  n_samples   : Number of samples to extract")
#         println("  --seed      : (Optional) Random seed for reproducibility")
#         exit(1)
#     end
    
#     input_file = ARGS[1]
#     n_samples = parse(Int, ARGS[2])
    
#     # Parse optional seed argument
#     seed = nothing
#     if length(ARGS) >= 4 && ARGS[3] == "--seed"
#         seed = parse(Int, ARGS[4])
#     end
    
#     return input_file, n_samples, seed
# end

# Main script
println("="^70)
println("Extended XYZ Subsampling Script")
println("="^70)

# Parse arguments
# input_file, n_samples, seed = parse_arguments()
input_file = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large-c/data/replica_1_samples.xyz"
n_samples = 100
seed = 42
# Check if input file exists
if !isfile(input_file)
    println("Error: Input file not found: $input_file")
    exit(1)
end

# Set random seed if provided
if seed !== nothing
    Random.seed!(seed)
    println("Random seed set to: $seed")
end

# Load data
println("\nLoading data from: $input_file")
data_all = ExtXYZ.load(input_file)
n_total = length(data_all)
println("Total available configurations: $n_total")

# Validate n_samples
if n_samples > n_total
    println("Error: Requested $n_samples samples but only $n_total available")
    exit(1)
end

if n_samples <= 0
    println("Error: Number of samples must be positive")
    exit(1)
end

# Subsample without replacement
println("\nSubsampling $n_samples configurations...")
sampled_indices = sort(Random.randperm(n_total)[1:n_samples])
println("Selected indices: $sampled_indices")

# Extract subsampled data with force conversion
subsampled_data = convert_forces_to_svector.(data_all[sampled_indices])

# Generate output filename
input_dir = dirname(input_file)
input_basename = basename(input_file)
name_without_ext = replace(input_basename, r"\.(xyz|extxyz)$" => "")

# Create output filename with subsample info
if seed !== nothing
    output_basename = "$(name_without_ext)_subsample_$(n_samples)_seed$(seed).xyz"
else
    output_basename = "$(name_without_ext)_subsample_$(n_samples).xyz"
end

output_file = joinpath(input_dir, output_basename)

# Save subsampled data
println("\nSaving subsampled data to: $output_file")
ExtXYZ.save(output_file, subsampled_data)

println("\n" * "="^70)
println("Subsampling completed successfully!")
println("="^70)
println("Input file:  $input_file")
println("Output file: $output_file")
println("Total configurations:     $n_total")
println("Subsampled configurations: $n_samples")
println("Percentage sampled:       $(@sprintf("%.2f", 100 * n_samples / n_total))%")
