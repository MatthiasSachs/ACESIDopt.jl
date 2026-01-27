#=
Compare Active Learning Methods
Load error.yaml files from different experiments and plot comparisons
=#

using YAML
using Plots
using Printf

# Base directory for experiments
const RESULTS_DIR = joinpath(@__DIR__, "results")

# Experiment names to compare
const EXPERIMENTS = [
    "AL_ABSID_BLR_20it-ACE-320K6-K1200-random-start",
    "AL_HAL_BLR_20it-ACE-320K6-K1200-random-start",
    "AL_US_BLR_20it-ACE-320K6-K1200-random-start"
]

# Method labels for plots
const METHOD_LABELS = [
    "ABSID",
    "HAL",
    "US"
]

# Colors for each method
const METHOD_COLORS = [:blue, :red, :green]

# Markers for each method
const METHOD_MARKERS = [:circle, :square, :diamond]

println("="^70)
println("Comparing Active Learning Methods")
println("="^70)

# Load errors from all experiments
errors_data = Dict()
for (exp_name, label) in zip(EXPERIMENTS, METHOD_LABELS)
    errors_file = joinpath(RESULTS_DIR, exp_name, "errors.yaml")
    
    if !isfile(errors_file)
        @warn "Error file not found: $errors_file"
        continue
    end
    
    println("Loading errors from: $exp_name")
    errors_data[label] = YAML.load_file(errors_file)
end

if isempty(errors_data)
    error("No error files found. Please check the experiment names and paths.")
end

# Create output directory for comparison plots
comparison_dir = joinpath(RESULTS_DIR, "method_comparison")
if !isdir(comparison_dir)
    mkpath(comparison_dir)
    println("Created comparison directory: $comparison_dir")
end

# Extract large test set errors for all methods
println("\nGenerating comparison plots...")

# Plot 1: Energy RMSE on Large Test Set
p1 = plot(xlabel="Active Learning Iteration",
          ylabel="Energy RMSE (eV)",
          title="Energy RMSE - Large Test Set",
          yscale=:log10,
          legend=:topright,
          legendfontsize=10,
          linewidth=2,
          markersize=5,
          grid=true)

for (i, label) in enumerate(METHOD_LABELS)
    if haskey(errors_data, label)
        iterations = errors_data[label]["iterations"]
        energy_rmse = errors_data[label]["large_test"]["energy_rmse"]
        plot!(p1, iterations, energy_rmse,
              label=label,
              marker=METHOD_MARKERS[i],
              color=METHOD_COLORS[i])
    end
end

energy_rmse_filename = joinpath(comparison_dir, "comparison_energy_rmse_large_test.png")
savefig(p1, energy_rmse_filename)
println("Saved Energy RMSE comparison: $energy_rmse_filename")

# Plot 2: Energy MAE on Large Test Set
p2 = plot(xlabel="Active Learning Iteration",
          ylabel="Energy MAE (eV)",
          title="Energy MAE - Large Test Set",
          yscale=:log10,
          legend=:topright,
          legendfontsize=10,
          linewidth=2,
          markersize=5,
          grid=true)

for (i, label) in enumerate(METHOD_LABELS)
    if haskey(errors_data, label)
        iterations = errors_data[label]["iterations"]
        energy_mae = errors_data[label]["large_test"]["energy_mae"]
        plot!(p2, iterations, energy_mae,
              label=label,
              marker=METHOD_MARKERS[i],
              color=METHOD_COLORS[i])
    end
end

energy_mae_filename = joinpath(comparison_dir, "comparison_energy_mae_large_test.png")
savefig(p2, energy_mae_filename)
println("Saved Energy MAE comparison: $energy_mae_filename")

# Plot 3: Forces RMSE on Large Test Set
p3 = plot(xlabel="Active Learning Iteration",
          ylabel="Forces RMSE (eV/Å)",
          title="Forces RMSE - Large Test Set",
          yscale=:log10,
          legend=:topright,
          legendfontsize=10,
          linewidth=2,
          markersize=5,
          grid=true)

for (i, label) in enumerate(METHOD_LABELS)
    if haskey(errors_data, label)
        iterations = errors_data[label]["iterations"]
        forces_rmse = errors_data[label]["large_test"]["forces_rmse"]
        plot!(p3, iterations, forces_rmse,
              label=label,
              marker=METHOD_MARKERS[i],
              color=METHOD_COLORS[i])
    end
end

forces_rmse_filename = joinpath(comparison_dir, "comparison_forces_rmse_large_test.png")
savefig(p3, forces_rmse_filename)
println("Saved Forces RMSE comparison: $forces_rmse_filename")

# Plot 4: Forces MAE on Large Test Set
p4 = plot(xlabel="Active Learning Iteration",
          ylabel="Forces MAE (eV/Å)",
          title="Forces MAE - Large Test Set",
          yscale=:log10,
          legend=:topright,
          legendfontsize=10,
          linewidth=2,
          markersize=5,
          grid=true)

for (i, label) in enumerate(METHOD_LABELS)
    if haskey(errors_data, label)
        iterations = errors_data[label]["iterations"]
        forces_mae = errors_data[label]["large_test"]["forces_mae"]
        plot!(p4, iterations, forces_mae,
              label=label,
              marker=METHOD_MARKERS[i],
              color=METHOD_COLORS[i])
    end
end

forces_mae_filename = joinpath(comparison_dir, "comparison_forces_mae_large_test.png")
savefig(p4, forces_mae_filename)
println("Saved Forces MAE comparison: $forces_mae_filename")

# Create a combined 2x2 plot
p_combined = plot(p1, p2, p3, p4,
                  layout=(2, 2),
                  size=(1200, 1000),
                  plot_title="Active Learning Method Comparison - Large Test Set",
                  plot_titlefontsize=14)

combined_filename = joinpath(comparison_dir, "comparison_all_metrics_large_test.png")
savefig(p_combined, combined_filename)
println("Saved combined comparison: $combined_filename")

# Print final error values for comparison
println("\n" * "="^70)
println("Final Error Values (Last Iteration)")
println("="^70)

for label in METHOD_LABELS
    if haskey(errors_data, label)
        n_iters = length(errors_data[label]["iterations"])
        println("\n$label (Iteration $n_iters):")
        @printf("  Energy RMSE: %.6f eV\n", errors_data[label]["large_test"]["energy_rmse"][end])
        @printf("  Energy MAE:  %.6f eV\n", errors_data[label]["large_test"]["energy_mae"][end])
        @printf("  Forces RMSE: %.6f eV/Å\n", errors_data[label]["large_test"]["forces_rmse"][end])
        @printf("  Forces MAE:  %.6f eV/Å\n", errors_data[label]["large_test"]["forces_mae"][end])
    end
end

println("\n" * "="^70)
println("Method Comparison Complete!")
println("Plots saved to: $comparison_dir")
println("="^70)
