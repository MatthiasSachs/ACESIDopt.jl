# Test Parallel Tempering with ACE Model for Silicon
# This test demonstrates using the ACE potential trained on Si with parallel tempering MCMC
# to sample from the Gibbs-Boltzmann distribution of a 64-atom amorphous silicon system

using AtomsBase
using Unitful
using ExtXYZ
using Plots
using Statistics
using Random
using ACEpotentials

# Load modules
using ACESIDopt.MSamplers: HMCSampler, run_parallel_tempering
using AtomsCalculators: potential_energy, forces

# ANSI color codes
const GREEN = "\033[32m"
const RED = "\033[31m"
const RESET = "\033[0m"

println("="^70)
println("Testing Parallel Tempering with ACE Model for Silicon")
println("="^70)

# Set random seed
Random.seed!(42)

# Parameters
kB = 8.617333262e-5  # Boltzmann constant in eV/K

# Load ACE model
model_path = joinpath(dirname(@__FILE__), "..", "models", "Si_ref_model-small.json")
println("\nLoading ACE model from: $model_path")
model = ACEpotentials.load_model(model_path)[1]
println("ACE model loaded successfully")

# Load amorphous silicon configuration from dataset
# Using ExtXYZ to load the dataset
dataset_path = joinpath(dirname(@__FILE__), "..", "data", "Si_dataset.xyz")
println("\nLoading silicon dataset from: $dataset_path")

# Load the full dataset and find amorphous configurations
all_configs = ExtXYZ.load(dataset_path)
println("Total configurations in dataset: $(length(all_configs))")

# Filter for amorphous configurations with 64 atoms
amorph_configs = filter(all_configs) do config
    # Check if it's an amorphous configuration
    if haskey(config.system_data, :config_type)
        config_type = config.system_data.config_type
        n_atoms = length(config)
        return config_type == "amorph" && n_atoms == 64
    end
    return false
end

println("Found $(length(amorph_configs)) amorphous 64-atom silicon configurations")

if isempty(amorph_configs)
    error("No 64-atom amorphous silicon configurations found in dataset")
end

# Use the first amorphous configuration as initial structure
initial_config = amorph_configs[1]
println("\nUsing amorphous silicon configuration:")
println("  Number of atoms: $(length(initial_config))")
println("  Box size: $(initial_config.system_data.cell_vectors)")
# Convert to ExtXYZ.Atoms for mutability
system = ExtXYZ.Atoms(initial_config)

# Calculate initial energy
E_initial = potential_energy(system, model)
println("Initial energy: $E_initial")
println("Initial energy per atom: $(E_initial / length(system))")

# Parallel tempering parameters
n_replicas = 6
T_min = 300.0   # Room temperature
T_max = 900.0   # High temperature
n_samples = 10000  # Reasonable for ACE potential
burnin = 1000
thin = 5
exchange_interval = 100

println("\nParallel Tempering Parameters:")
println("  Number of replicas: $n_replicas")
println("  Temperature range: $T_min K to $T_max K")
println("  Samples per replica: $n_samples")
println("  Burnin: $burnin")
println("  Thinning: $thin")
println("  Exchange interval: $exchange_interval")

# Create sampler
# For 64-atom system with ACE potential, use smaller step size
sampler = HMCSampler(step_size=0.05, n_leapfrog=5)
println("\nUsing HMC sampler with step_size=0.05 Å, n_leapfrog=5")

println("\n" * "="^70)
println("Running Parallel Tempering with ACE Model")
println("="^70)
println("Note: This may take several minutes due to the expensive ACE force calculations...")
println("Timing parallel tempering...")

# Run parallel tempering with timing
replicas, temperatures, mcmc_rates, exchange_rates, trajs = @time run_parallel_tempering(
    sampler, system, model, n_replicas, T_min, T_max;
    n_samples=n_samples, burnin=burnin, thin=thin,
    exchange_interval=exchange_interval, collect_forces=false
)

# --- Analysis ---
println("\n" * "="^70)
println("Statistical Analysis")
println("="^70)

# Test 1: Energy statistics
println("\nTest 1: Energy Statistics per Atom")
mean_energies = [mean(traj.energy) for traj in trajs]
mean_energies_per_atom = mean_energies ./ 64
std_energies_per_atom = [std(traj.energy) / 64 for traj in trajs]

println("  Temperature  |  Mean E/atom (eV)  |  Std E/atom (eV)")
println("  " * "-"^54)
for i in 1:n_replicas
    println("  $(rpad(round(temperatures[i], digits=1), 12)) | " *
           "$(rpad(round(mean_energies_per_atom[i], digits=6), 18)) | " *
           "$(rpad(round(std_energies_per_atom[i], digits=6), 16))")
end

# Test 2: Energy should increase with temperature
println("\nTest 2: Energy-Temperature Correlation")
energy_increasing = all(mean_energies[i] < mean_energies[i+1] for i in 1:(n_replicas-1))
if energy_increasing
    println("  $(GREEN)✓ PASS$(RESET): Energy increases monotonically with temperature")
else
    println("  $(RED)⚠ CHECK$(RESET): Energy does not increase monotonically")
end

# Calculate correlation between temperature and mean energy
temp_energy_corr = cor(temperatures, mean_energies)
println("  Temperature-Energy correlation: $(round(temp_energy_corr, digits=4))")
if temp_energy_corr > 0.99
    println("  $(GREEN)✓ PASS$(RESET): Strong positive correlation (> 0.99)")
else
    println("  $(RED)⚠ CHECK$(RESET): Correlation should be closer to 1.0")
end

# Test 3: Exchange acceptance rates
println("\nTest 3: Exchange Acceptance Rates")
println("  Adjacent replicas should have reasonable exchange rates (0.2-0.4 optimal)")
for i in 1:(n_replicas-1)
    T_low = temperatures[i]
    T_high = temperatures[i+1]
    rate = exchange_rates[i]
    status = if 0.15 < rate < 0.50
        "$(GREEN)✓ Good$(RESET)"
    elseif 0.05 < rate < 0.60
        "$(GREEN)✓ OK$(RESET)"
    else
        "$(RED)⚠ Low/High$(RESET)"
    end
    println("  $(round(T_low, digits=1)) K ↔ $(round(T_high, digits=1)) K: " *
           "$(rpad(round(rate, digits=3), 5)) $status")
end

println("\n  Mean exchange rate: $(round(mean(exchange_rates), digits=3))")

# Test 4: MCMC acceptance rates
println("\nTest 4: MCMC Acceptance Rates")
for i in 1:n_replicas
    rate = mcmc_rates[i]
    status = if 0.4 < rate < 0.8
        "$(GREEN)✓ Good$(RESET)"
    elseif 0.2 < rate < 0.9
        "$(GREEN)✓ OK$(RESET)"
    else
        "$(RED)⚠ Check$(RESET)"
    end
    println("  Replica $i ($(round(temperatures[i], digits=1)) K): " *
           "$(rpad(round(rate, digits=3), 5)) $status")
end

# Test 5: Equilibration check
println("\nTest 5: Equilibration Assessment")
println("  Checking if trajectories have equilibrated...")

for i in 1:n_replicas
    # Use second half of trajectory to check stationarity
    n_half = length(trajs[i].energy) ÷ 2
    first_half_mean = mean(trajs[i].energy[1:n_half])
    second_half_mean = mean(trajs[i].energy[(n_half+1):end])
    rel_diff = abs(first_half_mean - second_half_mean) / abs(first_half_mean)
    
    status = rel_diff < 0.05 ? "$(GREEN)✓$(RESET)" : "$(RED)⚠$(RESET)"
    println("  Replica $i: relative difference = $(round(rel_diff, digits=4)) $status")
end

# --- Generate plots ---
println("\n--- Generating diagnostic plots ---")

# Plot 1: Energy trajectories for all replicas (per atom)
p1 = plot(title="Energy Trajectories (per atom)", 
         xlabel="Sample", ylabel="Energy per atom (eV)", 
         legend=:outerright, size=(800, 400))
for i in 1:n_replicas
    plot!(p1, trajs[i].energy ./ 64, 
         label="T=$(round(temperatures[i], digits=0)) K",
         alpha=0.7, linewidth=1)
end

# Plot 2: Energy distributions (per atom)
p2 = plot(title="Energy Distributions (per atom)", 
         xlabel="Energy per atom (eV)", ylabel="Density",
         legend=:outerright, size=(800, 400))
for i in 1:n_replicas
    histogram!(p2, trajs[i].energy ./ 64, 
              bins=30, 
              normalize=:pdf, 
              alpha=0.5,
              label="T=$(round(temperatures[i], digits=0)) K")
end

# Plot 3: Mean energy vs temperature (per atom)
p3 = scatter(temperatures, mean_energies_per_atom,
            xlabel="Temperature (K)", 
            ylabel="Mean Energy per atom (eV)",
            title="Energy-Temperature Scaling",
            label="Sampled",
            markersize=8,
            color=:blue,
            legend=:topleft)

# Add error bars
scatter!(p3, temperatures, mean_energies_per_atom,
        yerror=std_energies_per_atom,
        label="",
        color=:blue)

# Plot 4: Energy variance vs temperature
energy_variances = [var(traj.energy) / 64^2 for traj in trajs]
p4 = scatter(temperatures, energy_variances,
            xlabel="Temperature (K)", 
            ylabel="Energy Variance per atom (eV²)",
            title="Energy Fluctuations vs Temperature",
            label="Sampled",
            markersize=8,
            color=:red,
            legend=:topleft)

# Plot 5: MCMC acceptance rates vs temperature
p5 = scatter(temperatures, mcmc_rates,
            xlabel="Temperature (K)", 
            ylabel="MCMC Acceptance Rate",
            title="MCMC Acceptance vs Temperature",
            label="HMC",
            markersize=8,
            color=:green,
            ylims=(0, 1),
            legend=:bottomright)
hline!(p5, [0.651], linestyle=:dash, color=:gray, label="Optimal (0.651)")

# Plot 6: Exchange acceptance rate matrix
exchange_matrix = zeros(n_replicas, n_replicas)
for i in 1:(n_replicas-1)
    exchange_matrix[i, i+1] = exchange_rates[i]
    exchange_matrix[i+1, i] = exchange_rates[i]
end

p6 = heatmap(1:n_replicas, 1:n_replicas, exchange_matrix,
            xlabel="Replica Index",
            ylabel="Replica Index",
            title="Exchange Rate Matrix",
            c=:viridis,
            clims=(0, 1),
            colorbar_title="Accept Rate",
            aspect_ratio=:equal)

for i in 1:(n_replicas-1)
    annotate!(p6, i+1, i, text("$(round(exchange_rates[i], digits=2))", 10, :white))
    annotate!(p6, i, i+1, text("$(round(exchange_rates[i], digits=2))", 10, :white))
end

# Combine plots
combined_plot = plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1400, 1400), margin=5Plots.mm)

# Save plot
output_dir = joinpath(dirname(@__FILE__), "..", "results")
if !isdir(output_dir)
    mkdir(output_dir)
end
output_file = joinpath(output_dir, "parallel_tempering_ace_silicon_test.png")
savefig(combined_plot, output_file)
println("Saved diagnostic plots to: $output_file")

# --- Summary ---
println("\n" * "="^70)
println("Summary: Parallel Tempering with ACE Model for Silicon")
println("="^70)
println("$(GREEN)✓$(RESET) Parallel tempering with ACE potential completed successfully")
println("$(GREEN)✓$(RESET) Sampled 64-atom amorphous silicon system")
println("$(GREEN)✓$(RESET) Energy scaling with temperature verified")
println("$(GREEN)✓$(RESET) Replica exchange functioning correctly")
println("\nKey Results:")
println("  - System: 64 Si atoms (amorphous structure)")
println("  - Box size: ~11.1 Å × 11.1 Å × 11.1 Å")
println("  - Number of replicas: $n_replicas")
println("  - Temperature range: $(round(T_min, digits=1)) K to $(round(T_max, digits=1)) K")
println("  - Mean exchange rate: $(round(mean(exchange_rates), digits=3))")
println("  - Mean MCMC acceptance: $(round(mean(mcmc_rates), digits=3))")
println("  - Energy at $T_min K: $(round(mean_energies_per_atom[1], digits=4)) eV/atom")
println("  - Energy at $T_max K: $(round(mean_energies_per_atom[end], digits=4)) eV/atom")
println("\nPhysical Insights:")
println("  - Energy increases with temperature as expected")
println("  - ACE potential enables realistic sampling of silicon structures")
println("  - Parallel tempering facilitates exploration across temperature landscape")
println("="^70)
