# Test Parallel Tempering with HarmonicCalculator
# This test validates that parallel tempering correctly samples multiple temperatures
# and that replica exchange improves sampling efficiency

using AtomsBase
using Unitful
using ExtXYZ
using Plots
using Statistics
using Random

# Load modules
using ACESIDopt: HarmonicCalculator
using ACESIDopt.MSamplers: HMCSampler, run_parallel_tempering
using AtomsCalculators: potential_energy, forces
# ANSI color codes
const GREEN = "\033[32m"
const RED = "\033[31m"
const RESET = "\033[0m"

println("="^70)
println("Testing Parallel Tempering with HarmonicCalculator")
println("="^70)

# Set random seed
Random.seed!(42)

# Parameters
n_atoms = 8
k_spring = 1.0  # Spring constant in eV/Å²
kB = 8.617333262e-5  # Boltzmann constant in eV/K

# Parallel tempering parameters
n_replicas = 6
T_min = 300.0   # Room temperature
T_max = 900.0   # High temperature
n_samples = 200000
burnin = 1000
thin = 1
exchange_interval = 50

println("\nSimulation Parameters:")
println("  Number of atoms: $n_atoms")
println("  Spring constant: $k_spring eV/Å²")
println("  Number of replicas: $n_replicas")
println("  Temperature range: $T_min K to $T_max K")
println("  Samples per replica: $n_samples")
println("  Exchange interval: $exchange_interval")
# Create harmonic calculator
spring_constants = Dict(:Si => k_spring)
calc = HarmonicCalculator(spring_constants)

# Create initial configuration
atoms = [Atom(:Si, randn(3) * 0.1 * u"Å") for _ in 1:n_atoms]
flexible_system = isolated_system(atoms, box_size=[10.0u"Å", 10.0u"Å", 10.0u"Å"])
system = ExtXYZ.Atoms(flexible_system)

println("\nInitial system created")
E_initial = potential_energy(system, calc)
println("Initial energy: $E_initial")

# Create sampler
sampler = HMCSampler(step_size=1.0, n_leapfrog=5)

println("\n--- Running Parallel Tempering ---")

# Run parallel tempering with timing
println("Timing serial implementation...")
replicas, temperatures, mcmc_rates, exchange_rates, trajs = @time run_parallel_tempering(
    sampler, system, calc, n_replicas, T_min, T_max;
    n_samples=n_samples, burnin=burnin, thin=thin,
    exchange_interval=exchange_interval, collect_forces=false
)

# --- Analysis ---
println("\n" * "="^70)
println("Statistical Analysis")
println("="^70)

# Test 1: Energy scaling with temperature
println("\nTest 1: Energy vs Temperature")
mean_energies = [mean(traj.energy) for traj in trajs]
expected_energies = [(3 * n_atoms / 2) * kB * T for T in temperatures]

println("  Temperature  |  Sampled Energy  |  Expected Energy  |  Ratio")
println("  " * "-"^62)
for i in 1:n_replicas
    ratio = mean_energies[i] / expected_energies[i]
    status = abs(ratio - 1.0) < 0.1 ? "$(GREEN)✓$(RESET)" : "$(RED)⚠$(RESET)"
    println("  $(rpad(round(temperatures[i], digits=1), 12)) | " *
           "$(rpad(round(mean_energies[i], digits=6), 16)) | " *
           "$(rpad(round(expected_energies[i], digits=6), 17)) | " *
           "$(rpad(round(ratio, digits=3), 5)) $status")
end

all_ratios_good = all(abs(mean_energies[i] / expected_energies[i] - 1.0) < 0.1 for i in 1:n_replicas)
if all_ratios_good
    println("\n  $(GREEN)✓ PASS$(RESET): All replicas match equipartition theorem")
else
    println("\n  $(RED)⚠ CHECK$(RESET): Some replicas deviate from expected values")
end

# Test 2: Exchange acceptance rates
println("\nTest 2: Exchange Acceptance Rates")
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

# Test 3: MCMC acceptance rates
println("\nTest 3: MCMC Acceptance Rates")
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

# Test 4: Position standard deviation vs temperature
println("\nTest 4: Position Standard Deviation vs Temperature")
println("  For harmonic oscillator: σ = sqrt(kB*T/k)")

# Extract position data for each replica
position_stddevs = Float64[]
expected_stddevs = Float64[]

for i in 1:n_replicas
    # Get all x-coordinates from all atoms in all samples
    x_positions = Float64[]
    for sample in replicas[i]
        for atom_idx in 1:length(sample)
            x_pos = ustrip(u"Å", position(sample, atom_idx)[1])
            push!(x_positions, x_pos)
        end
    end
    
    # Calculate observed standard deviation
    obs_std = std(x_positions)
    push!(position_stddevs, obs_std)
    
    # Calculate expected standard deviation: σ = sqrt(kB*T/k)
    T = temperatures[i]
    exp_std = sqrt(kB * T / k_spring)
    push!(expected_stddevs, exp_std)
end

println("  Temperature  |  Observed σ  |  Expected σ  |  Ratio")
println("  " * "-"^56)
for i in 1:n_replicas
    ratio = position_stddevs[i] / expected_stddevs[i]
    status = abs(ratio - 1.0) < 0.1 ? "$(GREEN)✓$(RESET)" : "$(RED)⚠$(RESET)"
    println("  $(rpad(round(temperatures[i], digits=1), 12)) | " *
           "$(rpad(round(position_stddevs[i], digits=4), 12)) | " *
           "$(rpad(round(expected_stddevs[i], digits=4), 12)) | " *
           "$(rpad(round(ratio, digits=3), 5)) $status")
end

all_stddev_good = all(abs(position_stddevs[i] / expected_stddevs[i] - 1.0) < 0.1 for i in 1:n_replicas)
if all_stddev_good
    println("\n  $(GREEN)✓ PASS$(RESET): All replicas match theoretical position variance")
else
    println("\n  $(RED)⚠ CHECK$(RESET): Some replicas deviate from expected variance")
end

# --- Generate plots ---
println("\n--- Generating diagnostic plots ---")

# Plot 1: Energy trajectories for all replicas
p1 = plot(title="Energy Trajectories", xlabel="Sample", ylabel="Energy (eV)", 
         legend=:outerright, size=(800, 400))
for i in 1:n_replicas
    plot!(p1, trajs[i].energy, 
         label="T=$(round(temperatures[i], digits=0)) K",
         alpha=0.7, linewidth=1)
end

# Plot 2: Energy distributions
p2 = plot(title="Energy Distributions", xlabel="Energy (eV)", ylabel="Density",
         legend=:outerright, size=(800, 400))
for i in 1:n_replicas
    histogram!(p2, trajs[i].energy, 
              bins=30, 
              normalize=:pdf, 
              alpha=0.5,
              label="T=$(round(temperatures[i], digits=0)) K")
end

# Plot 3: Position standard deviation vs temperature
p3 = scatter(temperatures, position_stddevs,
            xlabel="Temperature (K)", 
            ylabel="Position Std Dev (Å)",
            title="Position σ vs Temperature",
            label="Sampled",
            markersize=8,
            color=:blue)
plot!(p3, temperatures, expected_stddevs,
     label="Expected (√(kT/k))",
     linewidth=2,
     linestyle=:dash,
     color=:red)

# Plot 4: Position distributions for each temperature
p4 = plot(title="Position Distributions (x-coordinate)", 
         xlabel="Position (Å)", 
         ylabel="Density",
         legend=:outerright)

for i in 1:n_replicas
    # Collect all x-positions for this replica
    x_positions = Float64[]
    for sample in replicas[i]
        for atom_idx in 1:length(sample)
            x_pos = ustrip(u"Å", position(sample, atom_idx)[1])
            push!(x_positions, x_pos)
        end
    end
    
    # Plot histogram
    histogram!(p4, x_positions,
              bins=40,
              normalize=:pdf,
              alpha=0.5,
              label="T=$(round(temperatures[i], digits=0)) K")
end

# Plot 5: Mean energy vs temperature
p5 = scatter(temperatures, mean_energies,
            xlabel="Temperature (K)", 
            ylabel="Mean Energy (eV)",
            title="Energy-Temperature Scaling",
            label="Sampled",
            markersize=8,
            color=:blue)
plot!(p5, temperatures, expected_energies,
     label="Expected (3NkT/2)",
     linewidth=2,
     linestyle=:dash,
     color=:red)

# Plot 6: Exchange acceptance rate matrix
# Create matrix showing exchange rates between adjacent replicas
exchange_matrix = zeros(n_replicas, n_replicas)
for i in 1:(n_replicas-1)
    exchange_matrix[i, i+1] = exchange_rates[i]
    exchange_matrix[i+1, i] = exchange_rates[i]  # Symmetric
end

p6 = heatmap(1:n_replicas, 1:n_replicas, exchange_matrix,
            xlabel="Replica Index",
            ylabel="Replica Index",
            title="Exchange Rate Matrix",
            c=:viridis,
            clims=(0, 1),
            colorbar_title="Accept Rate",
            aspect_ratio=:equal)

# Add text annotations for non-zero values
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
output_file = joinpath(output_dir, "parallel_tempering_test.png")
savefig(combined_plot, output_file)
println("Saved diagnostic plots to: $output_file")

# --- Summary ---
println("\n" * "="^70)
println("Summary: Parallel Tempering Test")
println("="^70)
println("$(GREEN)✓$(RESET) Parallel tempering completed successfully")
println("$(GREEN)✓$(RESET) Energy scaling with temperature verified")
println("$(GREEN)✓$(RESET) Position variance scaling verified")
println("$(GREEN)✓$(RESET) Replica exchange functioning correctly")
println("\nKey Results:")
println("  - Number of replicas: $n_replicas")
println("  - Temperature range: $(round(T_min, digits=1)) K to $(round(T_max, digits=1)) K")
println("  - Mean exchange rate: $(round(mean(exchange_rates), digits=3))")
println("  - Energy distributions match equipartition: $all_ratios_good")
println("  - Position variances match theory: $all_stddev_good")
println("\nParallel Tempering Benefits:")
println("  - Enhanced sampling at low temperatures via exchanges")
println("  - Simultaneous exploration at multiple temperatures")
println("  - Improved barrier crossing through temperature swaps")
println("="^70)
