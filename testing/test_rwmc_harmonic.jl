"""
Test RWMC sampling with HarmonicCalculator

This test applies Random Walk Monte Carlo (RWMC) sampling to sample the 
Gibbs-Boltzmann distribution for a system of 16 Si atoms with a harmonic 
potential centered at the origin and non-periodic boundary conditions.

The test verifies that:
1. RWMC converges to the correct distribution
2. Average energy matches theoretical expectation
3. Position distributions match the expected thermal distribution
"""

using AtomsBase
using AtomsBase: isolated_system, Atom
using ExtXYZ
using Unitful
using LinearAlgebra
using StaticArrays
using Statistics
using Random
using Plots

# Load the required modules
using ACESIDopt: HarmonicCalculator, MSamplers
using ACESIDopt.MSamplers: run_rwmc_sampling, RWMCSampler, run_sampler
using AtomsCalculators: potential_energy, forces

# ANSI color codes
const GREEN = "\033[32m"
const RED = "\033[31m"
const RESET = "\033[0m"

println("="^60)
println("Testing RWMC Sampling with HarmonicCalculator")
println("="^60)

# Set random seed for reproducibility
Random.seed!(42)

# Parameters
n_atoms = 16
T = 300.0  # Temperature in Kelvin (room temperature)
k_spring = 1.0  # Spring constant in eV/Å²
kB = 8.617333262e-5  # Boltzmann constant in eV/K

println("\nSimulation Parameters:")
println("  Number of atoms: $n_atoms")
println("  Temperature: $T K")
println("  Spring constant: $k_spring eV/Å²")
println("  Boltzmann constant: $kB eV/K")

# Create initial system with 16 Si atoms
# Start from random positions near origin
# Use ExtXYZ.Atoms for mutability (required by RWMC)
println("\n--- Creating Initial System ---")

# First create a FlexibleSystem, then convert to ExtXYZ.Atoms
initial_positions = []
for i in 1:n_atoms
    # Random positions in a small cube around origin
    pos = randn(3) * 0.5 * u"Å"
    push!(initial_positions, Atom(:Si, pos))
end

flexible_system = isolated_system(initial_positions)

# Convert to ExtXYZ.Atoms for mutability
system = ExtXYZ.Atoms(flexible_system)

println("Created system with $n_atoms Si atoms")
println("System type: $(typeof(system))")

# Create HarmonicCalculator
calc = HarmonicCalculator(:Si, k_spring)
println("Created HarmonicCalculator with k = $k_spring eV/Å²")

# Test initial energy
E_initial = potential_energy(system, calc)
println("Initial energy: $E_initial")

# RWMC sampling parameters
n_samples = 50000      # Number of samples to collect
burnin = 2000         # Burn-in steps
thin = 5              # Thinning interval
step_size = 0.05       # Step size for random walk (Å)

println("\n--- Running RWMC Sampling ---")
println("RWMC Parameters:")
println("  Samples to collect: $n_samples")
println("  Burn-in steps: $burnin")
println("  Thinning: $thin")
println("  Step size: $step_size Å")

# Run RWMC sampling using the new unified interface
rwmc_sampler = RWMCSampler(step_size=step_size)
samples, acceptance_rate, traj = run_sampler(rwmc_sampler, system, calc, T; n_samples=n_samples, burnin=burnin, thin=thin)

# Alternative: use the old interface (still supported for backward compatibility)
# samples, acceptance_rate, traj = run_rwmc_sampling(
#     system, calc, n_samples, T;
#     step_size=step_size,
#     burnin=burnin,
#     thin=thin
# )

println("\n--- Sampling Results ---")
println("Collected $(length(samples)) samples")
println("Acceptance rate: $(round(acceptance_rate, digits=3))")

# Analyze the results
println("\n--- Statistical Analysis ---")

# 1. Energy analysis
energies = traj.energy
mean_energy = mean(energies)
std_energy = std(energies)

println("\nEnergy Statistics:")
println("  Mean energy: $(round(mean_energy, digits=6)) eV")
println("  Std energy: $(round(std_energy, digits=6)) eV")

# For harmonic oscillator in 3D, each atom has 3 degrees of freedom
# Expected mean energy per atom: E = (3/2) * kB * T for kinetic energy
# For potential energy in canonical ensemble: <U> = (3*N/2) * kB * T
# Total equipartition: <U> = (3*N/2) * kB * T
n_dof = 3 * n_atoms
expected_mean_energy = 0.5 * n_dof * kB * T

println("\nTheoretical Expectations:")
println("  Degrees of freedom: $n_dof")
println("  Expected mean energy (equipartition): $(round(expected_mean_energy, digits=6)) eV")
println("  Expected energy per atom: $(round(expected_mean_energy/n_atoms, digits=6)) eV")

energy_ratio = mean_energy / expected_mean_energy
println("\nEnergy Ratio (sampled/expected): $(round(energy_ratio, digits=3))")

if abs(energy_ratio - 1.0) < 0.15  # Within 15% is reasonable
    println("$(GREEN)✓$(RESET) Mean energy is consistent with equipartition theorem")
else
    println("$(RED)⚠$(RESET) Mean energy deviates from expected (ratio: $(round(energy_ratio, digits=3)))")
    println("  This might indicate insufficient sampling or equilibration")
end

# 2. Position distribution analysis
println("\n--- Position Distribution Analysis ---")

# Extract positions from samples
all_positions = []
for sample in samples
    for i in 1:length(sample)
        pos = position(sample, i)
        pos_vec = ustrip.(u"Å", pos)
        push!(all_positions, pos_vec)
    end
end

# Calculate position statistics
x_coords = [pos[1] for pos in all_positions]
y_coords = [pos[2] for pos in all_positions]
z_coords = [pos[3] for pos in all_positions]

mean_x = mean(x_coords)
mean_y = mean(y_coords)
mean_z = mean(z_coords)
std_x = std(x_coords)
std_y = std(y_coords)
std_z = std(z_coords)

println("Position Statistics (all atoms, all samples):")
println("  Mean x: $(round(mean_x, digits=4)) Å")
println("  Mean y: $(round(mean_y, digits=4)) Å")
println("  Mean z: $(round(mean_z, digits=4)) Å")
println("  Std x: $(round(std_x, digits=4)) Å")
println("  Std y: $(round(std_y, digits=4)) Å")
println("  Std z: $(round(std_z, digits=4)) Å")

# For harmonic oscillator: σ² = kB*T/k
expected_std = sqrt(kB * T / k_spring)
println("\nTheoretical position variance:")
println("  Expected std (σ = √(kB*T/k)): $(round(expected_std, digits=4)) Å")

avg_std = (std_x + std_y + std_z) / 3
std_ratio = avg_std / expected_std
println("  Observed average std: $(round(avg_std, digits=4)) Å")
println("  Ratio (observed/expected): $(round(std_ratio, digits=3))")

if abs(std_ratio - 1.0) < 0.15  # Within 15%
    println("$(GREEN)✓$(RESET) Position variance is consistent with thermal distribution")
else
    println("$(RED)⚠$(RESET) Position variance deviates from expected")
end

# 3. Check for equilibration
println("\n--- Equilibration Check ---")
# Divide trajectory into first and second half
mid_point = div(length(energies), 2)
first_half_mean = mean(energies[1:mid_point])
second_half_mean = mean(energies[mid_point+1:end])

println("First half mean energy: $(round(first_half_mean, digits=6)) eV")
println("Second half mean energy: $(round(second_half_mean, digits=6)) eV")
println("Difference: $(round(abs(first_half_mean - second_half_mean), digits=6)) eV")

drift = abs(first_half_mean - second_half_mean) / mean_energy
if drift < 0.05  # Less than 5% drift
    println("$(GREEN)✓$(RESET) System appears well-equilibrated (drift < 5%)")
else
    println("$(RED)⚠$(RESET) System may not be fully equilibrated (drift: $(round(drift*100, digits=2))%)")
end

# 4. Autocorrelation analysis
println("\n--- Autocorrelation Analysis ---")
function autocorr(x, lag)
    n = length(x)
    if lag >= n
        return 0.0
    end
    x_mean = mean(x)
    c0 = sum((x .- x_mean).^2) / n
    c_lag = sum((x[1:n-lag] .- x_mean) .* (x[lag+1:n] .- x_mean)) / (n - lag)
    return c_lag / c0
end

max_lag = min(100, div(length(energies), 2))
lags = 1:10:max_lag
autocorrs = [autocorr(energies, lag) for lag in lags]

println("Energy autocorrelation at selected lags:")
for (lag, ac) in zip(lags, autocorrs)
    println("  Lag $lag: $(round(ac, digits=3))")
end

# Find decorrelation time (where autocorr drops below 1/e ≈ 0.368)
decorr_threshold = 0.368
decorr_time = findfirst(ac -> ac < decorr_threshold, autocorrs)
if decorr_time !== nothing
    println("Decorrelation time: ~$(lags[decorr_time]) steps")
else
    println("Decorrelation time: >$max_lag steps")
end

# 5. Create plots
println("\n--- Creating Diagnostic Plots ---")

# Plot 1: Energy trajectory
p1 = plot(energies, 
    xlabel="Sample", 
    ylabel="Energy (eV)",
    title="RWMC Energy Trajectory",
    label="Sampled Energy",
    linewidth=1,
    alpha=0.7)
hline!(p1, [expected_mean_energy], 
    label="Expected Mean",
    linewidth=2,
    linestyle=:dash,
    color=:red)

# Plot 2: Energy histogram
p2 = histogram(energies,
    bins=50,
    xlabel="Energy (eV)",
    ylabel="Frequency",
    title="Energy Distribution",
    label="Sampled",
    normalize=:pdf,
    alpha=0.7)
vline!(p2, [mean_energy],
    label="Sample Mean",
    linewidth=2,
    color=:red)
vline!(p2, [expected_mean_energy],
    label="Expected Mean",
    linewidth=2,
    linestyle=:dash,
    color=:green)

# Plot 3: Position distributions
p3 = histogram(x_coords,
    bins=50,
    xlabel="x position (Å)",
    ylabel="Frequency",
    title="Position Distribution (x-coordinate)",
    label="Sampled",
    normalize=:pdf,
    alpha=0.7)

# Plot 4: Autocorrelation
p4 = plot(collect(lags), autocorrs,
    xlabel="Lag",
    ylabel="Autocorrelation",
    title="Energy Autocorrelation",
    label="Autocorrelation",
    marker=:circle,
    linewidth=2)
hline!(p4, [0.0], label="Zero", linestyle=:dash, color=:black, linewidth=1)
hline!(p4, [decorr_threshold], label="1/e threshold", linestyle=:dash, color=:red, linewidth=1)

# Combine plots
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))

# Save plot
output_dir = joinpath(dirname(@__FILE__), "..", "results")
if !isdir(output_dir)
    mkdir(output_dir)
end
output_file = joinpath(output_dir, "rwmc_harmonic_test.png")
savefig(combined_plot, output_file)
println("Saved diagnostic plots to: $output_file")

# Summary
println("\n" * "="^60)
println("RWMC Sampling Test Summary")
println("="^60)
println("Acceptance rate: $(round(acceptance_rate*100, digits=1))%")
println("Energy convergence: $(abs(energy_ratio - 1.0) < 0.15 ? "$(GREEN)✓ PASS$(RESET)" : "$(RED)⚠ CHECK$(RESET)")")
println("Position variance: $(abs(std_ratio - 1.0) < 0.15 ? "$(GREEN)✓ PASS$(RESET)" : "$(RED)⚠ CHECK$(RESET)")")
println("Equilibration: $(drift < 0.05 ? "$(GREEN)✓ PASS$(RESET)" : "$(RED)⚠ CHECK$(RESET)")")
println("="^60)

# Overall assessment
if acceptance_rate > 0.2 && acceptance_rate < 0.7 &&
   abs(energy_ratio - 1.0) < 0.15 &&
   abs(std_ratio - 1.0) < 0.15 &&
   drift < 0.05
    println("\n$(GREEN)✓✓✓ All tests PASSED! ✓✓✓$(RESET)")
    println("RWMC sampling successfully samples the Gibbs-Boltzmann distribution")
else
    println("\n$(RED)⚠ Some tests did not pass optimally$(RESET)")
    println("Consider adjusting sampling parameters (step_size, burnin, n_samples)")
end

println("\n" * "="^60)
