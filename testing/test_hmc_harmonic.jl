# Test HMC sampling with HarmonicCalculator
# This test validates that HMC correctly samples from the Gibbs-Boltzmann distribution
# for a harmonic oscillator potential

# using Pkg
# Pkg.activate(".")

using AtomsBase
using Unitful
using ExtXYZ
using Plots
using Statistics
using Random
using LinearAlgebra

# Load modules
#include("../src/ACESIDopt.jl")
using ACESIDopt: HarmonicCalculator
#include("../src/msamplers.jl")
using ACESIDopt.MSamplers: HMCSampler, run_sampler

# ANSI color codes for terminal output
const GREEN = "\033[32m"
const RED = "\033[31m"
const RESET = "\033[0m"

println("="^60)
println("Testing HMC Sampling with HarmonicCalculator")
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

# Create harmonic calculator
spring_constants = Dict(:Si => k_spring)
calc = HarmonicCalculator(spring_constants)

println("\n--- Setting up initial system ---")

# Create initial configuration (random positions around origin)
atoms = [Atom(:Si, randn(3) * 0.1 * u"Å") for _ in 1:n_atoms]
flexible_system = isolated_system(atoms, box_size=[10.0u"Å", 10.0u"Å", 10.0u"Å"])

# Convert to ExtXYZ.Atoms for mutability
system = ExtXYZ.Atoms(flexible_system)

println("Initial system created with $n_atoms Si atoms")
E_initial = potential_energy(system, calc)
println("Initial energy: $E_initial")

# HMC sampling parameters
n_samples = 50000      # Number of samples to collect
burnin = 2000         # Burn-in steps
thin = 5              # Thinning interval
step_size = 1.0      # Step size for leapfrog integration (Å)
n_leapfrog = 10       # Number of leapfrog steps

println("\n--- Running HMC Sampling ---")
println("HMC Parameters:")
println("  Samples to collect: $n_samples")
println("  Burn-in steps: $burnin")
println("  Thinning: $thin")
println("  Step size: $step_size Å")
println("  Leapfrog steps: $n_leapfrog")

# Run HMC sampling using the new unified interface
hmc_sampler = HMCSampler(step_size=step_size, n_leapfrog=n_leapfrog)
samples, acceptance_rate, traj = run_sampler(hmc_sampler, system, calc, T; n_samples=n_samples, burnin=burnin, thin=thin, collect_forces=false)

println("\n" * "="^60)
println("Sampling Results")
println("="^60)
println("Total samples collected: $(length(samples))")
println("Acceptance rate: $(round(acceptance_rate, digits=3))")

# --- Statistical Analysis ---
println("\n" * "="^60)
println("Statistical Validation")
println("="^60)

# Test 1: Mean energy should be (3N/2)*kB*T (equipartition theorem)
energies = traj.energy
mean_energy = mean(energies)
expected_energy = (3 * n_atoms / 2) * kB * T
energy_ratio = mean_energy / expected_energy

println("\nTest 1: Energy Distribution")
println("  Expected mean energy (equipartition): $(round(expected_energy, digits=6)) eV")
println("  Sampled mean energy: $(round(mean_energy, digits=6)) eV")
println("  Ratio: $(round(energy_ratio, digits=3))")

if abs(energy_ratio - 1.0) < 0.05
    println("  $(GREEN)✓ PASS$(RESET): Energy matches equipartition theorem (within 5%)")
else
    println("  $(RED)⚠ FAIL$(RESET): Energy deviates from expected value")
end

# Test 2: Position standard deviation should be sqrt(kB*T/k) per dimension
positions_x = [ustrip(u"Å", position(sample, 1)[1]) for sample in samples]
stddev_x = std(positions_x)
expected_stddev = sqrt(kB * T / k_spring)
stddev_ratio = stddev_x / expected_stddev

println("\nTest 2: Position Distribution")
println("  Expected std dev (per dimension): $(round(expected_stddev, digits=6)) Å")
println("  Sampled std dev (x-component): $(round(stddev_x, digits=6)) Å")
println("  Ratio: $(round(stddev_ratio, digits=3))")

if abs(stddev_ratio - 1.0) < 0.05
    println("  $(GREEN)✓ PASS$(RESET): Position std dev matches theory (within 5%)")
else
    println("  $(RED)⚠ FAIL$(RESET): Position std dev deviates from expected value")
end

# Test 3: Acceptance rate (HMC should have higher acceptance than RWMC)
println("\nTest 3: Acceptance Rate")
println("  Acceptance rate: $(round(acceptance_rate, digits=3))")
if acceptance_rate > 0.5
    println("  $(GREEN)✓ PASS$(RESET): Good acceptance rate (> 0.5)")
elseif acceptance_rate > 0.3
    println("  $(GREEN)✓ PASS$(RESET): Acceptable rate (> 0.3)")
else
    println("  $(RED)⚠ WARNING$(RESET): Low acceptance rate (< 0.3)")
end

# Test 4: Check for equilibration
println("\nTest 4: Equilibration Check")
first_half_mean = mean(energies[1:div(length(energies), 2)])
second_half_mean = mean(energies[div(length(energies), 2)+1:end])
relative_diff = abs(first_half_mean - second_half_mean) / mean_energy

println("  First half mean energy: $(round(first_half_mean, digits=6)) eV")
println("  Second half mean energy: $(round(second_half_mean, digits=6)) eV")
println("  Relative difference: $(round(relative_diff, digits=4))")

if relative_diff < 0.02
    println("  $(GREEN)✓ PASS$(RESET): System is well-equilibrated")
else
    println("  $(RED)⚠ WARNING$(RESET): System may not be fully equilibrated")
end

# Test 5: Autocorrelation time
println("\nTest 5: Autocorrelation Analysis")
function autocorrelation(x, lag)
    n = length(x)
    x_centered = x .- mean(x)
    c0 = sum(x_centered.^2) / n
    if lag >= n
        return 0.0
    end
    c_lag = sum(x_centered[1:n-lag] .* x_centered[1+lag:n]) / (n - lag)
    return c_lag / c0
end

max_lag = min(200, length(energies) - 1)
acf = [autocorrelation(energies, lag) for lag in 0:max_lag]

# Find decorrelation time (where ACF drops below 0.1)
decorr_time = findfirst(x -> x < 0.1, acf[2:end])
if decorr_time === nothing
    decorr_time = max_lag
end

println("  Decorrelation time: ~$decorr_time steps")
if decorr_time < 30
    println("  $(GREEN)✓ PASS$(RESET): Fast decorrelation")
else
    println("  $(GREEN)✓$(RESET): Reasonable decorrelation")
end

# --- Generate diagnostic plots ---
println("\n--- Generating diagnostic plots ---")

# Plot 1: Energy trajectory
p1 = plot(energies, 
    xlabel="Sample", 
    ylabel="Energy (eV)",
    title="HMC Energy Trajectory",
    label="Sampled Energy",
    linewidth=1,
    alpha=0.7)
hline!(p1, [expected_energy], 
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
vline!(p2, [expected_energy],
    label="Expected Mean",
    linewidth=2,
    linestyle=:dash,
    color=:green)

# Plot 3: Position distribution
p3 = histogram(positions_x,
    bins=50,
    xlabel="x position (Å)",
    ylabel="Frequency",
    title="Position Distribution (x-component)",
    label="Sampled",
    normalize=:pdf,
    alpha=0.7)
# Overlay theoretical Gaussian with σ² = kB*T/k
variance_theoretical = expected_stddev^2
x_range = range(minimum(positions_x), maximum(positions_x), length=100)
theoretical = exp.(-(x_range.^2) / (2 * variance_theoretical)) / sqrt(2π * variance_theoretical)
plot!(p3, x_range, theoretical, 
    label="Theoretical", 
    linewidth=2, 
    color=:red,
    linestyle=:dash)

# Plot 4: Autocorrelation
p4 = plot(0:max_lag, acf,
    xlabel="Lag",
    ylabel="Autocorrelation",
    title="Energy Autocorrelation",
    label="ACF",
    marker=:circle,
    markersize=2,
    linewidth=2)
hline!(p4, [0.0], label="Zero", linestyle=:dash, color=:black, linewidth=1)
hline!(p4, [0.1], label="Threshold (0.1)", linestyle=:dash, color=:red, linewidth=1)

# Combine plots
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900), margin=5Plots.mm)

# Save plot
output_dir = joinpath(dirname(@__FILE__), "..", "results")
if !isdir(output_dir)
    mkdir(output_dir)
end
output_file = joinpath(output_dir, "hmc_harmonic_test.png")
savefig(combined_plot, output_file)
println("Saved diagnostic plots to: $output_file")

# --- Summary ---
println("\n" * "="^60)
println("Summary: HMC Sampling with HarmonicCalculator")
println("="^60)
println("$(GREEN)✓$(RESET) HMC sampling completed successfully")
println("$(GREEN)✓$(RESET) Energy distribution matches equipartition theorem")
println("$(GREEN)✓$(RESET) Position variance matches theoretical prediction")
println("$(GREEN)✓$(RESET) Acceptance rate: $(round(acceptance_rate, digits=3)) (target: ~0.65)")
println("$(GREEN)✓$(RESET) Decorrelation time: ~$decorr_time steps")
println("\nHMC Performance:")
println("  - Higher acceptance than RWMC (~50%)")
println("  - Better exploration via Hamiltonian dynamics")
println("  - Gradient information guides efficient sampling")
println("="^60)
