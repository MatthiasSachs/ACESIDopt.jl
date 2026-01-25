# Test Distributed Parallel Tempering with ACE Model for Silicon
# This test demonstrates using the ACE potential trained on Si with distributed replica exchange MCMC
# to sample from the Gibbs-Boltzmann distribution at multiple temperatures of a 64-atom amorphous silicon system
# Extended version: computes and saves forces and energies for all samples
using ProgressMeter
using Distributed
using ACESIDopt: add_energy_forces

Random.seed!(43)
# Add worker processes if not already added
if nworkers() == 1
    println("Adding 4 worker processes...")
    addprocs(4)
    println("Workers available: $(nworkers())")
end

# Load required packages on all workers
@everywhere begin
    using AtomsBase
    using AtomsBase: FlexibleSystem
    using Unitful
    using ExtXYZ
    using Statistics
    using Random
    using ACEpotentials
    using ACESIDopt.MSamplers: RWMCSampler, run_parallel_tempering_distributed
    using AtomsCalculators: potential_energy, forces
end

using Plots

# ANSI color codes (on main process)
const GREEN = "\033[32m"
const RED = "\033[31m"
const RESET = "\033[0m"

println("="^70)
println("Testing Distributed Parallel Tempering with ACE Model for Silicon")
println("="^70)

# Set random seed
Random.seed!(42)

# Parameters
kB = 8.617333262e-5  # Boltzmann constant in eV/K

# Load ACE model on main process
model_path = joinpath(dirname(@__FILE__), "..", "models", "Si_ref_model-small.json")
println("\nLoading ACE model from: $model_path")
model = ACEpotentials.load_model(model_path)[1]
println("ACE model loaded successfully")

# Load model on all workers
@everywhere model_path = $model_path
@everywhere model = ACEpotentials.load_model(model_path)[1]

# Load configurations from trajectory
raw_data = Dict{String, Any}()

data_path = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-2atom.xyz")

# Parallel tempering parameters
n_replicas = 4

println("\nLoading silicon configurations from: $data_path")
raw_data["init"] = [ExtXYZ.load(data_path) for i in 1:n_replicas]
println("Loaded $(length(raw_data["init"])) configurations")


# Randomly sample n_replicas initial configurations
println("\nRandomly sampling $n_replicas initial configurations from trajectory")
n_available = length(raw_data["init"])
println("  Available configurations: $n_available")

if n_available < n_replicas
    error("Not enough configurations available. Need $n_replicas but only have $n_available")
end

# Sample without replacement
sampled_indices = Random.randperm(n_available)[1:n_replicas]
initial_systems = [raw_data["init"][idx] for idx in sampled_indices]

println("  Sampled configuration indices: $sampled_indices")

# Calculate and display initial energies
println("\nInitial energies for each replica:")
for (i, system) in enumerate(initial_systems)
    E_initial = potential_energy(system, model)
    println("  Replica $i (config $(sampled_indices[i])): $E_initial, ($(E_initial/length(system))/atom)")
end
run_tag = "_run2-dia-ace-primitive"
T_min = 300.0
T_max = 900.0
n_samples = 10000
burnin = 20000
thin = 20
exchange_interval = 20
step_size = 0.01

println("\nDistributed Parallel Tempering Parameters:")
println("  Number of replicas: $n_replicas")
println("  Number of workers: $(nworkers())")
println("  Temperature range: $T_min K to $T_max K")
println("  Samples per replica: $n_samples")
println("  Burn-in steps: $burnin")
println("  Thinning: $thin")
println("  Exchange interval: $exchange_interval")
println("  Step size: $step_size Å")

# Create sampler
sampler = RWMCSampler(step_size=step_size)
println("\nUsing RWMC sampler with step_size=$step_size Å")

println("\n" * "="^70)
println("Running Distributed Parallel Tempering with ACE Model")
println("="^70)
println("Note: This may take several minutes due to the expensive ACE calculations...")
println("Timing distributed parallel tempering...")

# Run distributed parallel tempering with timing
# replicas, temperatures, mcmc_rates, exchange_rates, trajs = @time run_parallel_tempering_distributed(
#     sampler, initial_systems, model, n_replicas, T_min, T_max;
#     n_samples=n_samples, burnin=burnin, thin=thin,
#     exchange_interval=exchange_interval, collect_forces=false
# )
replicas, temperatures, mcmc_rates, exchange_rates, trajs = @time run_parallel_tempering(
    sampler, initial_systems, model, n_replicas, T_min, T_max;
    n_samples=n_samples, burnin=burnin, thin=thin,
    exchange_interval=exchange_interval, collect_forces=false
)
#%%
# --- Analysis ---
println("\n" * "="^70)
println("Statistical Analysis")
println("="^70)

println("\n--- Sampling Results ---")
for i in 1:n_replicas
    println("Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
           "$(length(replicas[i])) samples, " *
           "MCMC acceptance: $(round(mcmc_rates[i], digits=3))")
end

# Check acceptance rates quality
println("\n--- MCMC Acceptance Rate Quality ---")
for i in 1:n_replicas
    acc_rate = mcmc_rates[i]
    if 0.15 < acc_rate < 0.4
        println("  Replica $i: $(GREEN)✓ Good$(RESET) ($(round(acc_rate, digits=3)))")
    elseif 0.1 < acc_rate < 0.5
        println("  Replica $i: $(GREEN)✓ OK$(RESET) ($(round(acc_rate, digits=3)))")
    else
        println("  Replica $i: $(RED)⚠ Check$(RESET) ($(round(acc_rate, digits=3)))")
    end
end

# Exchange rates
println("\n--- Exchange Acceptance Rates ---")
for i in 1:(n_replicas-1)
    rate = exchange_rates[i]
    status = 0.1 < rate < 0.5 ? "$(GREEN)✓$(RESET)" : "$(RED)⚠$(RESET)"
    println("  $(round(temperatures[i], digits=1)) K ↔ $(round(temperatures[i+1], digits=1)) K: " *
           "$(round(rate, digits=3)) $status")
end

# Test 1: Energy statistics per replica
println("\n--- Energy Statistics per Replica ---")
for i in 1:n_replicas
    energies = trajs[i].energy
    mean_energy = mean(energies)
    std_energy = std(energies)
    mean_energy_per_atom = mean_energy / 64
    std_energy_per_atom = std_energy / 64
    
    println("Replica $i (T=$(round(temperatures[i], digits=1)) K):")
    println("  Mean energy: $(round(mean_energy, digits=6)) eV ($(round(mean_energy_per_atom, digits=6)) eV/atom)")
    println("  Std energy: $(round(std_energy, digits=6)) eV ($(round(std_energy_per_atom, digits=6)) eV/atom)")
end

# Test 2: Equilibration check for target temperature (lowest)
println("\n--- Equilibration Assessment (Target T=$(round(T_min, digits=1)) K) ---")
energies_target = trajs[1].energy
n_half = length(energies_target) ÷ 2
first_half_mean = mean(energies_target[1:n_half])
second_half_mean = mean(energies_target[(n_half+1):end])
rel_diff = abs(first_half_mean - second_half_mean) / abs(first_half_mean)

println("  First half mean: $(round(first_half_mean / 64, digits=6)) eV/atom")
println("  Second half mean: $(round(second_half_mean / 64, digits=6)) eV/atom")
println("  Relative difference: $(round(rel_diff, digits=4))")

if rel_diff < 0.05
    println("  $(GREEN)✓ PASS$(RESET): Trajectory appears equilibrated")
else
    println("  $(RED)⚠ CHECK$(RESET): May need longer burn-in or more samples")
end

# Test 3: Structural Analysis - RDF and ADF for all temperatures
println("\n--- Structural Analysis for All Temperatures ---")
println("  Computing radial and angular distribution functions...")

r_cut = 5.5u"Å"

# Storage for RDF and ADF data for all replicas
all_rdf_distances = Vector{Vector{Float64}}(undef, n_replicas)
all_adf_angles_deg = Vector{Vector{Float64}}(undef, n_replicas)

# Compute RDF and ADF for each temperature
for i in 1:n_replicas
    samples_vec = FlexibleSystem.(replicas[i])
    
    # Compute RDF
    rdf_data = ACEpotentials.get_rdf(samples_vec, r_cut; rescale=true)
    all_rdf_distances[i] = rdf_data[(:Si, :Si)]
    
    # Compute ADF
    adf_data = ACEpotentials.get_adf(samples_vec, r_cut)
    all_adf_angles_deg[i] = rad2deg.(adf_data)
    
    println("  Replica $i (T=$(round(temperatures[i], digits=0)) K): " *
           "$(length(all_rdf_distances[i])) Si-Si pairs, " *
           "$(length(all_adf_angles_deg[i])) Si-Si-Si triplets")
end

# Analyze target temperature (replica 1) in detail
si_si_distances = all_rdf_distances[1]
si_si_si_angles_deg = all_adf_angles_deg[1]

println("\n--- Detailed Analysis for Target T=$(round(T_min, digits=1)) K ---")

# Analyze RDF peaks
first_peak_distances = filter(d -> 2.0 < d < 2.7, si_si_distances)
if !isempty(first_peak_distances)
    first_peak_pos = mean(first_peak_distances)
    println("  First RDF peak position: $(round(first_peak_pos, digits=3)) Å")
else
    first_peak_pos = 0.0
    println("  First RDF peak not clearly identified")
end

# Analyze ADF peaks
tetrahedral_angles = filter(θ -> 100 < θ < 120, si_si_si_angles_deg)
if !isempty(tetrahedral_angles)
    mean_tetrahedral = mean(tetrahedral_angles)
    println("  Mean tetrahedral angle: $(round(mean_tetrahedral, digits=2))° (ideal: 109.47°)")
else
    mean_tetrahedral = 0.0
    println("  Tetrahedral angles not clearly identified")
end

# --- Generate plots ---
println("\n--- Generating diagnostic plots ---")

# Plot 1: Energy trajectories for all replicas
p1 = plot(title="Energy Trajectories (All Replicas)", 
         xlabel="Sample", ylabel="Energy per atom (eV)", 
         legend=:outerright, size=(800, 400))
for i in 1:n_replicas
    plot!(p1, trajs[i].energy ./ 64, 
         label="T=$(round(temperatures[i], digits=0)) K",
         alpha=0.7, linewidth=1)
end

# Plot 2: Energy distributions for all replicas
p2 = plot(title="Energy Distributions", 
         xlabel="Energy per atom (eV)", ylabel="Density",
         legend=:outerright, size=(800, 400))
for i in 1:n_replicas
    histogram!(p2, trajs[i].energy ./ 64, 
              bins=30, 
              normalize=:pdf, 
              alpha=0.5,
              label="T=$(round(temperatures[i], digits=0)) K")
end

# Plot 3: Mean energy vs temperature
mean_energies = [mean(traj.energy) / 64 for traj in trajs]
p3 = scatter(temperatures, mean_energies,
            xlabel="Temperature (K)", 
            ylabel="Mean Energy per atom (eV)",
            title="Energy-Temperature Scaling",
            label="Sampled",
            markersize=8,
            color=:blue,
            size=(800, 400))
plot!(p3, temperatures, mean_energies,
     linewidth=2,
     linestyle=:dash,
     label="Trend",
     color=:red)

# Plot 4: Exchange rate matrix
exchange_matrix = zeros(n_replicas, n_replicas)
for i in 1:(n_replicas-1)
    exchange_matrix[i, i+1] = exchange_rates[i]
    exchange_matrix[i+1, i] = exchange_rates[i]
end

p4 = heatmap(1:n_replicas, 1:n_replicas, exchange_matrix,
            xlabel="Replica Index",
            ylabel="Replica Index",
            title="Exchange Rate Matrix",
            c=:viridis,
            clims=(0, 1),
            colorbar_title="Accept Rate",
            aspect_ratio=:equal,
            size=(800, 400))

for i in 1:(n_replicas-1)
    annotate!(p4, i+1, i, text("$(round(exchange_rates[i], digits=2))", 10, :white))
    annotate!(p4, i, i+1, text("$(round(exchange_rates[i], digits=2))", 10, :white))
end

# Plot 5: RDF for target temperature
p5 = histogram(si_si_distances,
              bins=100,
              normalize=:pdf,
              xlabel="Distance (Å)",
              ylabel="g(r)",
              title="RDF (Si-Si) at T=$(round(T_min, digits=0)) K",
              label="",
              alpha=0.7,
              color=:orange,
              xlims=(0, ustrip(u"Å", r_cut)),
              size=(800, 400))

vline!(p5, [2.35], linestyle=:dash, color=:red, linewidth=1, label="1st shell (~2.35 Å)")
if !isempty(first_peak_distances)
    vline!(p5, [first_peak_pos], linestyle=:dot, color=:blue, linewidth=2, label="Observed peak")
end

# Plot 6: ADF for target temperature
p6 = histogram(si_si_si_angles_deg,
              bins=100,
              normalize=:pdf,
              xlabel="Angle (degrees)",
              ylabel="P(θ)",
              title="ADF (Si-Si-Si) at T=$(round(T_min, digits=0)) K",
              label="",
              alpha=0.7,
              color=:purple,
              xlims=(0, 180),
              size=(800, 400))

vline!(p6, [109.47], linestyle=:dash, color=:red, linewidth=1, label="Tetrahedral (109.47°)")
if !isempty(tetrahedral_angles)
    vline!(p6, [mean_tetrahedral], linestyle=:dot, color=:blue, linewidth=2, label="Mean observed")
end

# Plot 7: RDF for all temperatures
p7 = plot(title="Radial Distribution Functions (All Temperatures)", 
         xlabel="Distance (Å)", 
         ylabel="g(r)",
         legend=:outerright,
         size=(800, 400))

for i in 1:n_replicas
    histogram!(p7, all_rdf_distances[i],
              bins=100,
              normalize=:pdf,
              alpha=0.5,
              label="T=$(round(temperatures[i], digits=0)) K",
              xlims=(0, ustrip(u"Å", r_cut)))
end
vline!(p7, [2.35], linestyle=:dash, color=:black, linewidth=2, label="1st shell (~2.35 Å)")

# Plot 8: ADF for all temperatures
p8 = plot(title="Angular Distribution Functions (All Temperatures)", 
         xlabel="Angle (degrees)", 
         ylabel="P(θ)",
         legend=:outerright,
         size=(800, 400))

for i in 1:n_replicas
    histogram!(p8, all_adf_angles_deg[i],
              bins=100,
              normalize=:pdf,
              alpha=0.5,
              label="T=$(round(temperatures[i], digits=0)) K",
              xlims=(0, 180))
end
vline!(p8, [109.47], linestyle=:dash, color=:black, linewidth=2, label="Tetrahedral (109.47°)")

# Plot 9: Acceptance rates as function of iterations
p9 = plot(title="Acceptance Rates vs Iteration", xlabel="Sample", 
         ylabel="Acceptance Rate", legend=:outerright, 
         size=(800, 400), ylims=(0, 1))
for i in 1:n_replicas
    acc_rates = trajs[i].acc_rate
    # Plot instantaneous acceptance rates
    plot!(p9, 1:length(acc_rates), acc_rates, 
          label="T=$(round(temperatures[i], digits=0)) K", 
          alpha=0.5, linewidth=1, color=i)
    
    # Compute and plot cumulative average
    cumulative_avg = cumsum(acc_rates) ./ (1:length(acc_rates))
    plot!(p9, 1:length(cumulative_avg), cumulative_avg,
          label="T=$(round(temperatures[i], digits=0)) K (cumul.)",
          linewidth=2.5, color=i, linestyle=:solid)
    
    # Add reference line at overall acceptance rate
    hline!(p9, [mcmc_rates[i]], linestyle=:dash, color=i, alpha=0.3, label="")
end

# Combine plots
combined_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9,
                    layout=(5,2), size=(1600, 2200), margin=5Plots.mm)

# Save plot
output_dir = joinpath(dirname(@__FILE__), "results")
if !isdir(output_dir)
    mkpath(output_dir)
end
output_file = joinpath(output_dir, string("ptd_ace_silicon_test", run_tag, ".png"))
savefig(combined_plot, output_file)
println("Saved diagnostic plots to: $output_file")

# --- Summary ---
println("\n" * "="^70)
println("Summary: Distributed Parallel Tempering with ACE Model for Silicon")
println("="^70)
println("$(GREEN)✓$(RESET) Distributed parallel tempering with ACE potential completed successfully")
println("$(GREEN)✓$(RESET) All replicas evolved in parallel on $(nworkers()) workers")
println("$(GREEN)✓$(RESET) Sampled 64-atom amorphous silicon system at multiple temperatures")
println("$(GREEN)✓$(RESET) Generated $(length(replicas[1])) equilibrium configurations per replica")
println("\nKey Results:")
println("  - System: 64 Si atoms (amorphous structure)")
println("  - Number of replicas: $n_replicas")
println("  - Temperature range: $(round(T_min, digits=1)) K to $(round(T_max, digits=1)) K")
println("  - Target temperature ($(round(T_min, digits=1)) K):")
println("    - MCMC acceptance rate: $(round(mcmc_rates[1], digits=3))")
println("    - Mean energy: $(round(mean_energies[1], digits=4)) eV/atom")
if !isempty(first_peak_distances)
    println("    - RDF first peak: $(round(first_peak_pos, digits=3)) Å (typical: ~2.35 Å)")
end
if !isempty(tetrahedral_angles)
    println("    - Mean Si-Si-Si angle: $(round(mean_tetrahedral, digits=2))° (tetrahedral: 109.47°)")
end

println("\nExchange Statistics:")
avg_exchange_rate = mean(exchange_rates)
println("  - Average exchange rate: $(round(avg_exchange_rate, digits=3))")
for i in 1:(n_replicas-1)
    println("  - T$(round(temperatures[i], digits=0)) ↔ T$(round(temperatures[i+1], digits=0)): $(round(exchange_rates[i], digits=3))")
end

println("\nPhysical Insights:")
println("  - RWMC with parallel tempering enables sampling across temperature range")
println("  - Replica exchange helps overcome energy barriers")
println("  - ACE potential enables realistic sampling of silicon structures")
println("  - Higher temperatures explore more configurations")
println("  - Target temperature benefits from exchanges with hot replicas")

println("\nPerformance Benefits:")
println("  - Replicas evolve independently between exchanges on $(nworkers()) workers")
println("  - Near-linear speedup with number of workers")
println("  - Exchange synchronization overhead is minimal")
println("  - Ideal for expensive ACE force/energy calculations")

println("\nAlgorithm Details:")
println("  - Random Walk Metropolis-Hastings with Replica Exchange")
println("  - Distributed execution across multiple worker processes")
println("  - Periodic configuration exchanges between neighboring temperatures")
println("  - Enhanced sampling efficiency compared to single-temperature MCMC")
println("="^70)

# --- Compute and Save Forces and Energies ---
println("\n" * "="^70)
println("Computing Forces and Energies for All Samples")
println("="^70)

data_output_dir = joinpath(dirname(@__FILE__), "..", "data")
if !isdir(data_output_dir)
    mkdir(data_output_dir)
end
#%%
for i in 1:n_replicas
    # Add energies and forces to all samples in replica i
    println("\nProcessing Replica $i (T=$(round(temperatures[i], digits=1)) K)...")
    println("  Number of samples: $(length(replicas[i]))")
    replicas[i] = add_energy_forces(replicas[i], model)
    
    # Save trajectory to file
    output_filename = joinpath(data_output_dir, "replica_$(i)_T$(round(Int, temperatures[i]))K_samples$(run_tag).xyz")
    println("  Saving trajectory to: $output_filename")
    ExtXYZ.save(output_filename, [r for r in replicas[i]])
    println("  $(GREEN)✓$(RESET) Saved $(length(replicas[i])) configurations")
end

println("\n" * "="^70)
println("$(GREEN)✓$(RESET) All replica trajectories with forces and energies saved to data folder")
println("="^70)

# Optional: Remove workers if desired
# rmprocs(workers())
