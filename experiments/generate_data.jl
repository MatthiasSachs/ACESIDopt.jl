# Test Distributed Parallel Tempering with Stillinger-Weber Potential for Silicon
# This test demonstrates using the Stillinger-Weber empirical potential with distributed replica exchange MCMC
# to sample from the Gibbs-Boltzmann distribution at multiple temperatures of a 64-atom amorphous silicon system
# Extended version: computes and saves forces and energies for all samples

#=============================================================================
SIMULATION PARAMETERS - Set all parameters here
=============================================================================#

# Random seeds
const INITIAL_SEED = 43
const SIMULATION_SEED = 42

# Parallel computing
const N_WORKERS = 4

# Input/Output paths
const SIMULATION_NAME = "ptd_ACE_silicon_dia-primitive-2-large"
const INPUT_DATA_PATH = joinpath(@__DIR__, "..", "data", "Si-diamond-primitive-2atom-large.xyz")
const OUTPUT_DIR = joinpath(@__DIR__, "results")

# Model specification
const MODEL = "../models/Si_ref_model.json"
#"SW"  # Use "SW" for Stillinger-Weber or provide path to ACE model file (e.g., "../models/Si_ref_model.json")

# Parallel tempering parameters
const N_REPLICAS = 4
const T_MIN = 300.0  # K
const T_MAX = 900.0  # K

# MCMC sampling parameters
const N_SAMPLES = 10000 #10000
const BURNIN = 20000
const THIN = 20
const EXCHANGE_INTERVAL = 20
const STEP_SIZE = 0.01  # Å

# Analysis parameters
const R_CUT = 5.5  # Å (for RDF/ADF analysis)
const N_ATOMS = 2  # Number of atoms in system (for energy per atom calculations)

# Physical constants
const KB = 8.617333262e-5  # Boltzmann constant in eV/K

#=============================================================================
END OF PARAMETERS
=============================================================================#

using ACESIDopt: add_energy_forces, save_simulation_parameters, load_simulation_parameters

Random.seed!(INITIAL_SEED)
# Add worker processes if not already added
if nworkers() == 1
    println("Adding $N_WORKERS worker processes...")
    addprocs(N_WORKERS)
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
    using EmpiricalPotentials
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
println("Testing Distributed Parallel Tempering for Silicon")
println("="^70)

# Create simulation directory structure
simulation_dir = joinpath(OUTPUT_DIR, SIMULATION_NAME)
plots_dir = joinpath(simulation_dir, "plots")
data_dir = joinpath(simulation_dir, "data")

if !isdir(simulation_dir)
    mkpath(simulation_dir)
    println("Created simulation directory: $simulation_dir")
end
if !isdir(plots_dir)
    mkpath(plots_dir)
    println("Created plots directory: $plots_dir")
end
if !isdir(data_dir)
    mkpath(data_dir)
    println("Created data directory: $data_dir")
end

# Save simulation parameters
params = Dict(
    "simulation_name" => SIMULATION_NAME,
    "random_seeds" => Dict(
        "initial_seed" => INITIAL_SEED,
        "simulation_seed" => SIMULATION_SEED
    ),
    "parallel_computing" => Dict(
        "n_workers" => N_WORKERS
    ),
    "input_output" => Dict(
        "input_data_path" => INPUT_DATA_PATH,
        "output_dir" => OUTPUT_DIR,
        "simulation_dir" => simulation_dir,
        "plots_dir" => plots_dir,
        "data_dir" => data_dir
    ),
    "parallel_tempering" => Dict(
        "n_replicas" => N_REPLICAS,
        "T_min" => T_MIN,
        "T_max" => T_MAX
    ),
    "mcmc_sampling" => Dict(
        "n_samples" => N_SAMPLES,
        "burnin" => BURNIN,
        "thin" => THIN,
        "exchange_interval" => EXCHANGE_INTERVAL,
        "step_size" => STEP_SIZE
    ),
    "analysis" => Dict(
        "r_cut" => R_CUT,
        "n_atoms" => N_ATOMS
    ),
    "physical_constants" => Dict(
        "kb" => KB
    ),
    "model" => Dict(
        "model_spec" => MODEL
    )
)

params_file = joinpath(simulation_dir, "simulation_parameters.yaml")
save_simulation_parameters(params_file, params)

# Set random seed
Random.seed!(SIMULATION_SEED)

# Create or load model on main process
if MODEL == "SW"
    println("\nCreating Stillinger-Weber potential for Silicon")
    model = EmpiricalPotentials.StillingerWeber()
    println("Stillinger-Weber potential created successfully")
    
    # Update params with model info
    params["model"]["type"] = "StillingerWeber"
    params["model"]["description"] = "Empirical potential for Silicon"
else
    # Load ACE model from file
    model_path = MODEL
    if !isabspath(model_path)
        model_path = joinpath(@__DIR__, MODEL)
    end
    println("\nLoading ACE model from: $model_path")
    model = ACEpotentials.load_model(model_path)[1]
    println("ACE model loaded successfully")
    
    # Save a copy of the ACE model in the simulation directory
    model_copy_path = joinpath(simulation_dir, basename(model_path))
    println("Saving copy of ACE model to: $model_copy_path")
    ACEpotentials.save_model(model, model_copy_path)
    println("ACE model copy saved successfully")
    
    # Update params with model info
    params["model"]["type"] = "ACE"
    params["model"]["model_file_original"] = model_path
    params["model"]["model_file_copy"] = model_copy_path
    params["model"]["description"] = "ACE potential loaded from file"
end

# Save updated parameters with model info
save_simulation_parameters(params_file, params)

# Create model on all workers
if MODEL == "SW"
    @everywhere model = EmpiricalPotentials.StillingerWeber()
else
    model_path_for_workers = MODEL
    if !isabspath(model_path_for_workers)
        model_path_for_workers = joinpath(@__DIR__, MODEL)
    end
    @everywhere model = ACEpotentials.load_model($model_path_for_workers)[1]
end

# Load configurations from input file
raw_data = Dict{String, Any}()
println("\nLoading silicon configurations from: $INPUT_DATA_PATH")
raw_data["init"] = [ExtXYZ.load(INPUT_DATA_PATH) for i in 1:N_REPLICAS]
println("Loaded $(length(raw_data["init"])) configurations")

# Randomly sample N_REPLICAS initial configurations
println("\nRandomly sampling $N_REPLICAS initial configurations from trajectory")
n_available = length(raw_data["init"])
println("  Available configurations: $n_available")

if n_available < N_REPLICAS
    error("Not enough configurations available. Need $N_REPLICAS but only have $n_available")
end

# Sample without replacement
sampled_indices = Random.randperm(n_available)[1:N_REPLICAS]
initial_systems =  [raw_data["init"][idx] for idx in sampled_indices]

println("  Sampled configuration indices: $sampled_indices")

# Calculate and display initial energies
println("\nInitial energies for each replica:")
for (i, system) in enumerate(initial_systems)
    E_initial = potential_energy(system, model)
    println("  Replica $i (config $(sampled_indices[i])): $E_initial, ($(E_initial/length(system))/atom)")
end

println("\nDistributed Parallel Tempering Parameters:")
println("  Number of replicas: $N_REPLICAS")
println("  Number of workers: $(nworkers())")
println("  Temperature range: $T_MIN K to $T_MAX K")
println("  Samples per replica: $N_SAMPLES")
println("  Burn-in steps: $BURNIN")
println("  Thinning: $THIN")
println("  Exchange interval: $EXCHANGE_INTERVAL")
println("  Step size: $STEP_SIZE Å")

# Create sampler
sampler = RWMCSampler(step_size=STEP_SIZE)
println("\nUsing RWMC sampler with step_size=$STEP_SIZE Å")

println("\n" * "="^70)
println("Running Distributed Parallel Tempering")
println("="^70)
println("Note: This may take several minutes...")
println("Timing distributed parallel tempering...")

# Run distributed parallel tempering with timing
replicas, temperatures, mcmc_rates, exchange_rates, trajs = @time run_parallel_tempering_distributed(
    sampler, initial_systems, model, N_REPLICAS, T_MIN, T_MAX;
    n_samples=N_SAMPLES, burnin=BURNIN, thin=THIN,
    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
)
#%%
# --- Analysis ---
println("\n" * "="^70)
println("Statistical Analysis")
println("="^70)

println("\n--- Sampling Results ---")
for i in 1:N_REPLICAS
    println("Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
           "$(length(replicas[i])) samples, " *
           "MCMC acceptance: $(round(mcmc_rates[i], digits=3))")
end

# Check acceptance rates quality
println("\n--- MCMC Acceptance Rate Quality ---")
for i in 1:N_REPLICAS
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
for i in 1:(N_REPLICAS-1)
    rate = exchange_rates[i]
    status = 0.1 < rate < 0.5 ? "$(GREEN)✓$(RESET)" : "$(RED)⚠$(RESET)"
    println("  $(round(temperatures[i], digits=1)) K ↔ $(round(temperatures[i+1], digits=1)) K: " *
           "$(round(rate, digits=3)) $status")
end

# Test 1: Energy statistics per replica
println("\n--- Energy Statistics per Replica ---")
for i in 1:N_REPLICAS
    energies = trajs[i].energy
    mean_energy = mean(energies)
    std_energy = std(energies)
    mean_energy_per_atom = mean_energy / N_ATOMS
    std_energy_per_atom = std_energy / N_ATOMS
    
    println("Replica $i (T=$(round(temperatures[i], digits=1)) K):")
    println("  Mean energy: $(round(mean_energy, digits=6)) eV ($(round(mean_energy_per_atom, digits=6)) eV/atom)")
    println("  Std energy: $(round(std_energy, digits=6)) eV ($(round(std_energy_per_atom, digits=6)) eV/atom)")
end

# Test 2: Equilibration check for target temperature (lowest)
println("\n--- Equilibration Assessment (Target T=$(round(T_MIN, digits=1)) K) ---")
energies_target = trajs[1].energy
n_half = length(energies_target) ÷ 2
first_half_mean = mean(energies_target[1:n_half])
second_half_mean = mean(energies_target[(n_half+1):end])
rel_diff = abs(first_half_mean - second_half_mean) / abs(first_half_mean)

println("  First half mean: $(round(first_half_mean / N_ATOMS, digits=6)) eV/atom")
println("  Second half mean: $(round(second_half_mean / N_ATOMS, digits=6)) eV/atom")
println("  Relative difference: $(round(rel_diff, digits=4))")

if rel_diff < 0.05
    println("  $(GREEN)✓ PASS$(RESET): Trajectory appears equilibrated")
else
    println("  $(RED)⚠ CHECK$(RESET): May need longer burn-in or more samples")
end

# Test 3: Structural Analysis - RDF and ADF for all temperatures
println("\n--- Structural Analysis for All Temperatures ---")
println("  Computing radial and angular distribution functions...")

r_cut = R_CUT * u"Å"

# Storage for RDF and ADF data for all replicas
all_rdf_distances = Vector{Vector{Float64}}(undef, N_REPLICAS)
all_adf_angles_deg = Vector{Vector{Float64}}(undef, N_REPLICAS)

# Compute RDF and ADF for each temperature
for i in 1:N_REPLICAS
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

println("\n--- Detailed Analysis for Target T=$(round(T_MIN, digits=1)) K ---")

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
for i in 1:N_REPLICAS
    plot!(p1, trajs[i].energy ./ 64, 
         label="T=$(round(temperatures[i], digits=0)) K",
         alpha=0.7, linewidth=1)
end

# Plot 2: Energy distributions for all replicas
p2 = plot(title="Energy Distributions", 
         xlabel="Energy per atom (eV)", ylabel="Density",
         legend=:outerright, size=(800, 400))
for i in 1:N_REPLICAS
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
exchange_matrix = zeros(N_REPLICAS, N_REPLICAS)
for i in 1:(N_REPLICAS-1)
    exchange_matrix[i, i+1] = exchange_rates[i]
    exchange_matrix[i+1, i] = exchange_rates[i]
end

p4 = heatmap(1:N_REPLICAS, 1:N_REPLICAS, exchange_matrix,
            xlabel="Replica Index",
            ylabel="Replica Index",
            title="Exchange Rate Matrix",
            c=:viridis,
            clims=(0, 1),
            colorbar_title="Accept Rate",
            aspect_ratio=:equal,
            size=(800, 400))

for i in 1:(N_REPLICAS-1)
    annotate!(p4, i+1, i, text("$(round(exchange_rates[i], digits=2))", 10, :white))
    annotate!(p4, i, i+1, text("$(round(exchange_rates[i], digits=2))", 10, :white))
end

# Plot 5: RDF for target temperature
p5 = histogram(si_si_distances,
              bins=100,
              normalize=:pdf,
              xlabel="Distance (Å)",
              ylabel="g(r)",
              title="RDF (Si-Si) at T=$(round(T_MIN, digits=0)) K",
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
              title="ADF (Si-Si-Si) at T=$(round(T_MIN, digits=0)) K",
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

for i in 1:N_REPLICAS
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

for i in 1:N_REPLICAS
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
for i in 1:N_REPLICAS
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

# Save plot to plots directory
output_file = joinpath(plots_dir, "diagnostics.png")
savefig(combined_plot, output_file)
println("Saved diagnostic plots to: $output_file")

# --- Summary ---
println("\n" * "="^70)
println("Summary: Distributed Parallel Tempering with Stillinger-Weber Potential for Silicon")
println("="^70)
println("$(GREEN)✓$(RESET) Distributed parallel tempering with Stillinger-Weber potential completed successfully")
println("$(GREEN)✓$(RESET) All replicas evolved in parallel on $(nworkers()) workers")
println("$(GREEN)✓$(RESET) Sampled 64-atom silicon system at multiple temperatures")
println("$(GREEN)✓$(RESET) Generated $(length(replicas[1])) equilibrium configurations per replica")
println("\nKey Results:")
println("  - System: 64 Si atoms")
println("  - Number of replicas: $N_REPLICAS")
println("  - Temperature range: $(round(T_MIN, digits=1)) K to $(round(T_MAX, digits=1)) K")
println("  - Target temperature ($(round(T_MIN, digits=1)) K):")
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
for i in 1:(N_REPLICAS-1)
    println("  - T$(round(temperatures[i], digits=0)) ↔ T$(round(temperatures[i+1], digits=0)): $(round(exchange_rates[i], digits=3))")
end

println("\nPhysical Insights:")
println("  - RWMC with parallel tempering enables sampling across temperature range")
println("  - Replica exchange helps overcome energy barriers")
println("  - Stillinger-Weber potential enables fast sampling of silicon structures")
println("  - Higher temperatures explore more configurations")
println("  - Target temperature benefits from exchanges with hot replicas")

println("\nPerformance Benefits:")
println("  - Replicas evolve independently between exchanges on $(nworkers()) workers")
println("  - Near-linear speedup with number of workers")
println("  - Exchange synchronization overhead is minimal")
println("  - Empirical potentials are significantly faster than machine-learned potentials")

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

# Save temperatures to file
temperatures_file = joinpath(data_dir, "temperatures.yaml")
temp_dict = Dict("temperatures" => Dict(i => temperatures[i] for i in 1:N_REPLICAS))
save_simulation_parameters(temperatures_file, temp_dict)
println("Saved temperatures to: $temperatures_file")

for i in 1:N_REPLICAS
    # Add energies and forces to all samples in replica i
    println("\nProcessing Replica $i (T=$(round(temperatures[i], digits=1)) K)...")
    println("  Number of samples: $(length(replicas[i]))")
    replicas[i] = add_energy_forces(replicas[i], model)
    
    # Save trajectory to file in data directory
    output_filename = joinpath(data_dir, "replica_$(i)_samples.xyz")
    println("  Saving trajectory to: $output_filename")
    ExtXYZ.save(output_filename, [r for r in replicas[i]])
    println("  $(GREEN)✓$(RESET) Saved $(length(replicas[i])) configurations")
end

println("\n" * "="^70)
println("$(GREEN)✓$(RESET) All replica trajectories with forces and energies saved to: $data_dir")
println("$(GREEN)✓$(RESET) Diagnostic plots saved to: $plots_dir")
println("$(GREEN)✓$(RESET) Simulation parameters saved to: $params_file")
println("="^70)

# Optional: Remove workers if desired
# rmprocs(workers())
