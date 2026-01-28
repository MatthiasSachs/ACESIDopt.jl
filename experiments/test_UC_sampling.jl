#=
Active Learning - Silicon with Distributed Parallel Tempering
Training from generated samples with PT sampling
Using ACEfit.BLR instead of sklearn BayesianRidge
=#

#=============================================================================
SIMULATION PARAMETERS - Set all parameters here
=============================================================================#

# Random seeds
const RANDOM_SEED = 2234

# Parallel computing
const N_WORKERS = 4

# Experiment identification
const EXPERIMENT_NAME = "test"

# "AL_ABSID_BLR_10it-SW"
#"AL_US-SW-5-demo"
const OUTPUT_DIR = joinpath(@__DIR__, "results")

# Data paths
const TEST_DATA_PATH = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_dia-primitive-2-high-K1200/data/replica_1_samples.xyz"
#"/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/examples/../data/Si-diamond-primitive-8atoms.xyz"
#"/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_dia-primitive-2-temp/data/replica_1_samples.xyz"
# "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_dia-primitive-2-temp/data/replica_1_samples.xyz"
# "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_dia-primitive-2-temp/data/replica_1_samples.xyz"
# "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_DW_dia-primitive-2-temp/data/replica_2_samples.xyz"
#"/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"
# "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-very-large-high-temp/data/replica_1_samples.xyz"
#"/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"
const LARGE_TEST_DATA_PATH = TEST_DATA_PATH
const TEST_THINNING = 10  # Thinning factor for test data
const LARGE_TEST_DATA_THINNING = 1  # Thinning factor for large test data
const INIT_CONFIGS_PATH = TEST_DATA_PATH # "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"  # Path to initial training candidates

# Reference model specification
const REF_MODEL = "../models/Si_ref_model-small-2.json"
#"SW" # "../models/Si_ref_model.json" # "SW"  # Use "SW" for Stillinger-Weber or provide path to ACE model file (e.g., "../models/Si_ref_model.json")

# Initial training set
const N_INITIAL_TRAIN = 1  # Number of initial training samples
const INITIAL_TRAIN_RAND = true  # If true, randomly select initial training samples; if false, use first N_INITIAL_TRAIN
const RANDOM_INIT = true  # If true, sample raw_data["init"] using query_US; if false, use configurations from INIT_CONFIGS_PATH

# Active learning parameters
const N_ACTIVE_ITERATIONS = 20  # Number of active learning iterations
const E_MAX_ACC = -320.6  # Maximum energy (eV) threshold for accepting queried configurations

# Query function selection
# Options: "TSSID", "ABSID", "US", "TrainData", "HAL"
const QUERY_FUNCTION = "HAL"

# Query function specific parameters
# For query_TrainData (callable by selecting QUERY_FUNCTION = "TrainData"):
const TRAIN_DATA_NAME = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_DW_dia-primitive-2-temp-2/data/replica_1_samples.xyz"
# "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"

# For query_HAL:
const TAU = 1.0  # Temperature annealing parameter for HAL
const SIGMA_STOP = 0.1  # Stopping criterion for HAL

# ACE model parameters
const ACE_ELEMENTS = [:Si]
const ACE_RCUT = 5.5  # Å
const ACE_ORDER = 3  # body-order - 1
const ACE_TOTALDEGREE = 6
const ACE_PRIOR_ORDER = 4

# Training weights
const WEIGHT_ENERGY = 30.0
const WEIGHT_FORCES = 1.0
const WEIGHT_VIRIAL = 1.0

# ACEfit.BLR parameters
const COMMITTEE_SIZE = 1000
const FACTORIZATION = :svd

# Distributed parallel tempering parameters
const N_REPLICAS = 4
const T_MIN = 1200.0  # K
const T_MAX = 1600.0  # K
const N_SAMPLES_PT = 100 #1000
const BURNIN_PT = 10000
const THIN_PT = 20 #100
const EXCHANGE_INTERVAL = 50
const STEP_SIZE_PT = 0.02  # Å
const R_CUT = 7.5  # Å (for RDF/ADF analysis)

# Cholesky jitter parameters
const MAX_JITTER_FRACTION = 1e-3

#=============================================================================
END OF PARAMETERS
=============================================================================#

using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel, save_simulation_parameters, load_simulation_parameters, comp_potE_MAE_RMSE
using ACEpotentials: _make_prior
using ACEpotentials
using ACEpotentials: make_atoms_data, assess_dataset, 
                     _rep_dimer_data_atomsbase, default_weights, AtomsData
using LinearAlgebra: Diagonal, I, inv
using ACESIDopt: comp_potE_error
using ACEfit
using ACEfit: bayesian_linear_regression
using ACEpotentials: set_committee!

using ACESIDopt: expected_red_variance, row_mapping, pred_variance
using ACESIDopt.QueryModels: query_TSSID, query_ABSID, query_US, query_TrainData, query_HAL
using StatsPlots
using Random

using Distributed
using ACESIDopt.MSamplers: RWMCSampler, run_parallel_tempering_distributed
using Random
using LinearAlgebra: cholesky
using ACESIDopt: cholesky_with_jitter, add_energy_forces, add_forces, add_energy, convert_forces_to_svector
using ACESIDopt: mflexiblesystem, queryASEModel
using AtomsCalculators: potential_energy, forces
using ACESIDopt: plot_forces_comparison, plot_energy_comparison, generate_ptd_diagnostics_and_log
using ProgressMeter
using EmpiricalPotentials
using ExtXYZ

println("="^70)
println("Active Learning - Silicon with Distributed Parallel Tempering")
println("Using ACEfit.BLR for Model Fitting")
println("="^70)

# Set random seed
Random.seed!(RANDOM_SEED)

# Add worker processes if not already added
if nworkers() == 1
    println("Adding $N_WORKERS worker processes...")
    addprocs(N_WORKERS)
    println("Workers available: $(nworkers())")
end

# Load required packages on all workers
@everywhere begin
    using ACESIDopt
    using AtomsBase
    using AtomsBase: FlexibleSystem
    using Unitful
    using ExtXYZ
    using Statistics
    using Random
    using ACEpotentials
    using ACESIDopt.MSamplers: RWMCSampler, run_parallel_tempering_distributed
    using AtomsCalculators: potential_energy, forces
    using EmpiricalPotentials
end

# Define directory paths (not created)
experiment_dir = joinpath(OUTPUT_DIR, EXPERIMENT_NAME)
plots_dir = joinpath(experiment_dir, "plots")
data_dir = joinpath(experiment_dir, "data")
models_dir = joinpath(experiment_dir, "models")
other_data_dir = joinpath(experiment_dir, "other-data")
pt_diagnostics_dir = joinpath(plots_dir, "pt_diagnostics")
test_predictions_dir = joinpath(plots_dir, "test_predictions")
train_predictions_dir = joinpath(plots_dir, "train_predictions")
train_new_samples_dir = joinpath(plots_dir, "train_new_samples")

# Save simulation parameters
params = Dict(
    "experiment_name" => EXPERIMENT_NAME,
    "random_seed" => RANDOM_SEED,
    "parallel_computing" => Dict(
        "n_workers" => N_WORKERS
    ),
    "input_output" => Dict(
        "output_dir" => OUTPUT_DIR,
        "experiment_dir" => experiment_dir,
        "plots_dir" => plots_dir,
        "data_dir" => data_dir,
        "models_dir" => models_dir,
        "test_data_path" => TEST_DATA_PATH,
        "large_test_data_path" => LARGE_TEST_DATA_PATH,
        "test_thinning" => TEST_THINNING,
        "large_test_data_thinning" => LARGE_TEST_DATA_THINNING,
        "init_configs_path" => INIT_CONFIGS_PATH
    ),
    "training" => Dict(
        "n_initial_train" => N_INITIAL_TRAIN,
        "initial_train_rand" => INITIAL_TRAIN_RAND,
        "random_init" => RANDOM_INIT,
        "n_active_iterations" => N_ACTIVE_ITERATIONS
    ),
    "query_function" => Dict(
        "method" => QUERY_FUNCTION,
        "train_data_name" => TRAIN_DATA_NAME,
        "tau" => TAU,
        "sigma_stop" => SIGMA_STOP
    ),
    "ace_model" => Dict(
        "elements" => string.(ACE_ELEMENTS),
        "rcut" => ACE_RCUT,
        "order" => ACE_ORDER,
        "totaldegree" => ACE_TOTALDEGREE,
        "prior_order" => ACE_PRIOR_ORDER
    ),
    "weights" => Dict(
        "energy" => WEIGHT_ENERGY,
        "forces" => WEIGHT_FORCES,
        "virial" => WEIGHT_VIRIAL
    ),
    "acefit_blr" => Dict(
        "committee_size" => COMMITTEE_SIZE,
        "factorization" => string(FACTORIZATION)
    ),
    "parallel_tempering" => Dict(
        "n_replicas" => N_REPLICAS,
        "T_min" => T_MIN,
        "T_max" => T_MAX,
        "n_samples" => N_SAMPLES_PT,
        "burnin" => BURNIN_PT,
        "thin" => THIN_PT,
        "exchange_interval" => EXCHANGE_INTERVAL,
        "step_size" => STEP_SIZE_PT,
        "r_cut" => R_CUT
    ),
    "reference_model" => Dict(
        "model_spec" => REF_MODEL
    )
)

params_file = joinpath(experiment_dir, "simulation_parameters.yaml")
# save_simulation_parameters(params_file, params)  # Disabled: no file writing

# Load data - using generated samples as training/candidates
raw_data = Dict{String, Any}()
raw_data["test"] = ExtXYZ.load(TEST_DATA_PATH)

using ACESIDopt: convert_forces_to_svector

# Subsample data points for initial training set
n_total = length(raw_data["test"])
println("Total available samples: $n_total")
train_indices = sort(Random.randperm(n_total)[1:N_INITIAL_TRAIN])
test_indices = setdiff(1:n_total, train_indices)

println("Selected training indices: $train_indices")
println("Remaining test samples: $(length(test_indices))")

TEST_THINNING=100
# Create test set from remaining data
raw_data_train = convert_forces_to_svector.(raw_data["test"][1:5])

# ACE model for Silicon (single element)
model = ace1_model(elements = ACE_ELEMENTS,
                   rcut = ACE_RCUT,
                   order = ACE_ORDER,        # body-order - 1
                   totaldegree = ACE_TOTALDEGREE);

Psqrt = _make_prior(model, ACE_PRIOR_ORDER, nothing) # square root of prior precision matrix

my_weights() = Dict("default"=>Dict("E"=>WEIGHT_ENERGY, "F"=>WEIGHT_FORCES, "V"=>WEIGHT_VIRIAL))

data_train = make_atoms_data(raw_data_train, model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = my_weights())
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 
Yw_train = W_train .* Y_train

# Use ACEfit.BLR instead of sklearn BayesianRidge
solver = ACEfit.BLR(committee_size = COMMITTEE_SIZE, factorization = FACTORIZATION)
result = bayesian_linear_regression(Awp_train, Yw_train; solver.kwargs..., ret_covar = true)

# Extract parameters
Σ = result["covar"]
coef_tilde = result["C"]
α = result["var_e"]  # noise precision (variance of observation noise)
coef = Psqrt \ coef_tilde

# Set linear parameters for the model
ACEpotentials.Models.set_linear_parameters!(model, coef)



# Sample initial systems for surrogate PT
n_available = length(raw_data_train)

# Create samplers
sampler_biased = RWMCSampler(step_size=STEP_SIZE_PT)
sampler_gibbs = RWMCSampler(step_size=STEP_SIZE_PT)

# Keep sampling until we get valid candidates with non-empty exp_red
exp_red = Float64[]
attempt = 0

# Sample initial systems for biased PT
sampled_indices_biased = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
if length(sampled_indices_biased) < N_REPLICAS
    # If not enough unique samples, repeat some
    while length(sampled_indices_biased) < N_REPLICAS
        push!(sampled_indices_biased, rand(1:n_available))
    end
end
initial_systems_biased = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_biased]

# Sample initial systems for candidate PT
sampled_indices_gibbs = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
if length(sampled_indices_gibbs) < N_REPLICAS
    while length(sampled_indices_gibbs) < N_REPLICAS
        push!(sampled_indices_gibbs, rand(1:n_available))
    end
end
initial_systems_gibbs = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_gibbs]

# Run surrogate and candidate PT in parallel using @sync
biased_replicas, biased_temperatures, biased_mcmc_rates, biased_exchange_rates, biased_trajs = nothing, nothing, nothing, nothing, nothing
gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, gibbs_exchange_rates, gibbs_trajs = nothing, nothing, nothing, nothing, nothing


println("  Starting gibbs PT...")
gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, gibbs_exchange_rates, gibbs_trajs = 
    run_parallel_tempering_distributed(
        sampler_gibbs, initial_systems_gibbs, ts_model, N_REPLICAS, T_MIN, T_MAX;
        n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
        exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
    )
println("  Gibbs PT complete!")

gibbs_replicas
# Diagnostics generation disabled (no file writing)
println("\nSkipping biased PT diagnostics (file writing disabled)...")
biased_diag_dir = joinpath(pt_diagnostics_dir, "biased")
# mkpath(biased_diag_dir)  # Disabled: no directory creation
# generate_ptd_diagnostics_and_log(biased_replicas, biased_temperatures, biased_mcmc_rates, 
#                                  biased_exchange_rates, biased_trajs, biased_diag_dir, t, R_CUT * u"Å")

# Diagnostics generation disabled (no file writing)
println("\nSkipping gibbs PT diagnostics (file writing disabled)...")
gibbs_diag_dir = joinpath(pt_diagnostics_dir, "gibbs")
# mkpath(gibbs_diag_dir)  # Disabled: no directory creation
# generate_ptd_diagnostics_and_log(gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, 
#                                  gibbs_exchange_rates, gibbs_trajs, gibbs_diag_dir, t, R_CUT * u"Å")
gibbs_replicas
# Extract samples from target temperature (lowest temperature, replica 1)
gibbs_samples = gibbs_replicas[1]

# Add energies to surrogate samples and energy+forces to candidate samples
if constraint === nothing
    raw_data_tgibbs = [at for at in add_energy_forces(gibbs_samples, model)]
else
    raw_data_tgibbs = [at for at in add_energy_forces(gibbs_samples, model) if constraint(at)]
end


data_tgibbs = make_atoms_data(raw_data_tgibbs, model; 
                            energy_key = "energy", 
                            force_key = nothing, 
                            virial_key = nothing, 
                            weights = my_weights())
n_data_tgibbs = length(raw_data_tgibbs)
        
A_tgibbs, Y_tgibbs, W_tgibbs = ACEfit.assemble(data_tgibbs, model)
Awp_tgibbs = Diagonal(W_tgibbs) * (A_tgibbs / Psqrt)

# Randomly sample a new atomic configuration using query_US
println("\nSampling new atomic configuration using query_US...")
atoms = ExtXYZ.Atoms(initial_systems_gibbs[1])
function randomize_positions!(atoms)
    
    cell = AtomsBase.cell_vectors(atoms)
    n_atoms = length(atoms)
    
    # Generate new positions uniformly in the cell
    for i in 1:n_atoms
        # Position = sum of cell vectors with random coefficients [0,1]
        atoms.atom_data.position[i] = sum(cell[j] * rand() for j in 1:3)
    end
end

randomize_positions!(atoms)
ref_energy = potential_energy(atoms, ref_model)
ref_forces = forces(atoms, ref_model)
    
at = add_energy_forces(atoms, ref_energy, ref_forces)

at_data = make_atoms_data([at], model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = my_weights())
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 





predictive_variance(model, at, Σ; Psqrt=Psqrt)
using ACEpotentials: site_descriptors
using ACEfit: feature_matrix
x = site_descriptors(at, model) 
x[2]

# size(Awp_tgibbs)
# using ACEpotentials: energy_forces_virial_basis

# bas_eval = energy_forces_virial_basis(at, model)
# bas_eval.energy
# bas_eval.forces
# x = descriptors_vector(model, at)
# feature_matrix(at_data, model)
#%%
# ExpVarFunctor and related utilities are now in src/utils.jl
using ACESIDopt: ExpVarFunctor, massemble, mset_positions!, mget_positions


g = ExpVarFunctor(Σ, Awp_tgibbs, α, at, model)
x = mget_positions(at)
g(x)

using FiniteDifferences

# Compute function value
g_val = g(x)
println("Function value g(x): $g_val")

fdm = central_fdm(5, 1)  # 5-point stencil, 1st derivative
grad_g_f(x) = FiniteDifferences.grad(fdm, g, x)[1]
grad_g_f(x)



println("Gradient computed via finite differences")
println("Gradient norm: $(norm(grad_g))")




at_data[1]
AtomsData(at)


at_data = make_atoms_data([at], model; energy_key = "energy", 
                            force_key = "forces", 
                            virial_key = nothing,
                            weights = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
                    )
a,_,_ = ACEfit.assemble(at_data, model)
expected_red_variance(Σ, Awp_tgibbs, a, α) 

# Test position get/set utilities
x = mget_positions(at)
mset_positions!(at, x)


expected_red_variance(Σ, Awp_tgibbs, at, α)


if !isempty(exp_red)
    idx = argmax(exp_red)
    println("\nSelected candidate $idx with max expected variance reduction: $(exp_red[idx])")

    # Use reference model for ground truth evaluation
    println("Computing energy and forces with reference potential...")
    
    # Compute energy and forces with reference model
    selected_system = raw_data_tgibbs[idx]
    ref_energy = potential_energy(selected_system, ref_model)
    ref_forces = forces(selected_system, ref_model)
    
    selected_system = add_energy_forces(selected_system, ref_energy, ref_forces)
else
    println("Warning: exp_red is empty, no valid candidates found, resampling...")
end


# Check if selected system energy is below threshold
selected_energy = selected_system.system_data.energy
println("\nSelected system energy: $selected_energy eV")

if selected_energy <= E_MAX_ACC
    push!(raw_data_train, deepcopy(selected_system))
    println("Configuration ACCEPTED (E = $selected_energy eV <= E_MAX_ACC = $E_MAX_ACC eV)")
    println("Training set size: $(length(raw_data_train))")
    
    # Plotting disabled (no file writing)
    # p_energy_train = plot_energy_comparison(raw_data_train, model,
    #                            joinpath(train_new_samples_dir, "train_energy_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
    # 
    # p_forces_train = plot_forces_comparison(raw_data_train, model, 
    #                                joinpath(train_new_samples_dir, "train_forces_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
else
    global n_rejected += 1
    println("Configuration REJECTED (E = $selected_energy eV > E_MAX_ACC = $E_MAX_ACC eV)")
    println("Total rejected queries: $n_rejected")
    println("Training set size remains: $(length(raw_data_train))")
end


# End timing the main loop
main_loop_end_time = time()