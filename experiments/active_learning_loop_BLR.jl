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
const EXPERIMENT_NAME = "AL_US-SW-1"
const OUTPUT_DIR = joinpath(@__DIR__, "results")

# Data paths
const TEST_DATA_PATH = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"
const LARGE_TEST_DATA_PATH = TEST_DATA_PATH
const TEST_THINNING = 10  # Thinning factor for test data
const LARGE_TEST_DATA_THINNING = 10  # Thinning factor for large test data
const INIT_CONFIGS_PATH = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"  # Path to initial training candidates

# Reference model specification
const REF_MODEL = "SW" # "../models/Si_ref_model.json" # "SW"  # Use "SW" for Stillinger-Weber or provide path to ACE model file (e.g., "../models/Si_ref_model.json")

# Initial training set
const N_INITIAL_TRAIN = 3  # Number of initial training samples
const INITIAL_TRAIN_RAND = true  # If true, randomly select initial training samples; if false, use first N_INITIAL_TRAIN

# Active learning parameters
const N_ACTIVE_ITERATIONS = 100  # Number of active learning iterations

# Query function selection
# Options: "TSSID", "ABSID", "US", "TrainData", "HAL"
const QUERY_FUNCTION = "US"

# Query function specific parameters
# For query_TrainData (callable by selecting QUERY_FUNCTION = "TrainData"):
const TRAIN_DATA_NAME = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"

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
const T_MIN = 300.0  # K
const T_MAX = 900.0  # K
const N_SAMPLES_PT = 1000
const BURNIN_PT = 10000
const THIN_PT = 10
const EXCHANGE_INTERVAL = 50
const STEP_SIZE_PT = 0.02  # Å
const R_CUT = 5.5  # Å (for RDF/ADF analysis)

# Cholesky jitter parameters
const MAX_JITTER_FRACTION = 1e-3

#=============================================================================
END OF PARAMETERS
=============================================================================#

using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel, save_simulation_parameters, load_simulation_parameters
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

# Create experiment directory structure under experiments/results/
experiment_dir = joinpath(OUTPUT_DIR, EXPERIMENT_NAME)
plots_dir = joinpath(experiment_dir, "plots")
data_dir = joinpath(experiment_dir, "data")
models_dir = joinpath(experiment_dir, "models")
other_data_dir = joinpath(experiment_dir, "other-data")
pt_diagnostics_dir = joinpath(plots_dir, "pt_diagnostics")

if !isdir(experiment_dir)
    mkpath(experiment_dir)
    println("Created experiment directory: $experiment_dir")
end
for dir in [plots_dir, data_dir, models_dir, other_data_dir, pt_diagnostics_dir]
    if !isdir(dir)
        mkpath(dir)
    end
end

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
save_simulation_parameters(params_file, params)

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

# Create test set from remaining data
raw_data["test"] = convert_forces_to_svector.(raw_data["test"][test_indices[1:TEST_THINNING:end]])

#%%
# ACE model for Silicon (single element)
model = ace1_model(elements = ACE_ELEMENTS,
                   rcut = ACE_RCUT,
                   order = ACE_ORDER,        # body-order - 1
                   totaldegree = ACE_TOTALDEGREE);

Psqrt = _make_prior(model, ACE_PRIOR_ORDER, nothing) # square root of prior precision matrix

my_weights() = Dict("default"=>Dict("E"=>WEIGHT_ENERGY, "F"=>WEIGHT_FORCES, "V"=>WEIGHT_VIRIAL))
data = Dict{String, Any}()
row_mappings = Dict{String, Any}()
n_data = Dict{String, Int}()
for s in ["test"]
    data[s] = make_atoms_data(raw_data[s], model; 
                            energy_key = "energy", 
                            force_key = "forces", 
                            virial_key = nothing, 
                            weights = my_weights())
    row_mappings[s] = row_mapping(data[s], model)
    n_data[s] = length(raw_data[s])
end

true_energies = [at.system_data.energy for at in raw_data["test"]]

# Extract true forces once (flatten all force components)
true_forces_all = Float64[]
for d in raw_data["test"]
    true_f = d.atom_data.forces
    for i in 1:length(true_f)
        if isa(true_f[i], Vector)
            # Vector{Float64}
            append!(true_forces_all, true_f[i])
        else
            # SVector{3, Float64}
            append!(true_forces_all, [true_f[i][1], true_f[i][2], true_f[i][3]])
        end
    end
end

# Number of active learning iterations
test_error = []

println("\nDistributed Parallel Tempering Parameters:")
println("  Number of replicas: $N_REPLICAS")
println("  Temperature range: $T_MIN K to $T_MAX K")
println("  Samples per replica: $N_SAMPLES_PT")
println("  Burn-in steps: $BURNIN_PT")
println("  Thinning: $THIN_PT")
println("  Exchange interval: $EXCHANGE_INTERVAL")
println("  Step size: $STEP_SIZE_PT Å")

# Create or load reference model on main process
if REF_MODEL == "SW"
    println("\nCreating Stillinger-Weber potential as reference model")
    ref_model = EmpiricalPotentials.StillingerWeber()
    println("Stillinger-Weber potential created successfully")
    
    # Update params with model info
    params["reference_model"]["type"] = "StillingerWeber"
    params["reference_model"]["description"] = "Empirical potential for Silicon"
else
    # Load ACE model from file
    ref_model_path = REF_MODEL
    if !isabspath(ref_model_path)
        ref_model_path = joinpath(@__DIR__, REF_MODEL)
    end
    println("\nLoading ACE reference model from: $ref_model_path")
    ref_model = ACEpotentials.load_model(ref_model_path)[1]
    println("ACE reference model loaded successfully")
    
    # Save a copy of the ACE model in the experiment directory
    ref_model_copy_path = joinpath(experiment_dir, "ref_" * basename(ref_model_path))
    println("Saving copy of ACE reference model to: $ref_model_copy_path")
    ACEpotentials.save_model(ref_model, ref_model_copy_path)
    println("ACE reference model copy saved successfully")
    
    # Update params with model info
    params["reference_model"]["type"] = "ACE"
    params["reference_model"]["model_file_original"] = ref_model_path
    params["reference_model"]["model_file_copy"] = ref_model_copy_path
    params["reference_model"]["description"] = "ACE potential loaded from file"
end

# Save updated parameters with reference model info
save_simulation_parameters(params_file, params)

# Create reference model on all workers
if REF_MODEL == "SW"
    @everywhere ref_model = EmpiricalPotentials.StillingerWeber()
else
    ref_model_path_for_workers = REF_MODEL
    if !isabspath(ref_model_path_for_workers)
        ref_model_path_for_workers = joinpath(@__DIR__, REF_MODEL)
    end
    @everywhere ref_model = ACEpotentials.load_model($ref_model_path_for_workers)[1]
end

# Load model on all workers
@everywhere model_path = nothing  # Will be updated in the loop

# Load initial training configurations from specified file
println("\nLoading initial training candidates from: $INIT_CONFIGS_PATH")
raw_data["init_all"] = ExtXYZ.load(INIT_CONFIGS_PATH)
n_init_total = length(raw_data["init_all"])
println("Total available initial training candidates: $n_init_total")

# Select N_INITIAL_TRAIN configurations for initial training set
if INITIAL_TRAIN_RAND
    println("Randomly selecting initial training samples...")
    init_train_indices = sort(Random.randperm(n_init_total)[1:N_INITIAL_TRAIN])
else
    println("Using first $N_INITIAL_TRAIN configurations (no random selection)...")
    init_train_indices = 1:N_INITIAL_TRAIN
end
println("Selected initial training indices: $init_train_indices")

# Store selected initial configurations
raw_data["init"] = convert_forces_to_svector.(raw_data["init_all"][init_train_indices])

# Initialize training data with selected configurations
raw_data_train = deepcopy(raw_data["init"])
println("Initialized training set with $(length(raw_data_train)) configurations")

for t in 1:N_ACTIVE_ITERATIONS
    println("\n" * "="^70)
    println("Active Learning Iteration $t")
    println("="^70)
    
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
    
    # Set committee if available
    if haskey(result, "committee")
        co_coeffs = result["committee"]
        co_ps_vec = [ Psqrt \ co_coeffs[:,i] for i in 1:size(co_coeffs,2) ]
        set_committee!(model, co_ps_vec)
    end

    # Save the fitted model
    model_filename = joinpath(models_dir, "model_iter_$(lpad(t, 3, '0')).ace")
    ACEpotentials.save_model(model, model_filename)
    println("Saved model: $model_filename")
    
    push!(test_error, comp_potE_error(raw_data["test"], model))

    p_energy = plot_energy_comparison(raw_data["test"], model,
                               joinpath(plots_dir, "energy_scatter_iter_$(lpad(t, 3, '0')).png"))
    
    p_forces = plot_forces_comparison(raw_data["test"], model, 
                                   joinpath(plots_dir, "forces_scatter_iter_$(lpad(t, 3, '0')).png"))

    p_energy_train = plot_energy_comparison(raw_data_train, model,
                               joinpath(plots_dir, "train_energy_scatter_iter_$(lpad(t, 3, '0')).png"))
    
    p_forces_train = plot_forces_comparison(raw_data_train, model, 
                                   joinpath(plots_dir, "train_forces_scatter_iter_$(lpad(t, 3, '0')).png"))

    #%% Query next candidate using selected query function
    println("\nUsing query function: $QUERY_FUNCTION")
    
    if QUERY_FUNCTION == "TSSID"
        selected_system = query_TSSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                                      plots_dir=plots_dir,
                                      other_data_dir=other_data_dir,
                                      pt_diagnostics_dir=pt_diagnostics_dir,
                                      t=t,
                                      N_REPLICAS=N_REPLICAS,
                                      T_MIN=T_MIN,
                                      T_MAX=T_MAX,
                                      N_SAMPLES_PT=N_SAMPLES_PT,
                                      BURNIN_PT=BURNIN_PT,
                                      THIN_PT=THIN_PT,
                                      EXCHANGE_INTERVAL=EXCHANGE_INTERVAL,
                                      STEP_SIZE_PT=STEP_SIZE_PT,
                                      R_CUT=R_CUT)
    elseif QUERY_FUNCTION == "ABSID"
        selected_system = query_ABSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                                      plots_dir=plots_dir,
                                      other_data_dir=other_data_dir,
                                      pt_diagnostics_dir=pt_diagnostics_dir,
                                      t=t,
                                      N_REPLICAS=N_REPLICAS,
                                      T_MIN=T_MIN,
                                      T_MAX=T_MAX,
                                      N_SAMPLES_PT=N_SAMPLES_PT,
                                      BURNIN_PT=BURNIN_PT,
                                      THIN_PT=THIN_PT,
                                      EXCHANGE_INTERVAL=EXCHANGE_INTERVAL,
                                      STEP_SIZE_PT=STEP_SIZE_PT,
                                      R_CUT=R_CUT)
    elseif QUERY_FUNCTION == "US"
        selected_system = query_US(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights)
    elseif QUERY_FUNCTION == "TrainData"
        selected_system = query_TrainData(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                                         train_data_name=TRAIN_DATA_NAME)
    elseif QUERY_FUNCTION == "HAL"
        selected_system = query_HAL(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                                    TAU=TAU,
                                    SIGMA_STOP=SIGMA_STOP,
                                    plots_dir=plots_dir,
                                    t=t,
                                    T_MIN=T_MIN,
                                    N_SAMPLES_PT=N_SAMPLES_PT,
                                    BURNIN_PT=BURNIN_PT,
                                    THIN_PT=THIN_PT,
                                    STEP_SIZE_PT=STEP_SIZE_PT)
    else
        error("Unknown query function: $QUERY_FUNCTION. Options: TSSID, ABSID, US, TrainData, HAL")
    end
    
    push!(raw_data_train, deepcopy(selected_system))

    p_energy_train = plot_energy_comparison(raw_data_train, model,
                               joinpath(plots_dir, "train_energy_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
    
    p_forces_train = plot_forces_comparison(raw_data_train, model, 
                                   joinpath(plots_dir, "train_forces_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
end

#%%
# Plot and save test error evolution
iterations = 1:length(test_error)
p_error = plot(iterations, test_error, 
    xlabel="Active Learning Iteration",
    ylabel="Test RMSE (eV)",
    title="Test Error Evolution During Active Learning (Using ACEfit.BLR)",
    label="Test Error", 
    marker=:circle, 
    linewidth=2,
    markersize=5,
    legend=:topright)

# Save the error evolution plot
error_plot_filename = joinpath(experiment_dir, "test_error_evolution.png")
savefig(p_error, error_plot_filename)
println("Saved error evolution plot: $error_plot_filename")

#%%
# Load large test set
raw_data_large_all = ExtXYZ.load(LARGE_TEST_DATA_PATH)
raw_data["test-large"] = convert_forces_to_svector.(raw_data_large_all[1:LARGE_TEST_DATA_THINNING:end])

large_test_error = Float64[]
for i in 1:length(test_error)
    model_filename = joinpath(models_dir, "model_iter_$(lpad(i, 3, '0')).ace")
    tmodel = ACEpotentials.load_model(model_filename)[1]
    push!(large_test_error, comp_potE_error(raw_data["test-large"], tmodel))
    @printf("Large test set error for model iter %d: %.4f eV\n", i, large_test_error[end])
end

# Plot combined error evolution
iterations = 1:length(test_error)
p_error = plot(iterations, test_error, 
    xlabel="Active Learning Iteration",
    ylabel="Test RMSE (eV)",
    title="Test Error Evolution During Active Learning (Using ACEfit.BLR)",
    label="Test Error (small)", 
    marker=:circle, 
    linewidth=2,
    markersize=5,
    legend=:topright, yscale=:log10)

# Add large test set error
plot!(p_error, iterations, large_test_error,
    label="Test Error (large)",
    marker=:diamond,
    linewidth=2,
    markersize=4)

# Save the combined error evolution plot
error_plot_filename = joinpath(experiment_dir, "test_error_evolution_all.png")
savefig(p_error, error_plot_filename)
println("Saved combined error evolution plot: $error_plot_filename")

println("\n" * "="^70)
println("Active Learning with Distributed Parallel Tempering Complete!")
println("Using ACEfit.BLR for model fitting")
println("="^70)

# Optional: Remove workers if desired
# rmprocs(workers())
