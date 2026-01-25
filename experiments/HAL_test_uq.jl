#=
Active Learning - Silicon with Distributed Parallel Tempering
Training from generated samples with PT sampling
Using Stillinger-Weber potential as reference
=#

#=============================================================================
SIMULATION PARAMETERS - Set all parameters here
=============================================================================#

# Random seeds
const RANDOM_SEED = 2234

# Parallel computing
const N_WORKERS = 4

# Experiment identification
const EXPERIMENT_NAME = "AL_ACE_silicon_dia-primitive-2-large-d"
const OUTPUT_DIR = joinpath(@__DIR__, "results")

# Data paths
const TEST_DATA_PATH = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz"
#"/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/Si-primitive-sw/replica_1_T300K_samples_run1-sw-high-exchanges.xyz"
const LARGE_TEST_DATA_PATH = TEST_DATA_PATH
#"/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/Si-primitive-sw/replica_1_T300K_samples_run1-sw-high-exchanges.xyz"
const TEST_THINNING = 10  # Thinning factor for test data

# Reference model specification
const REF_MODEL = "../models/Si_ref_model.json" #"SW"  # Use "SW" for Stillinger-Weber or provide path to ACE model file (e.g., "../models/Si_ref_model.json")

# Initial training set
const N_INITIAL_TRAIN = 10  # Number of initial training samples

# Active learning parameters
const N_ACTIVE_ITERATIONS = 5  # Number of active learning iterations

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

# Bayesian Ridge regression parameters
const BR_ALPHA_1 = 1e-6
const BR_ALPHA_2 = 1e-6
const BR_LAMBDA_1 = 1e-6
const BR_LAMBDA_2 = 1e-6
const BR_TOL = 1e-6
const BR_MAX_ITER = 300

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

# HAL parameters
const TAU = 0.1  # Uncertainty scaling factor for HAL
const SIGMA_STOP = 0.5 # eV/atom - stopping criterion for HAL based on predicted uncertainty


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
using PythonCall
using ACESIDopt: comp_potE_error

using ACESIDopt: expected_red_variance, row_mapping, pred_variance
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
println("Using Stillinger-Weber Potential as Reference")
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
tsmodels_dir = joinpath(experiment_dir, "models-ts")
pt_diagnostics_dir = joinpath(plots_dir, "pt_diagnostics")

if !isdir(experiment_dir)
    mkpath(experiment_dir)
    println("Created experiment directory: $experiment_dir")
end
for dir in [plots_dir, data_dir, models_dir, tsmodels_dir, pt_diagnostics_dir]
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
        "test_thinning" => TEST_THINNING
    ),
    "training" => Dict(
        "n_initial_train" => N_INITIAL_TRAIN,
        "n_active_iterations" => N_ACTIVE_ITERATIONS
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
    "bayesian_ridge" => Dict(
        "alpha_1" => BR_ALPHA_1,
        "alpha_2" => BR_ALPHA_2,
        "lambda_1" => BR_LAMBDA_1,
        "lambda_2" => BR_LAMBDA_2,
        "tol" => BR_TOL,
        "max_iter" => BR_MAX_ITER
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
raw_data["test-all"] = ExtXYZ.load(TEST_DATA_PATH)

using ACESIDopt: convert_forces_to_svector

# Subsample data points for initial training set
n_total = length(raw_data["test-all"])
println("Total available samples: $n_total")
train_indices = sort(Random.randperm(n_total)[1:N_INITIAL_TRAIN])
test_indices = setdiff(1:n_total, train_indices)

println("Selected training indices: $train_indices")
println("Remaining test samples: $(length(test_indices))")

# Create training set from subsampled data
raw_data["train"]= convert_forces_to_svector.(raw_data["test-all"][train_indices])

# Create test set from remaining data
raw_data["test"] = convert_forces_to_svector.(raw_data["test-all"][test_indices[1:TEST_THINNING:end]])


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
for s in ["test", "train"]
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
#%%
raw_data_train = deepcopy(raw_data["train"][1:3])


model = ace1_model(elements = [:Si,],
                   Eref = [:Si => -158.54496821],
                   order = 3,
                   totaldegree = 8);
# solver = ACEfit.BLR(committee_size = 1000, factorization = :svd)
# g = acefit!(raw_data_train,  model;
#         solver = solver,
#         energy_key = "energy", force_key = "forces",
#         verbose = false);


mm_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
data_train = make_atoms_data(raw_data_train, model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = mm_weights())
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Psqrt =  I #_make_prior(model, ACE_PRIOR_ORDER, nothing)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 
Yw_train = W_train .* Y_train

#
#
#
using LinearAlgebra
using ACEfit: bayesian_linear_regression
solver = ACEfit.BLR(committee_size = 10000, factorization = :svd)
result1 = bayesian_linear_regression(Awp_train, Yw_train; solver.kwargs..., ret_covar = true)
result1["covar"]
coeffs = Psqrt \ result1["C"]   
# dispatch setting of parameters 
ACEpotentials.Models.set_linear_parameters!(model, coeffs)
if haskey(result1, "committee")
    co_coeffs = result1["committee"]
    co_ps_vec = [ Psqrt \ co_coeffs[:,i] for i in 1:size(co_coeffs,2) ]
    set_committee!(model, co_ps_vec)
end

# If covar_sqrt is needed:
# X ,Y = Awp_train, Yw_train
# var_0, var_e = result1["var_0"], result1["var_e"]
# elapsed = @elapsed U, S, V = svd!(X; full = false, alg = LinearAlgebra.QRIteration())
# @info "SVD completed after $(elapsed/60) minutes"
# UT_Y = transpose(U) * Y
# UT_Y[1:length(S)] .*= var_0 .* S ./ (var_0 .* S .* S .+ var_e)
# c = V * UT_Y[1:length(S)]
# covar_diag = 1.0 ./ (S .* S / var_e .+ 1.0 / var_0)
# covar_sqrt = V * Diagonal(sqrt.(covar_diag)) * transpose(V) 
# # covar_sqrt*transpose(covar_sqrt) - result1["covar"]
# fieldnames(typeof(model.model.Vref.E0))


function predictive_variance(x::Vector, covar::Matrix; var_e=0.0)
    return dot(x, covar * x) + var_e
end
function predictive_variance(x::Vector, covar::Matrix, Psqrt;  var_e=0.0)
    xt = Psqrt \ x
    return predictive_variance(xt, covar; var_e=var_e)
end
function predictive_variance(model, atom::AtomsBase.AbstractSystem, covar::Matrix; Psqrt=I, var_e=0.0)
    # Check wheter this should be indeed the sum or variance
    x = sum(site_descriptors(atom, model))
    return predictive_variance(x, covar, Psqrt; var_e=var_e)
end


model_ref_energies = []
model_energies = []
model_energy_error = []
model_std = []
model_std2 = []
for atoms in raw_data["test"] 
    ene, co_ene = @committee potential_energy(atoms, model)
    e_ref = atoms.system_data.energy
    push!(model_energy_error, abs(ustrip(ene) - ustrip(e_ref))/length(atoms))
    push!(model_ref_energies, ustrip(e_ref/length(atoms)))
    push!(model_energies, ustrip(ene/length(atoms)))
    push!(model_std, ustrip(std(co_ene/length(atoms))))
    push!(model_std2, sqrt(predictive_variance(model, atoms, result1["covar"]; Psqrt=Psqrt, var_e=0))/length(atoms))
end

model_std./model_std2
model_std./model_std2

using ACEpotentials: site_descriptors
descriptors = [site_descriptors(atoms, model) for atoms in raw_data["test"]]

for i=1:length(raw_data["test"])
    @assert (model_energies[i]-sum(coeffs.*mean(descriptors[i]))-model.model.Vref.E0[atomic_number(:Si)]) < 1e-10
end
typeof(raw_data["test"][1]) <: AtomsBase.AbstractSystem

# Plot model uncertainty vs model error
p0 = scatter(model_std2, model_energy_error,
    xlabel="Model Std Dev (eV/atom)",
    ylabel="Model Energy Error (eV/atom)",
    title="Model Uncertainty2 vs Energy Error",
    label="Test Data",
    marker=:circle,
    markersize=5,
    legend=:topleft)
savefig(p0, joinpath(plots_dir, "uncertainty2_vs_error.png"))
println("Saved uncertainty2 vs error plot")



# Plot model energies vs reference energies
p1 = scatter(model_ref_energies, model_energies,
    xlabel="Reference Energy (eV/atom)",
    ylabel="Model Energy (eV/atom)",
    title="Model vs Reference Energies",
    label="Test Data",
    marker=:circle,
    markersize=5,
    legend=:topleft)
# Add diagonal line for perfect prediction
min_e = min(minimum(model_ref_energies), minimum(model_energies))
max_e = max(maximum(model_ref_energies), maximum(model_energies))
plot!(p1, [min_e, max_e], [min_e, max_e], label="Perfect prediction", linewidth=2, linestyle=:dash, color=:red)
savefig(p1, joinpath(plots_dir, "model_vs_ref_energies.png"))
println("Saved model vs reference energies plot")

# Plot model uncertainty vs model error
p2 = scatter(model_std, model_energy_error,
    xlabel="Model Std Dev (eV/atom)",
    ylabel="Model Energy Error (eV/atom)",
    title="Model Uncertainty (Committee) vs Energy Error",
    label="Test Data",
    marker=:circle,
    markersize=5,
    legend=:topleft)
savefig(p2, joinpath(plots_dir, "uncertainty_vs_error.png"))
println("Saved uncertainty vs error plot")

# Plot committee std vs covariance-based std
p3 = scatter(model_std, model_std2,
    xlabel="Committee Std Dev (eV/atom)",
    ylabel="Covariance-based Std Dev (eV/atom)",
    title="Committee vs Covariance Uncertainty",
    label="Test Data",
    marker=:circle,
    markersize=5,
    legend=:topleft)
# Add diagonal line for perfect agreement
min_s = min(minimum(model_std), minimum(model_std2))
max_s = max(maximum(model_std), maximum(model_std2))
plot!(p3, [min_s, max_s], [min_s, max_s], label="Perfect agreement", linewidth=2, linestyle=:dash, color=:red)
savefig(p3, joinpath(plots_dir, "committee_vs_covariance_uncertainty.png"))
println("Saved committee vs covariance uncertainty plot")


#%%#################################################



#%%
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

    sklearn_linear = pyimport("sklearn.linear_model")
    BayesianRidge = sklearn_linear.BayesianRidge
    br_model = BayesianRidge(fit_intercept=false, 
                            alpha_1=BR_ALPHA_1, 
                            alpha_2=BR_ALPHA_2, 
                            lambda_1=BR_LAMBDA_1, 
                            lambda_2=BR_LAMBDA_2, 
                            tol=BR_TOL, 
                            max_iter=BR_MAX_ITER)
    # fit model and export relevant parameters
    br_model.fit(Awp_train, Yw_train)
    Σ = pyconvert(Matrix, br_model.sigma_)
    coef_tilde = pyconvert(Array, br_model.coef_)
    α = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
    coef = Psqrt \ coef_tilde

    Σt = inv(Psqrt) * Σ * inv(Psqrt)
    Σt_chol, jitter = cholesky_with_jitter((Σt + transpose(Σt))/2; max_jitter_fraction=MAX_JITTER_FRACTION) 
    coef_rand = coef + Σt_chol.L * randn(size(coef))

    ts_model = deepcopy(model)
    ACEpotentials.Models.set_linear_parameters!(ts_model, coef_rand)
    ACEpotentials.Models.set_linear_parameters!(model, coef)

    # Save the fitted model
    model_filename = joinpath(models_dir, "model_iter_$(lpad(t, 3, '0')).ace")
    ACEpotentials.save_model(model, model_filename)
    println("Saved model: $model_filename")
    # Save the TS model
    ts_model_filename = joinpath(tsmodels_dir, "ts_model_iter_$(lpad(t, 3, '0')).ace")
    ACEpotentials.save_model(ts_model, ts_model_filename)
    println("Saved TS model: $ts_model_filename")

    # Load model on all workers
    @everywhere ts_model_filename = $ts_model_filename
    @everywhere ts_model = ACEpotentials.load_model(ts_model_filename)[1]

    push!(test_error, comp_potE_error(raw_data["test"], model))

    p_energy = plot_energy_comparison(raw_data["test"], model,
                               joinpath(plots_dir, "energy_scatter_iter_$(lpad(t, 3, '0')).png"))
    
    p_forces = plot_forces_comparison(raw_data["test"], model, 
                                   joinpath(plots_dir, "forces_scatter_iter_$(lpad(t, 3, '0')).png"))

    p_energy_train = plot_energy_comparison(raw_data_train, model,
                               joinpath(plots_dir, "train_energy_scatter_iter_$(lpad(t, 3, '0')).png"))
    
    p_forces_train = plot_forces_comparison(raw_data_train, model, 
                                   joinpath(plots_dir, "train_forces_scatter_iter_$(lpad(t, 3, '0')).png"))

    #%% Run distributed parallel tempering for surrogate and candidate sampling in parallel
    println("\nRunning Distributed Parallel Tempering for Surrogate Samples...")
    
    # Sample initial systems for surrogate PT
    n_available = length(raw_data_train)
    sampled_indices_sur = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_sur) < N_REPLICAS
        # If not enough unique samples, repeat some
        while length(sampled_indices_sur) < N_REPLICAS
            push!(sampled_indices_sur, rand(1:n_available))
        end
    end
    initial_systems_sur = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_sur]
    
    # Sample initial systems for candidate PT
    sampled_indices_cand = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_cand) < N_REPLICAS
        while length(sampled_indices_cand) < N_REPLICAS
            push!(sampled_indices_cand, rand(1:n_available))
        end
    end
    initial_systems_cand = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_cand]
    
    # Create samplers
    sampler_sur = RWMCSampler(step_size=STEP_SIZE_PT)
    sampler_cand = RWMCSampler(step_size=STEP_SIZE_PT)
    
    # Run surrogate and candidate PT in parallel using @sync
    sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = nothing, nothing, nothing, nothing, nothing
    cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = nothing, nothing, nothing, nothing, nothing
    
    @sync begin
        @async begin
            println("  Starting surrogate PT...")
            sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = 
                run_parallel_tempering_distributed(
                    sampler_sur, initial_systems_sur, ts_model, N_REPLICAS, T_MIN, T_MAX;
                    n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                )
            println("  Surrogate PT complete!")
        end
        
        @async begin
            println("  Starting candidate PT...")
            cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = 
                run_parallel_tempering_distributed(
                    sampler_cand, initial_systems_cand, ts_model, N_REPLICAS, T_MIN, T_MAX;
                    n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                )
            println("  Candidate PT complete!")
        end
    end
    
    # Generate diagnostics for surrogate PT
    println("\nGenerating surrogate PT diagnostics...")
    sur_diag_dir = joinpath(pt_diagnostics_dir, "surrogate")
    mkpath(sur_diag_dir)
    generate_ptd_diagnostics_and_log(sur_replicas, sur_temperatures, sur_mcmc_rates, 
                                     sur_exchange_rates, sur_trajs, sur_diag_dir, t, R_CUT * u"Å")
    
    # Generate diagnostics for candidate PT
    println("\nGenerating candidate PT diagnostics...")
    cand_diag_dir = joinpath(pt_diagnostics_dir, "candidate")
    mkpath(cand_diag_dir)
    generate_ptd_diagnostics_and_log(cand_replicas, cand_temperatures, cand_mcmc_rates, 
                                     cand_exchange_rates, cand_trajs, cand_diag_dir, t, R_CUT * u"Å")
    
    # Extract samples from target temperature (lowest temperature, replica 1)
    sur_samples = sur_replicas[1]
    cand_samples = cand_replicas[1]
    
    # Add energies to surrogate samples and energy+forces to candidate samples
    raw_data["tsur-nf"] = [at for at in add_energy(sur_samples, model)]
    raw_data["tcand"] = [at for at in add_energy_forces(cand_samples, model)]
    
    data["tsur-nf"] = make_atoms_data([at for at in raw_data["tsur-nf"]], model; 
                                energy_key = "energy", 
                                force_key = nothing, 
                                virial_key = nothing, 
                                weights = my_weights())
    row_mappings["tsur-nf"] = row_mapping(data["tsur-nf"], model)
    n_data["tsur-nf"] = length(raw_data["tsur-nf"])
    
    data["tcand"] = make_atoms_data([at for at in raw_data["tcand"]], model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = my_weights())
    row_mappings["tcand"] = row_mapping(data["tcand"], model)
    n_data["tcand"] = length(raw_data["tcand"])
    
    A = Dict{String, Matrix{Float64}}()
    Awp = Dict{String, Matrix{Float64}}()
    Y = Dict{String, Vector{Float64}}()
    Yw = Dict{String, Vector{Float64}}()
    W = Dict{String, Vector{Float64}}()
    # actual assembly of the least square system 
    for s in ["tsur-nf", "tcand"]
        A[s], Y[s], W[s] = ACEfit.assemble(data[s], model)
        Awp[s] = Diagonal(W[s]) * (A[s] / Psqrt) 
        Yw[s] = W[s] .* Y[s]
    end

    p_var = [pred_variance(Σ, Awp["tcand"][i,:], α) for i in 1:n_data["tcand"]] 
    p_var_mean = mean(p_var)
    
    exp_red = [(p_var[j] >= p_var_mean) ? expected_red_variance(Σ, Awp["tsur-nf"],  Awp["tcand"][j,:], α) : -1 for j in 1:n_data["tcand"] ] 
    idx = argmax(exp_red)

    println("\nSelected candidate $idx with max expected variance reduction: $(exp_red[idx])")

    # Use Stillinger-Weber potential for ground truth evaluation
    println("Computing energy and forces with Stillinger-Weber reference potential...")
    
    # Compute energy and forces with Stillinger-Weber model
    selected_system = raw_data["tcand"][idx]
    ref_energy = potential_energy(selected_system, ref_model)
    ref_forces = forces(selected_system, ref_model)
    
    add_energy_forces(raw_data["tcand"][idx], ref_energy, ref_forces)
    push!(raw_data_train, deepcopy(raw_data["tcand"][idx]))

    p_energy_train = plot_energy_comparison(raw_data_train, ts_model,
                               joinpath(plots_dir, "train_energy_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
    
    p_forces_train = plot_forces_comparison(raw_data_train, ts_model, 
                                   joinpath(plots_dir, "train_forces_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
end

#%%
# Plot and save test error evolution
iterations = 1:length(test_error)
p_error = plot(iterations, test_error, 
    xlabel="Active Learning Iteration",
    ylabel="Test RMSE (eV)",
    title="Test Error Evolution During Active Learning (Silicon PTD with SW)",
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
raw_data["test-large"] = ExtXYZ.load(LARGE_TEST_DATA_PATH)

large_test_error = Float64[]
for i in 1:length(test_error)
    model_filename = joinpath(models_dir, "model_iter_$(lpad(i, 3, '0')).ace")
    model = ACEpotentials.load_model(model_filename)[1]
    push!(large_test_error, comp_potE_error(raw_data["test-large"], model))
    @printf("Large test set error for model iter %d: %.4f eV\n", i, large_test_error[end])
end

# Plot combined error evolution
iterations = 1:length(test_error)
p_error = plot(iterations, test_error, 
    xlabel="Active Learning Iteration",
    ylabel="Test RMSE (eV)",
    title="Test Error Evolution During Active Learning (Silicon PTD with SW)",
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
println("Using Stillinger-Weber potential as reference")
println("="^70)

# Optional: Remove workers if desired
# rmprocs(workers())
