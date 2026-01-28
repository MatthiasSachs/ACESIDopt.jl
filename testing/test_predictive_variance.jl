#=
Test script: Fit ACE model to test dataset and evaluate predictive_variance
Following the same construction and fitting steps as active_learning_loop_BLR-random_start.jl
=#

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: save_simulation_parameters, load_simulation_parameters, comp_potE_MAE_RMSE
using ACEpotentials: _make_prior
using ACEpotentials: make_atoms_data, assess_dataset, default_weights, AtomsData
using LinearAlgebra: Diagonal, I, inv
using ACEfit
using ACEfit: bayesian_linear_regression
using ACEpotentials: set_committee!
using ACESIDopt: expected_red_variance, row_mapping, pred_variance, predictive_variance
using ACESIDopt.QueryModels: query_US
using Random
using ExtXYZ
using Unitful
using AtomsCalculators: potential_energy, forces
using EmpiricalPotentials
using ACESIDopt: convert_forces_to_svector, add_energy_forces
using Statistics

println("="^70)
println("Test: Fit ACE model and evaluate predictive_variance")
println("="^70)

# Set random seed
Random.seed!(2234)

#=============================================================================
PARAMETERS (following active_learning_loop_BLR-random_start.jl)
=============================================================================#

# Data path
const TEST_DATA_PATH = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_dia-primitive-2-high-K1200/data/replica_1_samples.xyz"
const TEST_THINNING = 10

# Reference model specification
const REF_MODEL = "../models/Si_ref_model-small-2.json"

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

# Number of training samples to use
const N_TRAIN = 50

# Number of test points for predictive variance evaluation
const N_TEST_POINTS = 10

#=============================================================================
LOAD DATA
=============================================================================#

println("\nLoading test data from: $TEST_DATA_PATH")
raw_data_all = ExtXYZ.load(TEST_DATA_PATH)
n_total = length(raw_data_all)
println("Total available samples: $n_total")

# Convert forces to SVector
raw_data_all = convert_forces_to_svector.(raw_data_all)

# Subsample: first N_TRAIN for training, rest (thinned) for testing
raw_data_train = raw_data_all[1:N_TRAIN]
raw_data_test = raw_data_all[(N_TRAIN+1):TEST_THINNING:end]

println("Training samples: $(length(raw_data_train))")
println("Test samples: $(length(raw_data_test))")

#=============================================================================
LOAD REFERENCE MODEL
=============================================================================#

ref_model_path = REF_MODEL
if !isabspath(ref_model_path)
    ref_model_path = joinpath(@__DIR__, REF_MODEL)
end
println("\nLoading ACE reference model from: $ref_model_path")
ref_model = ACEpotentials.load_model(ref_model_path)[1]
println("ACE reference model loaded successfully")

#=============================================================================
CREATE ACE MODEL
=============================================================================#

println("\nCreating ACE model...")
model = ace1_model(elements = ACE_ELEMENTS,
                   rcut = ACE_RCUT,
                   order = ACE_ORDER,
                   totaldegree = ACE_TOTALDEGREE)

Psqrt = _make_prior(model, ACE_PRIOR_ORDER, nothing) # square root of prior precision matrix
println("ACE model created successfully")

my_weights() = Dict("default"=>Dict("E"=>WEIGHT_ENERGY, "F"=>WEIGHT_FORCES, "V"=>WEIGHT_VIRIAL))

#=============================================================================
ASSEMBLE TRAINING DATA
=============================================================================#

println("\nAssembling training data...")
data_train = make_atoms_data(raw_data_train, model; 
                            energy_key = "energy", 
                            force_key = "forces", 
                            virial_key = nothing, 
                            weights = my_weights())

A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 
Yw_train = W_train .* Y_train

println("Training data assembled:")
println("  A_train size: $(size(A_train))")
println("  Y_train size: $(size(Y_train))")

#=============================================================================
FIT MODEL USING ACEfit.BLR
=============================================================================#

println("\nFitting ACE model using BLR...")
solver = ACEfit.BLR(committee_size = COMMITTEE_SIZE, factorization = FACTORIZATION)
result = bayesian_linear_regression(Awp_train, Yw_train; solver.kwargs..., ret_covar = true)

# Extract parameters
Σ = result["covar"]
coef_tilde = result["C"]
α = result["var_e"]  # noise precision (variance of observation noise)

println("Model fitted successfully!")
println("  Covariance matrix Σ size: $(size(Σ))")
println("  Coefficients size: $(size(coef_tilde))")
println("  Noise precision α: $α")
println("  Result keys: $(keys(result))")

# Set linear parameters for the model (proper way)
coef = Psqrt \ coef_tilde  # transform back to original space
ACEpotentials.Models.set_linear_parameters!(model, coef)

# Set committee if available
if haskey(result, "committee")
    co_coeffs = result["committee"]
    co_ps_vec = [ Psqrt \ co_coeffs[:,i] for i in 1:size(co_coeffs,2) ]
    set_committee!(model, co_ps_vec)
    println("Model parameters set (including committee of $(size(co_coeffs,2)) members)")
else
    println("Model parameters set (committee not available in result)")
end

#=============================================================================
EVALUATE MODEL ON TRAINING DATA
=============================================================================#

println("\n" * "="^70)
println("Evaluating model on training data")
println("="^70)

# Compute predictions (strip units)
train_pred_energies = [ustrip(potential_energy(at, model)) for at in raw_data_train]
train_true_energies = [at.system_data.energy for at in raw_data_train]

# Compute errors
train_energy_errors = train_pred_energies .- train_true_energies
train_rmse = sqrt(sum(train_energy_errors.^2) / length(train_energy_errors))
train_mae = sum(abs.(train_energy_errors)) / length(train_energy_errors)

println("Training energy errors:")
println("  RMSE: $(round(train_rmse, digits=6)) eV")
println("  MAE: $(round(train_mae, digits=6)) eV")

#=============================================================================
EVALUATE MODEL ON TEST DATA
=============================================================================#

println("\n" * "="^70)
println("Evaluating model on test data")
println("="^70)

# Compute predictions (strip units)
test_pred_energies = [ustrip(potential_energy(at, model)) for at in raw_data_test]
test_true_energies = [at.system_data.energy for at in raw_data_test]

# Compute errors
test_energy_errors = test_pred_energies .- test_true_energies
test_rmse = sqrt(sum(test_energy_errors.^2) / length(test_energy_errors))
test_mae = sum(abs.(test_energy_errors)) / length(test_energy_errors)

println("Test energy errors:")
println("  RMSE: $(round(test_rmse, digits=6)) eV")
println("  MAE: $(round(test_mae, digits=6)) eV")

#=============================================================================
GENERATE RANDOM TEST POINTS USING query_US
=============================================================================#

println("\n" * "="^70)
println("Generating random test points using query_US")
println("="^70)

random_test_points = []
for i in 1:N_TEST_POINTS
    println("\nGenerating random test point $i/$N_TEST_POINTS...")
    test_point = query_US(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights)
    push!(random_test_points, test_point)
    
    # Show some info
    E_ref = test_point.system_data.energy
    E_pred = ustrip(potential_energy(test_point, model))
    println("  Reference energy: $(round(E_ref, digits=4)) eV")
    println("  Predicted energy: $(round(E_pred, digits=4)) eV")
    println("  Error: $(round(abs(E_ref - E_pred), digits=4)) eV")
end

#=============================================================================
EVALUATE PREDICTIVE_VARIANCE AT TEST POINTS
=============================================================================#

println("\n" * "="^70)
println("Evaluating predictive_variance at test points")
println("="^70)

println("\nUsing training data samples:")
for i in 1:min(5, length(raw_data_train))
    at = raw_data_train[i]
    pred_var = predictive_variance(model, at, Σ; Psqrt=Psqrt)
    pred_std = sqrt(pred_var)
    println("  Training sample $i: pred_std = $(round(pred_std, digits=4)) eV")
end

println("\nUsing test data samples:")
for i in 1:min(5, length(raw_data_test))
    at = raw_data_test[i]
    pred_var = predictive_variance(model, at, Σ; Psqrt=Psqrt)
    pred_std = sqrt(pred_var)
    println("  Test sample $i: pred_std = $(round(pred_std, digits=4)) eV")
end

println("\nUsing random points generated by query_US:")
for i in 1:length(random_test_points)
    at = random_test_points[i]
    pred_var = predictive_variance(model, at, Σ; Psqrt=Psqrt)
    pred_std = sqrt(pred_var)
    E_ref = at.system_data.energy
    E_pred = ustrip(potential_energy(at, model))
    E_error = abs(E_ref - E_pred)
    println("  Random point $i:")
    println("    pred_std = $(round(pred_std, digits=4)) eV")
    println("    energy_error = $(round(E_error, digits=4)) eV")
    println("    ratio (uncertainty/error) = $(round(pred_std / max(E_error, 1e-10), digits=2))")
end

#=============================================================================
SUMMARY STATISTICS
=============================================================================#

println("\n" * "="^70)
println("Summary Statistics")
println("="^70)

# Compute predictive variance for all test points
all_test_pred_vars = [predictive_variance(model, at, Σ; Psqrt=Psqrt) for at in raw_data_test]
all_test_pred_stds = sqrt.(max.(all_test_pred_vars, 0.0))  # Clamp negative values to 0

# Compute predictive variance for random points
random_pred_vars = [predictive_variance(model, at, Σ; Psqrt=Psqrt) for at in random_test_points]
random_pred_stds = sqrt.(max.(random_pred_vars, 0.0))  # Clamp negative values to 0

# Compute energy errors for random points
random_energy_errors = [abs(ustrip(potential_energy(at, model)) - at.system_data.energy) for at in random_test_points]

println("\nTest data predictive std:")
println("  Mean: $(round(mean(all_test_pred_stds), digits=4)) eV")
println("  Std: $(round(std(all_test_pred_stds), digits=4)) eV")
println("  Min: $(round(minimum(all_test_pred_stds), digits=4)) eV")
println("  Max: $(round(maximum(all_test_pred_stds), digits=4)) eV")

println("\nRandom points predictive std:")
println("  Mean: $(round(mean(random_pred_stds), digits=4)) eV")
println("  Std: $(round(std(random_pred_stds), digits=4)) eV")
println("  Min: $(round(minimum(random_pred_stds), digits=4)) eV")
println("  Max: $(round(maximum(random_pred_stds), digits=4)) eV")

println("\nRandom points energy errors:")
println("  Mean: $(round(mean(random_energy_errors), digits=4)) eV")
println("  Std: $(round(std(random_energy_errors), digits=4)) eV")
println("  Min: $(round(minimum(random_energy_errors), digits=4)) eV")
println("  Max: $(round(maximum(random_energy_errors), digits=4)) eV")

println("\nCorrelation between predictive std and energy error (random points):")
if length(random_pred_stds) > 2
    using Statistics: cor
    correlation = cor(random_pred_stds, random_energy_errors)
    println("  Pearson correlation: $(round(correlation, digits=3))")
else
    println("  Not enough points for correlation")
end

println("\n" * "="^70)
println("Test completed successfully!")
println("="^70)
