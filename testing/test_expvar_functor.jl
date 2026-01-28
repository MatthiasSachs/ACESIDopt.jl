#=
Minimal test for ExpVarFunctor
Tests gradient-based optimization of expected variance reduction
=#

using ACESIDopt
using ACESIDopt: ExpVarFunctor, expected_red_variance, add_energy_forces, convert_forces_to_svector
using ACEpotentials
using ACEpotentials: make_atoms_data, _make_prior
using ACEfit
using ACEfit: bayesian_linear_regression
using AtomsCalculators: potential_energy, forces
using LinearAlgebra: Diagonal
using ExtXYZ
using Random
using FiniteDifferences

println("="^70)
println("Testing ExpVarFunctor for Expected Variance Reduction")
println("="^70)

# Set random seed for reproducibility
Random.seed!(42)

# Load test data
data_path = joinpath(@__DIR__, "../data/Si-diamond-primitive-8atoms.xyz")
println("\nLoading data from: $data_path")
raw_data = ExtXYZ.load(data_path)

# Convert and subsample
raw_data = convert_forces_to_svector.(raw_data[1:5])
println("Loaded $(length(raw_data)) configurations")

# Create ACE model
println("\nCreating ACE model...")
model = ace1_model(elements = [:Si],
                   rcut = 5.5,
                   order = 3,
                   totaldegree = 6)

# Prior matrix
Psqrt = _make_prior(model, 4, nothing)

# Weights
my_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))

# Prepare training data
println("\nFitting Bayesian linear regression model...")
data_train = make_atoms_data(raw_data, model; 
                            energy_key = "energy", 
                            force_key = "forces", 
                            virial_key = nothing, 
                            weights = my_weights())

A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Awp_train = Diagonal(W_train) * (A_train / Psqrt)
Yw_train = W_train .* Y_train

# Fit BLR model
solver = ACEfit.BLR(committee_size = 100, factorization = :svd)
result = bayesian_linear_regression(Awp_train, Yw_train; solver.kwargs..., ret_covar = true)

Σ = result["covar"]
coef_tilde = result["C"]
α = result["var_e"]
coef = Psqrt \ coef_tilde

# Set model parameters
ACEpotentials.Models.set_linear_parameters!(model, coef)

println("BLR fit complete:")
println("  Noise precision α = $α")
println("  Covariance matrix size: $(size(Σ))")

# Create test atomic system for optimization
at = deepcopy(raw_data[1])
println("\nTest system:")
println("  Number of atoms: $(length(at))")
println("  Initial energy: $(at.system_data.energy) eV")

# Create candidate data (use remaining data)
raw_data_candidates = raw_data[2:end]
data_candidates = make_atoms_data(raw_data_candidates, model; 
                                 energy_key = "energy", 
                                 force_key = nothing, 
                                 virial_key = nothing, 
                                 weights = my_weights())
A_cand, _, W_cand = ACEfit.assemble(data_candidates, model)
Awp_cand = Diagonal(W_cand) * (A_cand / Psqrt)

# Create ExpVarFunctor
println("\nCreating ExpVarFunctor...")
g = ExpVarFunctor(Σ, Awp_cand, α, at, model)

# Test evaluation
x0 = ACESIDopt.mget_positions(at)
println("\nTesting functor evaluation...")
println("  Position vector length: $(length(x0))")

g_val0 = g(x0)
println("  Initial expected variance reduction: $g_val0")

# Test with perturbed positions
x_pert = x0 .+ 0.1 * randn(length(x0))
g_val_pert = g(x_pert)
println("  Perturbed expected variance reduction: $g_val_pert")

# Compute gradient using finite differences
println("\nComputing gradient via finite differences...")
fdm = central_fdm(5, 1)
grad_g = FiniteDifferences.grad(fdm, g, x0)[1]
println("  Gradient norm: $(norm(grad_g))")
println("  Max gradient component: $(maximum(abs.(grad_g)))")

# Test gradient descent step
println("\nTesting gradient ascent step...")
step_size = 0.01
x_new = x0 .+ step_size * grad_g  # ascent to maximize variance reduction
g_val_new = g(x_new)
println("  New expected variance reduction: $g_val_new")
println("  Change: $(g_val_new - g_val0)")

if g_val_new > g_val0
    println("  ✓ Gradient ascent increased expected variance reduction")
else
    println("  ⚠ Gradient ascent did not increase expected variance reduction")
    println("    (May need smaller step size or different optimization strategy)")
end

println("\n" * "="^70)
println("ExpVarFunctor test complete!")
println("="^70)
