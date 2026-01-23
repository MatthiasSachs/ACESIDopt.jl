#%%
using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel
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

using ACESIDopt.MSamplers: run_rwmc_sampling
using Random
using LinearAlgebra: cholesky
using ACESIDopt: cholesky_with_jitter, add_energy_forces, add_forces, add_energy
using ACESIDopt: mflexiblesystem, queryASEModel
using AtomsCalculators: potential_energy, forces
using ACESIDopt: plot_forces_comparison, plot_energy_comparison
using ACESIDopt: convert_forces_to_svector

#%%

#=
Test RWMC Sampling with ACE Model for Silicon
=#
#%%
Random.seed!(1234)
experiment_name = "rwmc_silicon_test_1"

# Create experiment directory structure under experiments/
experiment_dir = joinpath(dirname(@__FILE__), experiment_name)
plots_dir = joinpath(experiment_dir, "plots")
mkpath(plots_dir)
models_dir = joinpath(experiment_dir, "models")
mkpath(models_dir)
mcmc_plots_dir = joinpath(plots_dir, "mcmc_plots")
mkpath(mcmc_plots_dir)

# Load Silicon data
raw_data = Dict{String, Any}()
raw_data["train"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_0_train_100frames.xyz")
raw_data["test"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_0_test_100frames.xyz")

#%%
# ACE model for Silicon (single element)
model = ace1_model(elements = [:Si],
                   rcut = 5.5,
                   order = 3,        # body-order - 1
                   totaldegree = 6);

Psqrt = _make_prior(model, 4, nothing) # square root of prior precision matrix

# Prepare training data
raw_data_train = convert_forces_to_svector.(raw_data["train"])

data_train = make_atoms_data(raw_data_train, model; 
                            energy_key = "energy", 
                            force_key = "forces", 
                            virial_key = nothing, 
                            weights = default_weights())

#%%
# Fit the ACE model using Bayesian Ridge Regression
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 
Yw_train = W_train .* Y_train

sklearn_linear = pyimport("sklearn.linear_model")
BayesianRidge = sklearn_linear.BayesianRidge
br_model = BayesianRidge(fit_intercept=false, 
                        alpha_1=1e-6, 
                        alpha_2=1e-6, 
                        lambda_1=1e-6, 
                        lambda_2=1e-6, 
                        tol=1e-6, 
                        max_iter=300)

# Fit model and export relevant parameters
br_model.fit(Awp_train, Yw_train)
Σ = pyconvert(Matrix, br_model.sigma_)
coef_tilde = pyconvert(Array, br_model.coef_)
α = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
coef = Psqrt \ coef_tilde

# Set linear parameters on the model
ACEpotentials.Models.set_linear_parameters!(model, coef)

# Save the fitted model
model_filename = joinpath(models_dir, "fitted_ace_model.ace")
ACEpotentials.save_model(model, model_filename)
println("Saved fitted ACE model: $model_filename")

#%%
# Compute and display training and test errors
train_error = comp_potE_error(raw_data["train"], model)
test_error = comp_potE_error(raw_data["test"], model)

@printf("Training RMSE: %.4f eV\n", train_error)
@printf("Test RMSE: %.4f eV\n", test_error)

# Plot energy comparison for training set
p_energy_train = plot_energy_comparison(raw_data_train, model,
                           joinpath(plots_dir, "train_energy_scatter.png"))

# Plot forces comparison for training set
p_forces_train = plot_forces_comparison(raw_data_train, model, 
                               joinpath(plots_dir, "train_forces_scatter.png"))

# Plot energy comparison for test set
p_energy_test = plot_energy_comparison(raw_data["test"], model,
                           joinpath(plots_dir, "test_energy_scatter.png"))

# Plot forces comparison for test set
p_forces_test = plot_forces_comparison(raw_data["test"], model, 
                               joinpath(plots_dir, "test_forces_scatter.png"))

#%%
# Run RWMC sampling targeting the Gibbs-Boltzmann distribution
rwmc_initial = deepcopy(raw_data["train"][1])
T_rwmc = 300.0  # K
n_rwmc_samples = 1000
step_size_rwmc = 0.05  # Å
burnin_rwmc = 500
thin_rwmc = 10



println("\nRunning RWMC sampling with ACE model...")
println("Temperature: $T_rwmc K")
println("Number of samples: $n_rwmc_samples")
println("Step size: $step_size_rwmc Å")
println("Burn-in: $burnin_rwmc")
println("Thinning: $thin_rwmc")

samples, acceptance_rate, trajectory = run_rwmc_sampling(
    rwmc_initial, model, n_rwmc_samples, T_rwmc;
    step_size=step_size_rwmc, burnin=burnin_rwmc, thin=thin_rwmc
)

@printf("\nRWMC acceptance rate: %.2f%%\n", acceptance_rate * 100)
@printf("Number of samples collected: %d\n", length(samples))

#%%
# Plot RWMC energy trajectory
p_mcmc = plot(1:length(trajectory.energy), ustrip.(trajectory.energy),
    xlabel="MCMC Iteration",
    ylabel="Energy (eV)",
    title="RWMC Energy Trajectory (T = $T_rwmc K)",
    label="Energy",
    linewidth=1.5,
    legend=:topright)
mcmc_filename = joinpath(mcmc_plots_dir, "rwmc_energy_trajectory.png")
savefig(p_mcmc, mcmc_filename)
println("Saved RWMC trajectory plot: $mcmc_filename")

#%%
# Plot energy histogram
energies = ustrip.(trajectory.energy)
p_hist = histogram(energies,
    xlabel="Energy (eV)",
    ylabel="Frequency",
    title="RWMC Energy Distribution (T = $T_rwmc K)",
    label="Sampled Energies",
    bins=50,
    legend=:topright)
hist_filename = joinpath(mcmc_plots_dir, "rwmc_energy_histogram.png")
savefig(p_hist, hist_filename)
println("Saved energy histogram: $hist_filename")

# Print energy statistics
@printf("\nEnergy statistics from RWMC sampling:\n")
@printf("Mean energy: %.4f eV\n", mean(energies))
@printf("Std energy: %.4f eV\n", std(energies))
@printf("Min energy: %.4f eV\n", minimum(energies))
@printf("Max energy: %.4f eV\n", maximum(energies))

#%%
# Plot autocorrelation of energies
using StatsBase: autocor

max_lag = min(500, length(energies) ÷ 2)
acf = autocor(energies, 1:max_lag)

p_acf = plot(1:max_lag, acf,
    xlabel="Lag",
    ylabel="Autocorrelation",
    title="Energy Autocorrelation Function",
    label="ACF",
    linewidth=2,
    legend=:topright)
hline!(p_acf, [0.0], linestyle=:dash, color=:black, label="")
acf_filename = joinpath(mcmc_plots_dir, "rwmc_energy_autocorrelation.png")
savefig(p_acf, acf_filename)
println("Saved autocorrelation plot: $acf_filename")

#%%
# Save sampled structures to XYZ file
samples_with_energy = [add_energy([s], model)[1] for s in samples]
samples_filename = joinpath(experiment_dir, "rwmc_samples.xyz")
ExtXYZ.save(samples_filename, samples_with_energy)
println("Saved RWMC samples: $samples_filename")

println("\nRWMC sampling experiment completed successfully!")