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
#%%

#=
Test - Silicon Active Learning
=#
#%%
Random.seed!(2234)
experiment_name = "active_silicon_test_10"

# Create experiment directory structure under experiments/results/
experiment_dir = joinpath(dirname(@__FILE__), "results", experiment_name)
plots_dir = joinpath(experiment_dir, "plots")
mkpath(plots_dir)  # Creates both experiment_dir and plots_dir if they don't exist
models_dir = joinpath(experiment_dir, "models")
mkpath(models_dir)
tsmodels_dir = joinpath(experiment_dir, "models-ts")
mkpath(tsmodels_dir)
mcmc_plots_dir = joinpath(plots_dir, "mcmc_plots")
mkpath(mcmc_plots_dir)

raw_data = Dict{String, Any}()
raw_data["candidates"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_0_train_100frames.xyz")
raw_data["test"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_0_test_100frames.xyz")
raw_data["sur"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_0_val_100frames.xyz")

using ACESIDopt: convert_forces_to_svector

#%%
# ACE model for Silicon (single element)
model = ace1_model(elements = [:Si],
                   rcut = 5.5,
                   order = 3,        # body-order - 1
                   totaldegree = 6);

Psqrt = _make_prior(model, 4, nothing) # square root of prior precision matrix

my_weights() = Dict("default"=>Dict("E"=>100.0, "F"=>1.0, "V"=>1.0))
data = Dict{String, Any}()
row_mappings = Dict{String, Any}()
n_data = Dict{String, Int}()
for s in ["test", "candidates"]
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

A = Dict{String, Matrix{Float64}}()
Awp = Dict{String, Matrix{Float64}}()
Y = Dict{String, Vector{Float64}}()
Yw = Dict{String, Vector{Float64}}()
W = Dict{String, Vector{Float64}}()
# actual assembly of the least square system 
for s in ["test", "candidates"]
    A[s], Y[s], W[s] = ACEfit.assemble(data[s], model)
    Awp[s] = Diagonal(W[s]) * (A[s] / Psqrt) 
    Yw[s] = W[s] .* Y[s]
end

I_start = 1:10:100
n_active = 10
raw_data_train = convert_forces_to_svector.(raw_data["candidates"][I_start])
test_error = []
cand_error = []
sur_error = []

for t in 1:n_active
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
                            alpha_1=1e-6, 
                            alpha_2=1e-6, 
                            lambda_1=1e-6, 
                            lambda_2=1e-6, 
                            tol=1e-6, 
                            max_iter=300)
    # fit model and export relevant parameters
    br_model.fit(Awp_train, Yw_train)
    Σ = pyconvert(Matrix, br_model.sigma_)
    coef_tilde = pyconvert(Array, br_model.coef_)
    α = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
    coef = Psqrt \ coef_tilde

    Σt = inv(Psqrt) * Σ * inv(Psqrt)
    Σt_chol, jitter = cholesky_with_jitter((Σt + transpose(Σt))/2; max_jitter_fraction=1e-3) 
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

    push!(test_error, comp_potE_error(raw_data["test"], model))

    p_energy = plot_energy_comparison(raw_data["test"], model,
                               joinpath(plots_dir, "energy_scatter_iter_$(lpad(t, 3, '0')).png"))
    
    # Scatter plot: true vs predicted forces for test set
    p_forces = plot_forces_comparison(raw_data["test"], model, 
                                   joinpath(plots_dir, "forces_scatter_iter_$(lpad(t, 3, '0')).png"))

    p_energy_train = plot_energy_comparison(raw_data_train, model,
                               joinpath(plots_dir, "train_energy_scatter_iter_$(lpad(t, 3, '0')).png"))
    
    # Scatter plot: true vs predicted forces for test set
    p_forces_train = plot_forces_comparison(raw_data_train, model, 
                                   joinpath(plots_dir, "train_forces_scatter_iter_$(lpad(t, 3, '0')).png"))

    #%%

    rwmc_initial = deepcopy(rand(raw_data_train))
    T_rwmc = 300.0  # K
    n_rwmc_samples = 100
    step_size_rwmc = 0.02  # Å
    burnin_rwmc = 2000
    thin_rwmc = 1000

    sur_samples, sur_acceptance, sur_traj = run_rwmc_sampling(
        rwmc_initial, ts_model, n_rwmc_samples, T_rwmc;
        step_size=step_size_rwmc, burnin=burnin_rwmc, thin=thin_rwmc
    )
    rwmc_initial = deepcopy(rand(raw_data_train))
    cand_samples, cand_acceptance, cand_traj = run_rwmc_sampling(
        rwmc_initial, ts_model, n_rwmc_samples, T_rwmc;
        step_size=step_size_rwmc, burnin=burnin_rwmc, thin=thin_rwmc
    )
    # @show ustrip.(sur_traj.energy)
    # Plot surrogate MCMC trajectory
    p_sur_mcmc = plot(1:length(sur_traj.energy), ustrip.(sur_traj.energy),
        xlabel="MCMC Iteration",
        ylabel="Energy (eV)",
        title="Surrogate MCMC Energy Trajectory (Iter $t)",
        label="Energy",
        # linewidth=1.5,
        legend=:topright)
    sur_mcmc_filename = joinpath(mcmc_plots_dir, "sur_mcmc_iter_$(lpad(t, 3, '0')).png")
    savefig(p_sur_mcmc, sur_mcmc_filename)
    println("Saved surrogate MCMC plot: $sur_mcmc_filename")
    
    # Plot candidate MCMC trajectory
    p_cand_mcmc = plot(1:length(cand_traj.energy), ustrip.(cand_traj.energy),
        xlabel="MCMC Iteration",
        ylabel="Energy (eV)",
        title="Candidate MCMC Energy Trajectory (Iter $t)",
        label="Energy",
        # linewidth=1.5,
        legend=:topright)
    cand_mcmc_filename = joinpath(mcmc_plots_dir, "cand_mcmc_iter_$(lpad(t, 3, '0')).png")
    savefig(p_cand_mcmc, cand_mcmc_filename)
    println("Saved candidate MCMC plot: $cand_mcmc_filename")



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

    system = mflexiblesystem(raw_data["tcand"][idx])
    EF=queryASEModel(system; calculator=:MACE)

    add_energy_forces(raw_data["tcand"][idx], EF.energy, EF.forces)
    push!(raw_data_train, deepcopy(raw_data["tcand"][idx]))


    p_energy_train = plot_energy_comparison(raw_data_train, ts_model,
                               joinpath(plots_dir, "train_energy_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
    
    # Scatter plot: true vs predicted forces for test set
    p_forces_train = plot_forces_comparison(raw_data_train, ts_model, 
                                   joinpath(plots_dir, "train_forces_scatter_before_fit_iter_$(lpad(t, 3, '0')).png");marked=[length(raw_data_train)])
end

#%%
# Plot and save test error evolution
iterations = 1:length(test_error)
p_error = plot(iterations, test_error, 
    xlabel="Active Learning Iteration",
    ylabel="Test RMSE (eV)",
    title="Test Error Evolution During Active Learning (Silicon)",
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
# Load large test sets from all replicas
raw_data["test-large-0"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_0_val_1000frames.xyz")
raw_data["test-large-1"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_1_val_1000frames.xyz")
raw_data["test-large-2"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_2_val_1000frames.xyz")
raw_data["test-large-3"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/silicon_remd_parallel1/replica_3_val_1000frames.xyz")

d_ind = [0, 1, 2, 3]
large_test_error = Dict(d => Float64[] for d in d_ind)
for i in 1:length(test_error)
    model_filename = joinpath(models_dir, "model_iter_$(lpad(i, 3, '0')).ace")
    model = ACEpotentials.load_model(model_filename)[1]
    for j in d_ind
        push!(large_test_error[j], comp_potE_error(raw_data["test-large-$j"], model))
        @printf("Large test set error (replica %d) for model iter %d: %.4f eV\n", j, i, large_test_error[j][end])
    end
end

# Plot combined error evolution
iterations = 1:length(test_error)
p_error = plot(iterations, test_error, 
    xlabel="Active Learning Iteration",
    ylabel="Test RMSE (eV)",
    title="Test Error Evolution During Active Learning (Silicon)",
    label="Test Error (small)", 
    marker=:circle, 
    linewidth=2,
    markersize=5,
    legend=:topright, yscale=:log10)

# Add large test set errors for each replica
for j in d_ind
    plot!(p_error, iterations, large_test_error[j],
        label="Test Error (large, replica $j)",
        marker=:auto,
        linewidth=2,
        markersize=4)
end

# Save the combined error evolution plot
error_plot_filename = joinpath(experiment_dir, "test_error_evolution_all.png")
savefig(p_error, error_plot_filename)
println("Saved combined error evolution plot: $error_plot_filename")