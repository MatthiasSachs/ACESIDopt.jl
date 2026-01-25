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

using Distributed
using ACESIDopt.MSamplers: RWMCSampler, run_parallel_tempering_distributed
using Random
using LinearAlgebra: cholesky
using ACESIDopt: cholesky_with_jitter, add_energy_forces, add_forces, add_energy, convert_forces_to_svector
using ACESIDopt: mflexiblesystem, queryASEModel
using AtomsCalculators: potential_energy, forces
using ACESIDopt: plot_forces_comparison, plot_energy_comparison, generate_ptd_diagnostics_and_log
using ProgressMeter

#%%

#=
Active Learning - Silicon with Distributed Parallel Tempering
Training from generated samples with PT sampling
=#
#%%
Random.seed!(2234)
experiment_name = "active_silicon_ptd_run1"

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

# Create experiment directory structure under experiments/results/
experiment_dir = joinpath(dirname(@__FILE__), "results", experiment_name)
plots_dir = joinpath(experiment_dir, "plots")
mkpath(plots_dir)  # Creates both experiment_dir and plots_dir if they don't exist
models_dir = joinpath(experiment_dir, "models")
mkpath(models_dir)
tsmodels_dir = joinpath(experiment_dir, "models-ts")
mkpath(tsmodels_dir)
pt_diagnostics_dir = joinpath(plots_dir, "pt_diagnostics")
mkpath(pt_diagnostics_dir)

# Load data - using generated samples as training/candidates
raw_data = Dict{String, Any}()
raw_data["test-all"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/Si-armorph-1/replica_1_T300K_samples_run3-high-exchanges.xyz")

using ACESIDopt: convert_forces_to_svector

# Subsample 5 data points for initial training set
n_total = length(raw_data["test-all"])
println("Total available samples: $n_total")
train_indices = sort(Random.randperm(n_total)[1:5])
test_indices = setdiff(1:n_total, train_indices)

println("Selected training indices: $train_indices")
println("Remaining test samples: $(length(test_indices))")

# Create training set from subsampled data
raw_data["train"]= convert_forces_to_svector.(raw_data["test-all"][train_indices])

# Create test set from remaining data
raw_data["test"] = convert_forces_to_svector.(raw_data["test-all"][test_indices[1:10:end]])

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
n_active = 10
test_error = []

# Distributed parallel tempering parameters
n_replicas = 4
T_min = 300.0
T_max = 900.0
n_samples_pt = 500
burnin_pt = 10000
thin_pt = 100
exchange_interval = 50
step_size_pt = 0.01
r_cut = 5.5u"Å"

println("\nDistributed Parallel Tempering Parameters:")
println("  Number of replicas: $n_replicas")
println("  Temperature range: $T_min K to $T_max K")
println("  Samples per replica: $n_samples_pt")
println("  Burn-in steps: $burnin_pt")
println("  Thinning: $thin_pt")
println("  Exchange interval: $exchange_interval")
println("  Step size: $step_size_pt Å")

# Load model on all workers
@everywhere model_path = nothing  # Will be updated in the loop

raw_data_train = deepcopy(raw_data["train"])
for t in 1:n_active
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
    sampled_indices_sur = Random.randperm(n_available)[1:min(n_replicas, n_available)]
    if length(sampled_indices_sur) < n_replicas
        # If not enough unique samples, repeat some
        while length(sampled_indices_sur) < n_replicas
            push!(sampled_indices_sur, rand(1:n_available))
        end
    end
    initial_systems_sur = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_sur]
    
    # Sample initial systems for candidate PT
    sampled_indices_cand = Random.randperm(n_available)[1:min(n_replicas, n_available)]
    if length(sampled_indices_cand) < n_replicas
        while length(sampled_indices_cand) < n_replicas
            push!(sampled_indices_cand, rand(1:n_available))
        end
    end
    initial_systems_cand = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_cand]
    
    # Create samplers
    sampler_sur = RWMCSampler(step_size=step_size_pt)
    sampler_cand = RWMCSampler(step_size=step_size_pt)
    
    # Run surrogate and candidate PT in parallel using @sync
    sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = nothing, nothing, nothing, nothing, nothing
    cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = nothing, nothing, nothing, nothing, nothing
    
    @sync begin
        @async begin
            println("  Starting surrogate PT...")
            sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = 
                run_parallel_tempering_distributed(
                    sampler_sur, initial_systems_sur, ts_model, n_replicas, T_min, T_max;
                    n_samples=n_samples_pt, burnin=burnin_pt, thin=thin_pt,
                    exchange_interval=exchange_interval, collect_forces=false
                )
            println("  Surrogate PT complete!")
        end
        
        @async begin
            println("  Starting candidate PT...")
            cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = 
                run_parallel_tempering_distributed(
                    sampler_cand, initial_systems_cand, ts_model, n_replicas, T_min, T_max;
                    n_samples=n_samples_pt, burnin=burnin_pt, thin=thin_pt,
                    exchange_interval=exchange_interval, collect_forces=false
                )
            println("  Candidate PT complete!")
        end
    end
    
    # Generate diagnostics for surrogate PT
    println("\nGenerating surrogate PT diagnostics...")
    sur_diag_dir = joinpath(pt_diagnostics_dir, "surrogate")
    mkpath(sur_diag_dir)
    generate_ptd_diagnostics_and_log(sur_replicas, sur_temperatures, sur_mcmc_rates, 
                                     sur_exchange_rates, sur_trajs, sur_diag_dir, t, r_cut)
    
    # Generate diagnostics for candidate PT
    println("\nGenerating candidate PT diagnostics...")
    cand_diag_dir = joinpath(pt_diagnostics_dir, "candidate")
    mkpath(cand_diag_dir)
    generate_ptd_diagnostics_and_log(cand_replicas, cand_temperatures, cand_mcmc_rates, 
                                     cand_exchange_rates, cand_trajs, cand_diag_dir, t, r_cut)
    
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

    # Load reference ACE model for ground truth evaluation
    ref_model_path = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/models/Si_ref_model-small.json"
    ref_model = ACEpotentials.load_model(ref_model_path)[1]
    
    # Compute energy and forces with reference model
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
    title="Test Error Evolution During Active Learning (Silicon PTD)",
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
    title="Test Error Evolution During Active Learning (Silicon PTD)",
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

println("\n" * "="^70)
println("Active Learning with Distributed Parallel Tempering Complete!")
println("="^70)

# Optional: Remove workers if desired
# rmprocs(workers())
