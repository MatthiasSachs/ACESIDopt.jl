#!/usr/bin/env julia

"""
Script to fit an ACE model with train/test split.

Training data:
- /experiments/results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz

Test data:
- /experiments/results/ptd_ACE_silicon_dia-primitive-2-large-b/data/replica_1_samples.xyz
"""

using Pkg
Pkg.activate(".")

using ACEpotentials, AtomsBase
using ExtXYZ
using ACEfit
using Printf
using Statistics
using ACEpotentials: make_atoms_data, ace1_model, set_committee!
using ACEpotentials
using LinearAlgebra: I, Diagonal
using LinearAlgebra
using Random
Random.seed!(1234)
# Path to training and test data
mfile = Dict{String,String}()
mfile["train"] = joinpath(@__DIR__, "results/ptd_ACE_silicon_dia-primitive-2-large/data/replica_1_samples.xyz")
mfile["test"] = joinpath(@__DIR__, "results/ptd_ACE_silicon_dia-primitive-2-large-c/data/replica_1_samples.xyz")

println("="^70)
println("Loading data...")
println("="^70)

raw_data = Dict{String, Vector{AbstractSystem}}()
# Load training data
for s in ["train", "test"]  
    raw_data[s] = ExtXYZ.load(mfile[s])
    @info "  Loaded $(length(raw_data[s])) $s configurations"
end


#%%
thin_train = 100
thin_test = 1
model = ace1_model(elements = [:Si,],
                   Eref = [:Si => -158.54496821],
                   order = 3,
                   totaldegree = 8);
# solver = ACEfit.BLR(committee_size = 1000, factorization = :svd)
# g = acefit!(raw_data_train,  model;
#         solver = solver,
#         energy_key = "energy", force_key = "forces",
#         verbose = false);
using ACEpotentials: _make_prior
mm_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
data_train = make_atoms_data(raw_data["train"][1:thin_train:end], model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = mm_weights())
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Psqrt =  _make_prior(model, 4, nothing)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 
Yw_train = W_train .* Y_train



using LinearAlgebra
using ACEfit: bayesian_linear_regression
using Random: randperm
using AtomsCalculators: potential_energy, forces
n_samples_max = 100
random_indices = randperm(size(Awp_train,1))[1:min(n_samples_max, size(Awp_train,1))]

# Initialize hierarchical dictionary properly
terrors = Dict{String,Dict{String,Dict{String,Vector{Float64}}}}()
for dataset in ["train", "test"]
    terrors[dataset] = Dict{String,Dict{String,Vector{Float64}}}()
    for metric in ["rmse", "mae"]
        terrors[dataset][metric] = Dict{String,Vector{Float64}}()
        for quantity in ["energy", "forces"]
            terrors[dataset][metric][quantity] = Float64[]
        end
    end
end

for i in 1:length(random_indices)
    @printf("Using training sample %d (index %d)\n", i, random_indices[i])

    train_inds = random_indices[1:i]
    solver = ACEfit.BLR(committee_size = 10000, factorization = :svd)
    result1 = bayesian_linear_regression(Awp_train[train_inds, :], Yw_train[train_inds]; solver.kwargs..., ret_covar = true)
    coeffs = Psqrt \ result1["C"]   
    # dispatch setting of parameters 
    ACEpotentials.Models.set_linear_parameters!(model, coeffs)
    if haskey(result1, "committee")
        co_coeffs = result1["committee"]
        co_ps_vec = [ Psqrt \ co_coeffs[:,i] for i in 1:size(co_coeffs,2) ]
        set_committee!(model, co_ps_vec)
    end

    #%% Compute test errors
    println("\n" * "="^70)
    println("Computing test errors...")
    println("="^70)


    # Compute test error on energy
    test_energy_errors = Float64[]
    for config in raw_data["test"][1:thin_test:end]
        pred_energy = potential_energy(config, model).val
        true_energy = config.system_data.energy
        error = (pred_energy - true_energy)
        push!(test_energy_errors, error^2)
    end

    push!(terrors["test"]["rmse"]["energy"], sqrt(mean(test_energy_errors)))
    push!(terrors["test"]["mae"]["energy"], mean(abs.(sqrt.(test_energy_errors))))

    @info "Energy errors (test):"
    @info "  RMSE: $(terrors["test"]["rmse"]["energy"][end]) eV"
    @info "  MAE: $(terrors["test"]["mae"]["energy"][end]) eV"
    using Unitful: ustrip
    # Compute test error on forces
    test_force_errors = Float64[]
    for config in raw_data["test"][1:thin_test:end]
        pred_forces = ustrip.(forces(config, model))
        true_forces = config.atom_data.forces
        for (pf, tf) in zip(pred_forces, true_forces)
            error_sq = sum((pf .- tf).^2)
            push!(test_force_errors, error_sq)
        end
    end

    push!(terrors["test"]["rmse"]["forces"], sqrt(mean(test_force_errors)))
    push!(terrors["test"]["mae"]["forces"], mean(abs.(sqrt.(test_force_errors))))

    @info "Force errors (test):"
    @info "  RMSE: $(terrors["test"]["rmse"]["forces"][end]) eV/Å"
    @info "  MAE: $(terrors["test"]["mae"]["forces"][end]) eV/Å"
    # Compute training errors for comparison
    println("\n" * "="^70)
    println("Computing training errors...")
    println("="^70)

    train_energy_errors = Float64[]
    for config in raw_data["train"][1:thin_train:end]
        pred_energy = potential_energy(config, model).val
        true_energy = config.system_data.energy
        error = (pred_energy - true_energy)
        push!(train_energy_errors, error^2)
    end

    push!(terrors["train"]["rmse"]["energy"], sqrt(mean(train_energy_errors)))
    push!(terrors["train"]["mae"]["energy"], mean(abs.(sqrt.(train_energy_errors))))

    @info "Energy errors (training):"
    @info "  RMSE: $(terrors["train"]["rmse"]["energy"][end]) eV"
    @info "  MAE: $(terrors["train"]["mae"]["energy"][end]) eV"

    train_force_errors = Float64[]
    for config in raw_data["train"][1:thin_train:end]
        pred_forces = ustrip.(forces(config, model))
        true_forces = config.atom_data.forces
        for (pf, tf) in zip(pred_forces, true_forces)
            error_sq = sum((pf .- tf).^2)
            push!(train_force_errors, error_sq)
        end
    end

    push!(terrors["train"]["rmse"]["forces"], sqrt(mean(train_force_errors)))
    push!(terrors["train"]["mae"]["forces"], mean(abs.(sqrt.(train_force_errors))))

    @info "Force errors (training):"
    @info "  RMSE: $(terrors["train"]["rmse"]["forces"][end]) eV/Å"
    @info "  MAE: $(terrors["train"]["mae"]["forces"][end]) eV/Å"

    println("\n" * "="^70)
    println("Model evaluation completed!")
    println("="^70)
end

#%% Generate plots
println("\n" * "="^70)
println("Generating plots...")
println("="^70)

using Plots

# Training set sizes (number of samples used)
training_sizes = 1:length(random_indices)

# Create figure with subplots
p1 = plot(training_sizes, terrors["test"]["rmse"]["energy"],
          label="Test",
          xlabel="Training Set Size",
          ylabel="Energy RMSE (eV)",
          title="Energy RMSE vs Training Size",
          marker=:circle,
          linewidth=2,
          legend=:topright,
          grid=true,
          yscale=:log10)
plot!(p1, training_sizes, terrors["train"]["rmse"]["energy"],
      label="Train",
      marker=:square,
      linewidth=2)

p2 = plot(training_sizes, terrors["test"]["rmse"]["forces"],
          label="Test",
          xlabel="Training Set Size",
          ylabel="Forces RMSE (eV/Å)",
          title="Forces RMSE vs Training Size",
          marker=:circle,
          linewidth=2,
          legend=:topright,
          grid=true,
          yscale=:log10)
plot!(p2, training_sizes, terrors["train"]["rmse"]["forces"],
      label="Train",
      marker=:square,
      linewidth=2)

p3 = plot(training_sizes, terrors["test"]["mae"]["energy"],
          label="Test",
          xlabel="Training Set Size",
          ylabel="Energy MAE (eV)",
          title="Energy MAE vs Training Size",
          marker=:circle,
          linewidth=2,
          legend=:topright,
          grid=true,
          yscale=:log10)
plot!(p3, training_sizes, terrors["train"]["mae"]["energy"],
      label="Train",
      marker=:square,
      linewidth=2)

p4 = plot(training_sizes, terrors["test"]["mae"]["forces"],
          label="Test",
          xlabel="Training Set Size",
          ylabel="Forces MAE (eV/Å)",
          title="Forces MAE vs Training Size",
          marker=:circle,
          linewidth=2,
          legend=:topright,
          grid=true,
          yscale=:log10)
plot!(p4, training_sizes, terrors["train"]["mae"]["forces"],
      label="Train",
      marker=:square,
      linewidth=2)

# Combine all plots
p_combined = plot(p1, p2, p3, p4, 
                  layout=(2,2), 
                  size=(1200, 900),
                  plot_title="Model Performance vs Training Set Size")

# Save plot
output_plot = joinpath(@__DIR__, "results/training_size_vs_error.png")
savefig(p_combined, output_plot)
@info "Plot saved to: $output_plot"

# Display plot
display(p_combined)

# Also create a focused plot on test RMSE only
p_test_only = plot(training_sizes, terrors["test"]["rmse"]["energy"],
                   label="Energy",
                   xlabel="Training Set Size",
                   ylabel="Test RMSE",
                   title="Test RMSE vs Training Set Size",
                   marker=:circle,
                   linewidth=2,
                   legend=:topright,
                   grid=true,
                   size=(800, 600),
                   yscale=:log10)
plot!(p_test_only, training_sizes, terrors["test"]["rmse"]["forces"],
      label="Forces",
      marker=:square,
      linewidth=2)

output_plot_test = joinpath(@__DIR__, "results/training_size_vs_test_rmse.png")
savefig(p_test_only, output_plot_test)
@info "Test RMSE plot saved to: $output_plot_test"

display(p_test_only)

println("\n" * "="^70)
println("All plots generated successfully!")
println("="^70)


