#%%
using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel
using ACEpotentials: _make_prior

using ACEpotentials: make_atoms_data, assess_dataset, 
                     _rep_dimer_data_atomsbase, default_weights, AtomsData
using LinearAlgebra: Diagonal, I, inv
using PythonCall
using ACESIDopt: comp_potE_error

using ACESIDopt: expected_red_variance, row_mapping, pred_variance
#%%


#%%
raw_data = Dict{String, Any}()
raw_data["train"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_train_100frames.xyz")
raw_data["test"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_test_100frames.xyz")
raw_data["val"] = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_val_100frames.xyz")


#%%
model = ace1_model(elements = [:C, :H, :O, :N],
                   rcut = 5.5,
                   order = 2,        # body-order - 1
                   totaldegree = 5 );

Psqrt = _make_prior(model, 4, nothing) # square root of prior precision matrix



data = Dict{String, Any}()
row_mappings = Dict{String, Any}()
n_data = Dict{String, Int}()
for s in ["train", "test", "val"]
    data[s] = make_atoms_data(raw_data[s], model; 
                            energy_key = "energy", 
                            force_key = "forces", 
                            virial_key = nothing, 
                            weights = default_weights())
    row_mappings[s] = row_mapping(data[s], model)
    n_data[s] = length(raw_data[s])
end

for s in ["train",  "val"]
    data[string(s,"-nf")] = make_atoms_data(raw_data[s], model; 
                            energy_key = "energy", 
                            force_key = nothing, 
                            virial_key = nothing, 
                            weights = default_weights())
    row_mappings[string(s,"-nf")] = row_mapping(data[string(s,"-nf")], model)
    n_data[string(s,"-nf")] = length(raw_data[s])
end

A = Dict{String, Matrix{Float64}}()
Awp = Dict{String, Matrix{Float64}}()
Y = Dict{String, Vector{Float64}}()
Yw = Dict{String, Vector{Float64}}()
W = Dict{String, Vector{Float64}}()
# actual assembly of the least square system 
for s in ["train", "test", "val","train-nf","val-nf"]
    A[s], Y[s], W[s] = ACEfit.assemble(data[s], model)
    Awp[s] = Diagonal(W[s]) * (A[s] / Psqrt) 
    Yw[s] = W[s] .* Y[s]
end



#%%
using StatsPlots


specs = [(A_train=Awp["train-nf"],Y_train=Yw["train-nf"], A_sel=Awp["train-nf"], row_mapping=row_mappings["train-nf"], A_val=Awp["val-nf"], text="Train and select based on energy only"),
         (A_train=Awp["train"],Y_train=Yw["train"], A_sel=Awp["train-nf"], row_mapping=row_mappings["train-nf"], A_val=Awp["val-nf"], text="Train on energy and forces, select based on energy only"),
         (A_train=Awp["train"],Y_train=Yw["train"], A_sel=Awp["train"], row_mapping=row_mappings["train"], A_val=Awp["val-nf"], text="Train and select based on energy and forces")]


for s in specs
    test_error = []
    train_error = []
    val_error = []
    I_active = Vector(1:5)
    n_active = 50
    sklearn_linear = pyimport("sklearn.linear_model")
    BayesianRidge = sklearn_linear.BayesianRidge
    br_model = BayesianRidge(fit_intercept=false, 
                            alpha_1=1e-6, 
                            alpha_2=1e-6, 
                            lambda_1=1e-6, 
                            lambda_2=1e-6, 
                            tol=1e-6, 
                            max_iter=300)
    for i in 1:n_active
        # fit model and export relevant parameters
        br_model.fit(s.A_train[I_active,:], s.Y_train[I_active])
        Σ = pyconvert(Matrix, br_model.sigma_)
        coef_tilde = pyconvert(Array, br_model.coef_)
        alpha = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
        coef = Psqrt \ coef_tilde
        ACEpotentials.Models.set_linear_parameters!(model, coef)
        push!(train_error, comp_potE_error(raw_data["train"], model))
        push!(test_error, comp_potE_error(raw_data["test"], model))
        push!(val_error, comp_potE_error(raw_data["val"], model))

        p_var = [pred_variance(Σ, s.A_train[i,:], alpha) for i in 1:n_data["train"]] 
        p_var_mean = mean(p_var[i] for i in 1:length(p_var) if i ∉ I_active)
        # @show i p_var_mean
        # select next active point
        exp_red = [(j ∉ I_active && p_var[j] >= p_var_mean) ? expected_red_variance(Σ, s.A_val,  s.A_sel[s.row_mapping[j],:], alpha) : -1 for j in 1:n_data["train"] ] 
        #exp_red = [(j ∉ I_active && p_var[j] >= p_var_mean) ? expected_red_variance(Σ, Awp["val-nf"],  Awp["train-nf"][j,:], alpha) : -1 for j in 1:n_data["train-nf"] ] 

        idx = argmax(exp_red)
        push!(I_active, idx) 
    end





    p = histogram([d.system_data.energy for d in raw_data["train"]], 
            bins=100, 
            xlabel="Energy (eV)", 
            ylabel="Frequency", 
            title=s.text)
    histogram!(p,[d.system_data.energy for d in raw_data["train"][I_active]], 
            bins=100, 
            xlabel="Energy (eV)", 
            ylabel="Frequency")
    histogram!(p,[d.system_data.energy for d in raw_data["val"]], 
            bins=100, 
            xlabel="Energy (eV)", 
            ylabel="Frequency")
    display(p)
    iterations = 1:n_active

    p = plot(iterations, train_error, 
        label="Train Error", 
        marker=:circle, 
        linewidth=2,
        xlabel="Active Learning Iteration",
        ylabel="RMSE (eV)",
        title=s.text,
        legend=:topright,
        yscale=:log10)
    plot!(iterations, test_error, 
        label="Test Error", 
        marker=:square, 
        linewidth=2)
    plot!(iterations, val_error, 
        label="Validation Error", 
        marker=:diamond, 
        linewidth=2)
    display(p)
end
