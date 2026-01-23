#%%
using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel
using ACEpotentials: _make_prior

using ACEpotentials: make_atoms_data, assess_dataset, 
                     _rep_dimer_data_atomsbase, default_weights, AtomsData
using LinearAlgebra: Diagonal, I
using PythonCall
using ACESIDopt: comp_potE_error
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
for s in ["train", "test", "val"]
    data[s] = make_atoms_data(raw_data[s], model; 
                            energy_key = "energy", 
                            force_key = nothing, 
                            virial_key = nothing, 
                            weights = default_weights())
end

A = Dict{String, Matrix{Float64}}()
Awp = Dict{String, Matrix{Float64}}()
Y = Dict{String, Vector{Float64}}()
Yw = Dict{String, Vector{Float64}}()
W = Dict{String, Vector{Float64}}()
# actual assembly of the least square system 
for s in ["train", "test", "val"]
    A[s], Y[s], W[s] = ACEfit.assemble(data[s], model)
    Awp[s] = Diagonal(W[s]) * (A[s] / Psqrt) 
    Yw[s] = W[s] .* Y[s]
end



#%%
# # Import Bayesian Ridge regression from scikit-learn
# sklearn_linear = pyimport("sklearn.linear_model")
# BayesianRidge = sklearn_linear.BayesianRidge
# # Create a Bayesian Ridge regression model
# br_model = BayesianRidge(fit_intercept=false, 
#                          alpha_1=1e-6, 
#                          alpha_2=1e-6, 
#                          lambda_1=1e-6, 
#                          lambda_2=1e-6, 
#                          tol=1e-6, 
#                          max_iter=300)


# # Fit the model
# br_model.fit(Awp["train"], Yw["train"])





# Convert sigma_ to Julia matrix
# Σ = pyconvert(Matrix, br_model.sigma_) #this is not correct yet. Need to account for prior precision P^2


# I_active = Vector(1:5)

# br_model = BayesianRidge(fit_intercept=false, 
#                          alpha_1=1e-6, 
#                          alpha_2=1e-6, 
#                          lambda_1=1e-6, 
#                          lambda_2=1e-6, 
#                          tol=1e-6, 
#                          max_iter=300)
# br_model.fit(Awp["train"][I_active,:], Yw["train"][I_active])


# Σ = pyconvert(Matrix, br_model.sigma_)
# coef_tilde = pyconvert(Array, br_model.coef_)
# intercept = pyconvert(Float64, br_model.intercept_)
# alpha = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
# lambda = pyconvert(Float64, br_model.lambda_) # estimate of prior weight precision

# coef = Psqrt \ coef_tilde
# ACEpotentials.Models.set_linear_parameters!(model, coef)

# using ACESIDopt: comp_potE_error
# @info("Training Errors")
# comp_potE_error(raw_data["train"], model)

# @info("Test Error")
# comp_potE_error(raw_data["test"], model)

# size(Σ)
# size(Awp["train"])
# size(Awp["train"][37:38,:])
# Awp["val"] *(Σ*Awp["train"][37:38,:])

#%%
# Awp_val = Awp["val"] # xtilde
# Awp_can = Awp["train"][37:38,:] # xstar
# s = Σ * transpose(Awp_val)
# g = sum((Awp_can*s).^2, dims=1) 
# size(s)
# size(Awp_can)
# size(Awp_val)
# Awp_val s[1,:]

# g / (1/alpha + transpose(Ap_can) * s)  
# #%%
# xstar = Awp["val"][1,:]
# xtilde = Awp["train"][37,:]
# s = Σ * xtilde
# (transpose(xstar) * s)^2 / (1/alpha + transpose(xtilde) * s)
#%%
# xstar = Awp["val"]
# xtilde = Awp["train"][37,:]
# s = Σ * xtilde
# sum((xstar * s).^2)

#/ (1/alpha + transpose(xtilde) * s)
# size(Awp["val"])
# for i in 1:size(Awp["val"],1)
#     a = Awp["val"][i,:]
#     @show size(a)
# end
#%%
function expected_red_variance(Σ, xstar::Vector{T}, xtilde::Vector{T}, alpha) where {T}
    s = Σ * xtilde
    return (transpose(xstar) * s)^2 / (1/alpha + transpose(xtilde) * s)
end

function expected_red_variance(Σ, Xstar::Matrix{T}, xtilde::Vector{T}, alpha) where {T}
    return sum(expected_red_variance(Σ, Xstar[i,:], xtilde, alpha) for i in 1:size(Xstar,1))
end

function expected_red_variance_fast(Σ, Xstar::Matrix{T}, xtilde::Vector{T}, alpha) where {T}
    s = Σ * xtilde
    return sum((Xstar * s).^2) / (1/alpha + transpose(xtilde) * s)
end

function pred_variance(Σ, xstar::Vector{T}, alpha) where {T}
    return transpose(xstar) * Σ * xstar + 1/alpha
end

# expected_red_variance(Σ, Awp["val"][1,:], Awp["train"][37,:], alpha)
# expected_red_variance_fast(Σ, Awp["val"], Awp["train"][37,:], alpha)
# expected_red_variance(Σ, Awp["val"], Awp["train"][37,:], alpha)
# exp_red = [expected_red_variance_fast(Σ, Awp["val"], Awp["train"][i,:], alpha) for i in 1:size(Awp["train"],1)]
# exp_red[argmax(exp_red)]
#%%
test_error = []
train_error = []
val_error = []
I_active = Vector(1:2)
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
    br_model.fit(Awp["train"][I_active,:], Yw["train"][I_active])
    Σ = pyconvert(Matrix, br_model.sigma_)
    coef_tilde = pyconvert(Array, br_model.coef_)
    alpha = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
    coef = Psqrt \ coef_tilde
    ACEpotentials.Models.set_linear_parameters!(model, coef)
    push!(train_error, comp_potE_error(raw_data["train"], model))
    push!(test_error, comp_potE_error(raw_data["test"], model))
    push!(val_error, comp_potE_error(raw_data["val"], model))

    p_var = [pred_variance(Σ, Awp["train"][i,:], alpha) for i in 1:size(Awp["train"],1)] 
    p_var_mean = mean(p_var[i] for i in 1:length(p_var) if i ∉ I_active)
    # @show i p_var_mean
    # select next active point
    exp_red = [(j ∉ I_active && p_var[j] >= p_var_mean) ? expected_red_variance_fast(Σ, Awp["val"], Awp["train"][j,:], alpha) : -1 for j in 1:size(Awp["train"],1)  ]
    idx = argmax(exp_red)
    push!(I_active, idx) 
end


# Plot the evolution of errors as a function of active learning iterations
iterations = 1:n_active

plot(iterations, train_error, 
     label="Train Error", 
     marker=:circle, 
     linewidth=2,
     xlabel="Active Learning Iteration",
     ylabel="RMSE (eV)",
     title="Error Evolution During Active Learning",
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

#%%