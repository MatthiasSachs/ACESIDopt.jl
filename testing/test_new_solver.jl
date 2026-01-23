#%%
using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel
#%%
mace_dataset_train = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_train_100frames.xyz")
mace_dataset_test = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_test_100frames.xyz")
mace_dataset_val = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_val_50frames.xyz")


#%%
model = ace1_model(elements = [:C, :H, :O, :N],
                   rcut = 5.5,
                   order = 2,        # body-order - 1
                   totaldegree = 4 );

# res = acefit!(mace_dataset_train, model;
#         solver=solver, data_keys...);
# solver = ACEfit.BLR(tol=0.0, factorization = :svd)
# model2, res2 = MFitmodel.acefit(mace_dataset_train, model; data_keys...);       
# res2

using ACEpotentials: _make_prior

using ACEpotentials: make_atoms_data, assess_dataset, 
                     _rep_dimer_data_atomsbase, default_weights, AtomsData
using LinearAlgebra: Diagonal, I

Psqrt = _make_prior(model, 4, nothing) # square root of prior precision matrix


data_train = make_atoms_data(mace_dataset_train, model; 
                        energy_key = "energy", 
                        force_key = nothing, 
                        virial_key = nothing, 
                        weights = default_weights())

# actual assembly of the least square system 
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
At_train = Diagonal(W_train) * (A_train / Psqrt) 
Yt_train = W_train .* Y_train

using PythonCall

# Import Bayesian Ridge regression from scikit-learn
sklearn_linear = pyimport("sklearn.linear_model")
BayesianRidge = sklearn_linear.BayesianRidge

# Create a Bayesian Ridge regression model
br_model = BayesianRidge(fit_intercept=false, 
                         alpha_1=1e-6, 
                         alpha_2=1e-6, 
                         lambda_1=1e-6, 
                         lambda_2=1e-6, 
                         tol=1e-6, 
                         max_iter=300)

# Fit the model
br_model.fit(At_train, Yt_train)



# Convert ctilde from Python array to Julia array
coef_tilde = pyconvert(Array, br_model.coef_)
coef = Psqrt \ coef_tilde

# = copy(c)
# Convert other br_model attributes to Julia
intercept = pyconvert(Float64, br_model.intercept_)
alpha = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
lambda = pyconvert(Float64, br_model.lambda_) # estimate of prior weight precision

# Convert sigma_ to Julia matrix
Σ = pyconvert(Matrix, br_model.sigma_) #this is not correct yet. Need to account for prior precision P^2

data_val = make_atoms_data(mace_dataset_val, model; 
                        energy_key = "energy", 
                        force_key = nothing, 
                        virial_key = nothing, 
                        weights = default_weights())

# actual assembly of the least square system 
A_val, Y_val, W_val = ACEfit.assemble(data_val, model)
At_val = Diagonal(W_val) * (A_val / Psqrt) 
Yt_val = W_val .* Y_val


function expected_red_variance(Σ, Awp_val, ap, alpha)
    s = Σ*ap
    g = sum((transpose(Awp_val)*s).^2)  
    return g / (1/alpha + ap * s) 
end

for ap in 

sum((Σ - inv(lambda * I + alpha * transpose(Awp) * Awp)).^2)

inv(Psqrt) * Σ * inv(Psqrt) - inv(Psqrt *(lambda * I + alpha * transpose(Awp) * Awp) * Psqrt)



P = Psqrt*Psqrt  # prior precision matrix
Lambda = lambda * Psqrt*Psqrt + alpha * transpose(A) * A # posterior precision matrix


sigma_corrected = inv(Lambda)

sigma_corrected = inv( lambda * P + alpha * transpose(A) * A )
sigma_corrected - inv(P) * sigma


ACEpotentials.Models.set_linear_parameters!(model, c)

using ACESIDopt: comp_potE_error
@info("Training Errors")
comp_potE_error(mace_dataset_train, model)

@info("Test Error")
comp_potE_error(mace_dataset_test, model)


#%%


