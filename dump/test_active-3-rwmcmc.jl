#%%
using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase
using ACESIDopt
using ACESIDopt: MFitmodel
using ACESIDopt.MSamplers: rwmc_step!, run_rwmc_sampling
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

#%%
using AtomsCalculators: forces, potential_energy
for system in raw_data["train"][I_active]
    @show potential_energy(system, model)
    @show system.system_data.energy
end

using StatsPlots


p = histogram([d.system_data.energy for d in raw_data["train"]], 
          bins=100, 
          xlabel="Energy (eV)", 
          ylabel="Frequency", 
          title="Histogram of Energies in Training Dataset")
histogram!(p,[d.system_data.energy for d in raw_data["train"][I_active]], 
          bins=100, 
          xlabel="Energy (eV)", 
          ylabel="Frequency", 
          title="Histogram of Energies in Training Dataset")
histogram!(p,[d.system_data.energy for d in raw_data["val"]], 
          bins=100, 
          xlabel="Energy (eV)", 
          ylabel="Frequency", 
          title="Histogram of Energies in Training Dataset")

#           br_model.fit(Awp["train"][I_active,:], Yw["train"][I_active])
# coef_tilde = pyconvert(Array, br_model.coef_)
# coef = Psqrt \ coef_tilde
# ACEpotentials.Models.set_linear_parameters!(model, coef)
# I_active
forces(raw_data["train"][I_active[1]], model) 
forces(raw_data["train"][I_active[2]], model)


# position(raw_data["train"][end],:)
# position(raw_data["train"][end-1],:)
# forces(raw_data["train"][end], model) 
# forces(raw_data["train"][end-1], model)

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
# BAOAB Langevin integrator for glycine simulation

using Unitful: ustrip, @u_str

# Run simulation of glycine at 300 K
system = raw_data["train"][1]  # Use first training structure as initial configuration
fsystem  = FlexibleSystem(system)

# getter functions
using AtomsBase: atomic_mass
typeof(system)
atomic_mass( system,1 )
species( system,1 )
system.atom_data.mass
typeof( system) 
mass(system,:)
mass(fsystem,:)
position(system,:)
velocity(system,:)
velocity(fsystem,:)



# Setter functions for AtomsBase.Atoms
import AtomsBase: set_position!, set_velocity!      
function AtomsBase.set_position!(sys::Atoms, i::Integer, x)
    sys.atom_data.position[i] = x
end

# Run RWMC sampling for glycine at 300 K
println("\n" * "="^60)
println("Starting Random Walk Monte Carlo sampling")
println("="^60)

rwmc_initial = deepcopy(raw_data["train"][1])
T_rwmc = 300.0  # K
n_rwmc_samples = 100000
step_size_rwmc = 0.01  # Å - tune this for good acceptance rate (aim for ~0.2-0.5)
burnin_rwmc = 2000
thin_rwmc = 10

rwmc_samples, rwmc_acceptance, rwmc_energies = run_rwmc_sampling(
    rwmc_initial, model, n_rwmc_samples, T_rwmc;
    step_size=step_size_rwmc, burnin=burnin_rwmc, thin=thin_rwmc
)

# Plot RWMC energy distribution
histogram(rwmc_energies,
         bins=50,
         xlabel="Potential Energy (eV)",
         ylabel="Frequency",
         title="RWMC Sampled Energy Distribution at $(T_rwmc) K",
         legend=false,
         alpha=0.7)

# Plot energy trace to check for convergence
plot(1:length(rwmc_energies), rwmc_energies,
     xlabel="Sample Number",
     ylabel="Potential Energy (eV)",
     title="RWMC Energy Trace",
     linewidth=1,
     legend=false)

#%%
mean(potential_energy(d, model) for d in rwmc_samples)

mean(potential_energy(d, model) for d in raw_data["train"])