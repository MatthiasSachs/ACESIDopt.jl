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

x =raw_data["train"][1]
potential_energy(x, model)
forces(x, model)




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

function AtomsBase.set_position!(sys::Atoms, positions)
    for (i,x) in enumerate(positions)
        AtomsBase.set_position!(sys, i, x)
    end
end

function AtomsBase.set_velocity!(sys::Atoms, i::Integer, vel)
    sys.atom_data.velocity[i] = vel
end

function AtomsBase.set_velocity!(sys::Atoms, velocities)
    for (i,vel) in enumerate(velocities)
        AtomsBase.set_velocity!(sys, i, vel)
    end
end

# set_position!(system,1, position(system,2) )
# set_position!(system, position(system,:))
# set_velocity!(system,1, velocity(system,2) )
# set_velocity!(system, velocity(system,:))

# These functions don't work yet for FlexibleSyste
# set_position!(fsystem,1, position(fsystem,2) )
# set_position!(fsystem, position(fsystem,:))
# set_velocity!(fsystem,1, velocity(fsystem,2) )
# set_velocity!(fsystem, velocity(fsystem,:))


# forces(ta, model)
# forces(fs, model)

using AtomsCalculators: potential_energy, forces
# potential_energy(system, model)

#%%
# Hamiltonian Monte Carlo (HMC) sampler for Gibbs-Boltzmann distribution

function kinetic_energy(momenta, masses)
    """
    Compute kinetic energy: K = Σ p²/(2m)
    momenta: 3 x natoms matrix (eV·fs/Å)
    masses: vector of masses (amu)
    Returns energy in eV
    """
    K = 0.0
    conversion = 1.0 / 98.2269  # Convert from (eV·fs/Å)²/amu to eV
    for i in 1:length(masses)
        K += sum(momenta[:, i].^2) / (2 * masses[i]) * conversion
    end
    return K
end

function hamiltonian(system, momenta, model, masses)
    """
    Compute total Hamiltonian H = U(q) + K(p)
    """
    U = ustrip(u"eV", potential_energy(system, model))
    K = kinetic_energy(momenta, masses)
    return U + K
end

function leapfrog_step(system, momenta, model, masses, ε)
    """
    One leapfrog integration step for Hamiltonian dynamics
    ε: step size (fs)
    """
    conversion = 98.2269  # eV/Å/amu to Å/fs²
    
    # Half step for momenta
    f = forces(system, model)
    f_matrix = hcat([ustrip.(u"eV/Å", f[i]) for i in 1:length(f)]...)  # 3 x natoms
    
    for i in 1:length(masses)
        momenta[:, i] .+= 0.5 * ε * f_matrix[:, i] * masses[i] / conversion
    end
    
    # Full step for positions
    positions = position(system, :)
    for i in 1:length(masses)
        new_pos = positions[i] .+ ε * momenta[:, i] / masses[i] * conversion * u"Å/fs" * u"fs"
        set_position!(system, i, new_pos)
    end
    
    # Half step for momenta with new forces
    # @show position(system, :)
    f = forces(system, model)
    f_matrix = hcat([ustrip.(u"eV/Å", f[i]) for i in 1:length(f)]...)
    
    for i in 1:length(masses)
        momenta[:, i] .+= 0.5 * ε * f_matrix[:, i] * masses[i] / conversion
    end
    
    return system, momenta
end

function sample_momenta(masses, T)
    """
    Sample momenta from Maxwell-Boltzmann distribution
    Returns momenta in units of eV·fs/Å
    """
    kB = 8.617333262e-5  # eV/K
    conversion = 98.2269  # to convert to eV·fs/Å
    
    momenta = zeros(3, length(masses))
    for i in 1:length(masses)
        # σ_p = sqrt(m * kB * T) in units of eV·fs/Å
        σ = sqrt(masses[i] * kB * T / conversion)
        momenta[:, i] = σ * randn(3)
    end
    return momenta
end

function hmc_step(system, model, masses, T, ε, L)
    """
    One HMC step with Metropolis acceptance
    
    Parameters:
    - system: current configuration
    - model: ACE potential
    - masses: atomic masses (amu)
    - T: temperature (K)
    - ε: leapfrog step size (fs)
    - L: number of leapfrog steps
    
    Returns: (new_system, accepted)
    """
    # Sample momenta
    p_current = sample_momenta(masses, T)
    
    # Calculate current Hamiltonian
    H_current = hamiltonian(system, p_current, model, masses)
    
    # Make a copy of the system for the proposal
    proposal_system = deepcopy(system)
    p_proposal = copy(p_current)
    
    # Leapfrog integration
    for step in 1:L
        proposal_system, p_proposal = leapfrog_step(proposal_system, p_proposal, model, masses, ε)
    end
    
    # Negate momenta (for reversibility, though not strictly necessary)
    p_proposal = -p_proposal
    
    # Calculate proposed Hamiltonian
    H_proposal = hamiltonian(proposal_system, p_proposal, model, masses)
    
    # Metropolis acceptance
    ΔH = H_proposal - H_current
    kB = 8.617333262e-5  # eV/K
    
    if rand() < exp(-ΔH / (kB * T))
        return proposal_system, true
    else
        return system, false
    end
end

function run_hmc_sampling(initial_system, model, n_samples, T; ε=0.1, L=10, burnin=100, thin=1)
    """
    Run HMC sampling to generate configurations from Gibbs-Boltzmann distribution
    
    Parameters:
    - initial_system: starting configuration
    - model: ACE potential
    - n_samples: number of samples to collect (after burnin and thinning)
    - T: temperature (K)
    - ε: leapfrog step size (fs), default 0.1
    - L: number of leapfrog steps per HMC step, default 10
    - burnin: number of initial steps to discard, default 100
    - thin: keep every thin-th sample, default 1
    
    Returns: (samples, acceptance_rate)
    """
    masses = [ustrip(u"u", atomic_mass(initial_system, i)) for i in 1:length(initial_system)]
    
    samples = []
    current_system = deepcopy(initial_system)
    
    n_total = burnin + n_samples * thin
    n_accepted = 0
    
    println("Running HMC sampling...")
    println("Burnin: $burnin steps")
    println("Collecting $n_samples samples with thinning=$thin")
    
    for step in 1:n_total
        current_system, accepted = hmc_step(current_system, model, masses, T, ε, L)
        
        if accepted
            n_accepted += 1
        end
        
        # Collect samples after burnin with thinning
        if step > burnin && (step - burnin) % thin == 0
            push!(samples, deepcopy(current_system))
        end
        
        if step % 100 == 0
            acc_rate = n_accepted / step
            println("Step $step / $n_total, Acceptance rate: $(round(acc_rate, digits=3))")
        end
    end
    
    acceptance_rate = n_accepted / n_total
    println("HMC sampling complete!")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate
end

# Run HMC sampling for glycine at 300 K
initial_system = raw_data["val"][1]
T = 300.0  # K
n_samples = 1000
ε = 0.005  # fs (step size)
L = 1  # number of leapfrog steps
burnin = 1
thin = 1

# forces(initial_system, model)
hmc_samples, acceptance_rate = run_hmc_sampling(initial_system, model, n_samples, T; 
                                                  ε=ε, L=L, burnin=burnin, thin=thin)
println("Generated $(length(hmc_samples)) HMC samples")

#%%
