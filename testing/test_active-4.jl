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

using Distributed


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
ARDRegressor = sklearn_linear.ARDRegression
br_model = ARDRegressor(
        max_iter=300,
        tol=1e-3,
        threshold_lambda= 1e4,
        fit_intercept=false,
    )
# BayesianRidge(fit_intercept=false, 
#                         alpha_1=1e-6, 
#                         alpha_2=1e-6, 
#                         lambda_1=1e-6, 
#                         lambda_2=1e-6, 
#                         tol=1e-6, 
#                         max_iter=300)
br_model.fit(Awp["train"][I_active,:], Yw["train"][I_active])

Σ = pyconvert(Matrix, br_model.sigma_)
br_model.n_features_in_
br_model.sigma_
lambda = pyconvert(Vector, br_model.lambda_)
lambda .>= 1e-1

coef_tilde = pyconvert(Array, br_model.coef_)
sum(coef_tilde .== 0)

alpha = pyconvert(Float64, br_model.alpha_) # estimate of noise precision
coef = Psqrt \ coef_tilde
ACEpotentials.Models.set_linear_parameters!(model, coef)
alpha
xstar = Awp["train"][1,:]
size(Σ)
transpose(xstar) * Σ * xstar + 1/alpha
end

p_var = [pred_variance(Σ, Awp["train"][i,:], alpha) for i in 1:size(Awp["train"],1)] 

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
# Velocity-Verlet integrator using BAB Strang splitting

using Unitful: ustrip, @u_str

function velocity_verlet_step!(system, model, dt, f)
    """
    Perform one velocity-Verlet integration step using BAB Strang splitting
    
    Parameters:
    - system: AtomsBase system with positions and velocities
    - model: ACE potential for computing forces
    - dt: timestep (fs)
    
    BAB Strang splitting:
    - B: Half-step velocity update with current forces
    - A: Full-step position update with updated velocities
    - B: Half-step velocity update with new forces
    """
    natoms = length(system)
    
    # Get masses
    masses = [ustrip(u"u", atomic_mass(system, i)) for i in 1:natoms]
    
    # Conversion factor: 1 eV/Å / 1 amu = 98.2269 Å/fs²
    conversion = 98.2269
    
    # === B: Half-step velocity update ===
    # f = forces(system, model)
    f_matrix = hcat([ustrip.(u"eV/Å", f[i]) for i in 1:natoms]...)  # 3 x natoms
    
    velocities = velocity(system, :)
    for i in 1:natoms
        v_curr = ustrip.(u"Å/fs", velocities[i])
        a_curr = f_matrix[:, i] / masses[i] * conversion
        v_new = (v_curr .+ 0.5 * dt * a_curr) * u"Å/fs"
        set_velocity!(system, i, v_new)
    end
    
    # === A: Full-step position update ===
    positions = position(system, :)
    velocities = velocity(system, :)  # Get updated velocities
    for i in 1:natoms
        p_curr = positions[i]
        v = ustrip.(u"Å/fs", velocities[i])
        new_pos = p_curr .+ dt * v * u"Å"
        set_position!(system, i, new_pos)
    end
    
    # === B: Half-step velocity update with new forces ===
    f_new = forces(system, model)
    #_,f_new =queryASEModel(mflexiblesystem(system); calculator=:MACE)
    f_new_matrix = hcat([ustrip.(u"eV/Å", f_new[i]) for i in 1:natoms]...)
    
    velocities = velocity(system, :)
    for i in 1:natoms
        v_curr = ustrip.(u"Å/fs", velocities[i])
        a_new = f_new_matrix[:, i] / masses[i] * conversion
        v_final = (v_curr .+ 0.5 * dt * a_new) * u"Å/fs"
        set_velocity!(system, i, v_final)
    end
    
    return system, f_new
end




function kinetic_energy(system)
    """
    Compute kinetic energy of an Atoms structure
    
    K = (1/2) * Σ m_i * v_i^2
    
    Parameters:
    - system: AtomsBase system with velocities
    
    Returns: kinetic energy in eV
    """
    natoms = length(system)
    masses = [ustrip(u"u", atomic_mass(system, i)) for i in 1:natoms]
    
    # Conversion factor: 1 eV/Å / 1 amu = 98.2269 Å/fs²
    conversion = 98.2269
    
    K = 0.0
    for i in 1:natoms
        v = ustrip.(u"Å/fs", velocity(system, i))
        K += 0.5 * masses[i] * sum(v.^2) / conversion
    end
    
    return K
end

# Initialize system with velocities from Maxwell-Boltzmann distribution
function initialize_velocities!(system, T)
    """
    Initialize velocities from Maxwell-Boltzmann distribution at temperature T
    """
    kB = 8.617333262e-5  # eV/K
    conversion = 98.2269
    
    natoms = length(system)
    masses = [ustrip(u"u", atomic_mass(system, i)) for i in 1:natoms]
    
    for i in 1:natoms
        σ = sqrt(kB * T / masses[i] * conversion)  # velocity in Å/fs
        v = σ * randn(3) * u"Å/fs"
        set_velocity!(system, i, v)
    end
    
    return system
end

#%%
# HMC Sampler building on velocity_verlet_step!

function hamiltonian(system, model)
    """
    Compute total Hamiltonian H = U(q) + K(p)
    where U is potential energy and K is kinetic energy
    """
    U = ustrip(u"eV", potential_energy(system, model))
    K = kinetic_energy(system)
    return U + K
end

function hmc_step!(system, model, T, dt, L)
    """
    Perform one HMC step using velocity-Verlet integration
    
    Parameters:
    - system: AtomsBase system (will be modified in place or replaced)
    - model: ACE potential
    - T: temperature (K)
    - dt: leapfrog timestep (fs)
    - L: number of leapfrog steps
    
    Returns: (new_system, accepted)
    """
    kB = 8.617333262e-5  # eV/K
    
    # Sample new momenta (velocities) from Maxwell-Boltzmann
    initialize_velocities!(system, T)
    
    # Compute current Hamiltonian
    H_current = hamiltonian(system, model)
    
    # Create proposal by running dynamics
    proposal_system = deepcopy(system)
    FlexibleSystem(proposal_system)
    f = forces(proposal_system, model)
    
    for step in 1:L
        proposal_system, f = velocity_verlet_step!(proposal_system, model, dt, f)
    end
    
    # Compute proposed Hamiltonian
    H_proposal = hamiltonian(proposal_system, model)
    
    # Metropolis acceptance criterion
    ΔH = H_proposal - H_current
    accept_prob = exp(-ΔH / (kB * T))
    
    if rand() < accept_prob
        # Accept: copy proposal back to system
        for i in 1:length(system)
            set_position!(system, i, position(proposal_system, i))
            set_velocity!(system, i, velocity(proposal_system, i))
        end
        return system, true
    else
        # Reject: keep current system (but resample velocities for next iteration)
        return system, false
    end
end

function run_hmc_sampling(initial_system, model, n_samples, T; dt=0.01, L=10, burnin=100, thin=1)
    """
    Run HMC sampling using velocity-Verlet integration
    
    Parameters:
    - initial_system: starting configuration
    - model: ACE potential
    - n_samples: number of samples to collect (after burnin and thinning)
    - T: temperature (K)
    - dt: leapfrog timestep (fs), default 0.01
    - L: number of leapfrog steps per HMC proposal, default 10
    - burnin: number of initial steps to discard, default 100
    - thin: keep every thin-th sample, default 1
    
    Returns: (samples, acceptance_rate, energies)
    """
    samples = []
    energies = Float64[]
    current_system = deepcopy(initial_system)
    
    n_total = burnin + n_samples * thin
    n_accepted = 0
    
    println("Running HMC sampling with velocity-Verlet...")
    println("Parameters: T=$T K, dt=$dt fs, L=$L steps")
    println("Burnin: $burnin steps")
    println("Collecting $n_samples samples with thinning=$thin")
    
    for step in 1:n_total
        current_system, accepted = hmc_step!(current_system, model, T, dt, L)
        
        if accepted
            n_accepted += 1
        end
        
        # Collect samples after burnin with thinning
        if step > burnin && (step - burnin) % thin == 0
            push!(samples, deepcopy(current_system))
            push!(energies, ustrip(u"eV", potential_energy(current_system, model)))
        end
        
        if step % 100 == 0
            acc_rate = n_accepted / step
            println("Step $step / $n_total, Acceptance rate: $(round(acc_rate, digits=3))")
        end
    end
    
    acceptance_rate = n_accepted / n_total
    println("HMC sampling complete!")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate, energies
end

# Run HMC sampling for glycine at 300 K
println("\n" * "="^60)
println("Starting HMC sampling")
println("="^60)

hmc_initial = deepcopy(raw_data["train"][1])
T_hmc = 300.0  # K
n_hmc_samples = 100
dt_hmc = 0.01  # fs
L_hmc = 10
burnin_hmc = 100
thin_hmc = 5

hmc_samples, hmc_acceptance, hmc_energies = run_hmc_sampling(
    hmc_initial, model, n_hmc_samples, T_hmc;
    dt=dt_hmc, L=L_hmc, burnin=burnin_hmc, thin=thin_hmc
)

# Plot HMC energy distribution
histogram(hmc_energies,
         bins=30,
         xlabel="Potential Energy (eV)",
         ylabel="Frequency",
         title="HMC Sampled Energy Distribution at $(T_hmc) K",
         legend=false)

#%%
# Run velocity-Verlet simulation for glycine
using Random
Random.seed!(1234)  # For reproducibility
initial_system = deepcopy(raw_data["train"][1])
T = 300.0  # K
initialize_velocities!(initial_system, T)

system = initial_system

f = forces(system, model)
pot_energy_traj = Float64[]
kinetic_energy_traj = Float64[]
dt_sim = 0.01  # fs
n_sim_steps = 100


for step in 1:n_sim_steps
    system, f = velocity_verlet_step!(system, model, dt_sim, f)
    # E,_=queryASEModel(mflexiblesystem(system); calculator=:MACE)
    #push!(pot_energy_traj, E.val)
    push!(pot_energy_traj, potential_energy(system, model).val)
    push!(kinetic_energy_traj, kinetic_energy(system))
end


using ACESIDopt
using ACESIDopt: mflexiblesystem, queryASEModel

# fsystem = mflexiblesystem(system)
# E1,F1=queryASEModel(system; calculator=:MACE)
# E2,F2=queryASEModel(mflexiblesystem(system); calculator=:MACE)
# F1-F2
# Compute total energy
total_energy_traj = pot_energy_traj .+ kinetic_energy_traj

# Plot energy trajectory
time_array = (0:n_sim_steps) * dt_sim
n_max = 100
plot(time_array[2:n_max], total_energy_traj[2:n_max],
     xlabel="Time (fs)",
     ylabel="Energy (eV)",
     title="Energy Evolution in Velocity-Verlet Dynamics",
     linewidth=2,
     label="Total Energy",
     legend=:right)

plot(time_array[2:n_max], pot_energy_traj[2:n_max],
      linewidth=2,
      label="Potential Energy")
plot!(time_array[2:n_max], kinetic_energy_traj[2:n_max],
      linewidth=2,
      label="Kinetic Energy")

#%%


# natoms = length(system)
# masses = [ustrip(u"u", atomic_mass(system, i)) for i in 1:natoms]
    
# Conversion factor: 1 eV/Å / 1 amu = 98.2269 Å/fs²
conversion = 98.2269
    
    # Get current forces
    f = forces(system, model)
    set_velocity!(system, velocity(system,:) + .5 *  dt * f / masses) 



n_steps = 1000
dt = 0.0001  # fs
save_freq = 10

vv_trajectory, vv_energies = run_velocity_verlet(initial_system, model, n_steps, dt; save_freq=save_freq)

# Plot energy conservation
time_steps = 0:n_steps
plot(time_steps * dt, vv_energies["total"], 
     label="Total Energy",
     xlabel="Time (fs)",
     ylabel="Energy (eV)",
     title="Energy Conservation in Velocity-Verlet Dynamics",
     linewidth=2,
     legend=:right)
plot!(time_steps * dt, vv_energies["kinetic"], 
      label="Kinetic Energy",
      linewidth=2)
plot!(time_steps * dt, vv_energies["potential"], 
      label="Potential Energy",
      linewidth=2)

#%%
