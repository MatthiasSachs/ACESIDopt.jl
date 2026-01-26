module MSamplers

using AtomsBase
using ExtXYZ: Atoms
using AtomsCalculators: potential_energy, forces
using Unitful: ustrip, @u_str
using StaticArrays
using StaticArrays: SVector, SMatrix
using Distributed, ProgressMeter
using AtomsCalculators
using ACESIDopt: predictive_variance
export rwmc_step!, run_rwmc_sampling, mala_step!, run_mala_sampling, hmc_step!, set_position!, set_velocity!
export RWMCSampler, MALASampler, HMCSampler, run_sampler, step!, tune_sampler
export run_parallel_tempering, run_parallel_tempering_distributed

"""
    RWMCSampler

Random Walk Monte Carlo sampler configuration.

# Fields
- `step_size::Float64`: magnitude of random displacement (Å)

# Example
```julia
sampler = RWMCSampler(step_size=0.1)
samples, acc_rate, traj = run_sampler(sampler, system, model, T; n_samples=5000, burnin=2000, thin=5)
```
"""
struct RWMCSampler
    step_size::Float64
    
    function RWMCSampler(; step_size::Float64=0.01)
        @assert step_size > 0 "step_size must be positive"
        new(step_size)
    end
end

"""
    MALASampler

Metropolis-Adjusted Langevin Algorithm (MALA) sampler configuration.

MALA uses gradient information (forces) to guide proposals toward favorable
regions of configuration space, making it more efficient than random walk.

# Fields
- `step_size::Float64`: step size parameter (Å)

# Example
```julia
sampler = MALASampler(step_size=0.1)
samples, acc_rate, traj = run_sampler(sampler, system, model, T; n_samples=5000, burnin=2000, thin=5, collect_forces=false)
```
"""
struct MALASampler
    step_size::Float64
    
    function MALASampler(; step_size::Float64=0.01)
        @assert step_size > 0 "step_size must be positive"
        new(step_size)
    end
end

"""
    HMCSampler

Hamiltonian Monte Carlo (HMC) sampler configuration.

HMC uses Hamiltonian dynamics with momentum variables to propose moves,
allowing for efficient exploration of configuration space with high acceptance rates.
Uses leapfrog integration to simulate Hamiltonian dynamics.

# Fields
- `step_size::Float64`: leapfrog integration step size ε (Å)
- `n_leapfrog::Int`: number of leapfrog steps L per trajectory

# Example
```julia
sampler = HMCSampler(step_size=0.05, n_leapfrog=10)
samples, acc_rate, traj = run_sampler(sampler, system, model, T; n_samples=5000, burnin=2000, thin=5)
```
"""
struct HMCSampler
    step_size::Float64
    n_leapfrog::Int
    
    function HMCSampler(; step_size::Float64=0.01, n_leapfrog::Int=10)
        @assert step_size > 0 "step_size must be positive"
        @assert n_leapfrog > 0 "n_leapfrog must be positive"
        new(step_size, n_leapfrog)
    end
end

# Setter functions for AtomsBase.Atoms
import AtomsBase: set_position!, set_velocity!

function AtomsBase.set_position!(sys::Atoms, i::Integer, x::SVector)
    sys.atom_data.position[i] = x
end

function AtomsBase.set_position!(sys::Atoms, i::Integer, x::Vector) 
    sys.atom_data.position[i] = copy(x)
end

function AtomsBase.set_position!(sys::Atoms, positions)
    for (i,x) in enumerate(positions)
        AtomsBase.set_position!(sys, i, x)
    end
end

function AtomsBase.set_velocity!(sys::Atoms, i::Integer, vel::SVector)
    sys.atom_data.velocity[i] = vel
end
function AtomsBase.set_velocity!(sys::Atoms, i::Integer, vel::Vector)
    sys.atom_data.velocity[i] = copy(vel)
end

function AtomsBase.set_velocity!(sys::Atoms, velocities)
    for (i,vel) in enumerate(velocities)
        AtomsBase.set_velocity!(sys, i, vel)
    end
end

function copy_positions(system::Atoms)
    return [copy(system.atom_data.position[i]) for i in 1:length(system)]
end


"""
    step!(system, sampler::RWMCSampler, model, T)

Perform one RWMC step using sampler configuration.

# Returns
- `(accepted, ΔU, U_current, nothing)`: acceptance flag, energy change, current energy, forces (none for RWMC)
"""
function step!(system, sampler::RWMCSampler, model, T)
    _, accepted, ΔU, U_current = rwmc_step!(system, model, T, sampler.step_size)
    return accepted, ΔU, U_current, nothing
end

"""
    step!(system, sampler::MALASampler, model, T)

Perform one MALA step using sampler configuration.

# Returns
- `(accepted, ΔU, U_current, f_current)`: acceptance flag, energy change, current energy, forces
"""
function step!(system, sampler::MALASampler, model, T)
    _, accepted, ΔU, U_current, f_current = mala_step!(system, model, T, sampler.step_size)
    return accepted, ΔU, U_current, f_current
end

"""
    step!(system, sampler::HMCSampler, model, T)

Perform one HMC step using sampler configuration.

# Returns
- `(accepted, ΔU, U_current, f_current)`: acceptance flag, energy change, current energy, forces
"""
function step!(system, sampler::HMCSampler, model, T)
    _, accepted, ΔU, U_current, f_current = hmc_step!(system, model, T, sampler.step_size, sampler.n_leapfrog)
    return accepted, ΔU, U_current, f_current
end

"""
    tune_sampler(sampler, system, model, T; 
                 n_tune=1000, target_accept=nothing, 
                 adapt_rate=0.1, min_step=1e-6, max_step=10.0)

Adaptively tune the step size of a sampler to achieve a target acceptance rate.

Uses a simple adaptive scheme where step size is adjusted based on the running
acceptance rate during a tuning phase. The target acceptance rates are:
- RWMC: 0.234 (optimal for high-dimensional Gaussian targets)
- MALA: 0.574 (optimal for high-dimensional with gradients)
- HMC: 0.651 (optimal for Hamiltonian Monte Carlo)

# Parameters
- `sampler`: RWMCSampler, MALASampler, or HMCSampler to tune
- `system`: atomic system for tuning
- `model`: potential model
- `T`: temperature (K)
- `n_tune`: number of tuning steps (default: 1000)
- `target_accept`: target acceptance rate (default: 0.234 for RWMC, 0.574 for MALA)
- `adapt_rate`: adaptation rate for step size adjustment (default: 0.1)
- `min_step`: minimum allowed step size (Å, default: 1e-6)
- `max_step`: maximum allowed step size (Å, default: 10.0)

# Returns
- New sampler with tuned step size
- Final acceptance rate during tuning
- Tuning trajectory with (step_sizes, acceptance_rates)

# Example
```julia
# Initial sampler with guess
sampler = RWMCSampler(step_size=0.1)

# Tune to optimal acceptance rate
tuned_sampler, acc_rate, traj = tune_sampler(sampler, system, model, 300.0; n_tune=2000)

# Use tuned sampler for production sampling
samples, _, _ = run_sampler(tuned_sampler, system, model, 300.0; n_samples=10000)
```
"""
function tune_sampler(sampler, system, model, T; 
                      n_tune::Int=1000, 
                      target_accept::Union{Nothing,Float64}=nothing,
                      adapt_rate::Float64=0.1,
                      min_step::Float64=1e-6,
                      max_step::Float64=10.0)
    
    # Set default target acceptance rates based on sampler type
    if target_accept === nothing
        if sampler isa RWMCSampler
            target_accept = 0.234  # Optimal for RWMC
        elseif sampler isa MALASampler
            target_accept = 0.574  # Optimal for MALA
        elseif sampler isa HMCSampler
            target_accept = 0.651  # Optimal for HMC
        else
            target_accept = 0.234  # Default fallback
        end
    end
    
    current_step_size = sampler.step_size
    tuning_system = deepcopy(system)
    
    n_accepted = 0
    step_sizes = Float64[]
    acceptance_rates = Float64[]
    
    sampler_type = typeof(sampler)
    println("Tuning $(sampler_type)...")
    println("Initial step size: $(round(current_step_size, digits=6)) Å")
    println("Target acceptance rate: $(round(target_accept, digits=3))")
    println("Tuning for $n_tune steps...")
    
    for i in 1:n_tune
        # Create temporary sampler with current step size
        if sampler isa RWMCSampler
            temp_sampler = RWMCSampler(step_size=current_step_size)
        elseif sampler isa MALASampler
            temp_sampler = MALASampler(step_size=current_step_size)
        elseif sampler isa HMCSampler
            temp_sampler = HMCSampler(step_size=current_step_size, n_leapfrog=sampler.n_leapfrog)
        end
        
        # Take a step
        accepted, _, _, _ = step!(tuning_system, temp_sampler, model, T)
        
        if accepted
            n_accepted += 1
        end
        
        # Calculate running acceptance rate
        current_accept_rate = n_accepted / i
        
        # Record trajectory every 10 steps
        if i % 10 == 0
            push!(step_sizes, current_step_size)
            push!(acceptance_rates, current_accept_rate)
        end
        
        # Adapt step size every 50 steps after initial 100 steps
        if i > 100 && i % 50 == 0
            # Adjust step size based on acceptance rate
            # If accept rate too high, increase step size
            # If accept rate too low, decrease step size
            ratio = current_accept_rate / target_accept
            
            # Use logarithmic adaptation for stability
            log_adjustment = adapt_rate * (ratio - 1.0)
            new_step_size = current_step_size * exp(log_adjustment)
            
            # Clamp to reasonable bounds
            current_step_size = clamp(new_step_size, min_step, max_step)
            
            if i % 200 == 0
                println("  Step $i: accept_rate=$(round(current_accept_rate, digits=3)), " *
                       "step_size=$(round(current_step_size, digits=6)) Å")
            end
        end
    end
    
    final_accept_rate = n_accepted / n_tune
    println("Tuning complete!")
    println("Final step size: $(round(current_step_size, digits=6)) Å")
    println("Final acceptance rate: $(round(final_accept_rate, digits=3))")
    
    # Create tuned sampler
    if sampler isa RWMCSampler
        tuned_sampler = RWMCSampler(step_size=current_step_size)
    elseif sampler isa MALASampler
        tuned_sampler = MALASampler(step_size=current_step_size)
    elseif sampler isa HMCSampler
        tuned_sampler = HMCSampler(step_size=current_step_size, n_leapfrog=sampler.n_leapfrog)
    else
        error("Unknown sampler type")
    end
    
    traj = (step_sizes=step_sizes, acceptance_rates=acceptance_rates)
    
    return tuned_sampler, final_accept_rate, traj
end

"""
    rwmc_step!(system, model, T, step_size)

Perform one Random Walk Monte Carlo step

# Parameters
- `system`: AtomsBase system (modified in place)
- `model`: ACE potential
- `T`: temperature (K)
- `step_size`: magnitude of random displacement (Å)

# Returns
- `(system, accepted, ΔU)`: modified system, acceptance flag, energy change
"""
function rwmc_step!(system, model, T, step_size)
    kB = 8.617333262e-5  # eV/K
    natoms = length(system)
    
    # Compute current energy
    U_current = ustrip(u"eV", potential_energy(system, model))
    
    # Store current positions
    current_positions = deepcopy(position(system, :))

    # Propose new positions: random displacement
    proposal_positions = position(system, :)


    for i in 1:natoms
        # Random displacement in each direction
        displacement = step_size * (randn(3) * u"Å")
        proposal_positions[i] = proposal_positions[i] .+ displacement
    end
    
    # Update system with proposed positions
    for i in 1:natoms
        set_position!(system, i, proposal_positions[i])
    end
    
    # Compute proposed energy
    U_proposal = ustrip(u"eV", potential_energy(system, model))
    
    # Metropolis acceptance criterion
    ΔU = U_proposal - U_current
    accept_prob = min(1.0, exp(-ΔU / (kB * T)))
    
    # @show current_positions-proposal_positions
    if rand() < accept_prob
        # Accept: keep new positions
        return system, true, ΔU, U_proposal
    else
        # Reject: restore old positions
        for i in 1:natoms
            set_position!(system, i, current_positions[i])
        end
        return system, false, ΔU, U_current
    end
end

"""
    run_rwmc_sampling(initial_system, model, n_samples, T; step_size=0.01, burnin=1000, thin=10)

Run Random Walk Monte Carlo sampling from Gibbs-Boltzmann distribution

# Parameters
- `initial_system`: starting configuration
- `model`: ACE potential defining the energy surface
- `n_samples`: number of samples to collect (after burnin and thinning)
- `T`: temperature (K)
- `step_size`: standard deviation of random displacement (Å), default 0.01
- `burnin`: number of initial steps to discard, default 1000
- `thin`: keep every thin-th sample, default 10

# Returns
- `(samples, acceptance_rate, traj)`: collected samples, acceptance rate, trajectory data
"""
function run_rwmc_sampling(initial_system, model, n_samples, T; step_size=0.01, burnin=1000, thin=10)
    samples = []
    traj = (energy=Float64[],)
    current_system = deepcopy(initial_system)
    
    n_total = burnin + n_samples * thin
    n_accepted = 0
    
    println("Running Random Walk Monte Carlo sampling...")
    println("Parameters: T=$T K, step_size=$step_size Å")
    println("Burnin: $burnin steps")
    println("Collecting $n_samples samples with thinning=$thin")
    
    for step in 1:n_total
        current_system, accepted, ΔU, U_current = rwmc_step!(current_system, model, T, step_size)
        
        if accepted
            n_accepted += 1
        end
        
        # Collect samples after burnin with thinning
        if step > burnin && (step - burnin) % thin == 0
            # Create new Atoms structure with updated energy
            updated_system_data = merge(current_system.system_data, (energy=U_current,))
            sample_system = Atoms(deepcopy(current_system.atom_data), deepcopy(updated_system_data))
            push!(samples, sample_system)
            push!(traj.energy, U_current)
        end
        
        if step % 1000 == 0
            acc_rate = n_accepted / step
            println("Step $step / $n_total, Acceptance rate: $(round(acc_rate, digits=3))")
        end
    end
    
    acceptance_rate = n_accepted / n_total
    println("RWMC sampling complete!")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate, traj
end

"""    mala_step!(system, model, T, step_size)

Perform one Metropolis-Adjusted Langevin Algorithm (MALA) step

MALA uses gradient information to propose moves that respect the energy landscape,
combining Langevin dynamics with Metropolis acceptance.

Proposal: x' = x + (step_size^2/2) * ∇U(x) + step_size * ξ
where ξ ~ N(0, I) is Gaussian noise

# Parameters
- `system`: AtomsBase system (modified in place)
- `model`: ACE potential
- `T`: temperature (K)
- `step_size`: step size parameter (Å)

# Returns
- `(system, accepted, ΔU)`: modified system, acceptance flag, energy change
"""
function mala_step!(system, model, T, step_size)
    kB = 8.617333262e-5  # eV/K
    natoms = length(system)
    
    # Get current positions and energy
    current_positions = deepcopy(position(system, :))
    U_current = ustrip(u"eV", potential_energy(system, model))
    
    # Get forces (negative gradient of potential)
    f_current = forces(system, model)
    
    # Propose new positions using Langevin dynamics
    # x' = x - (step_size^2/(2*kB*T)) * ∇U(x) + step_size * ξ
    # Note: forces = -∇U, so we add forces
    proposal_positions = position(system, :)
    for i in 1:natoms
        f_i = ustrip.(u"eV/Å", f_current[i])
        drift = (step_size^2 / (2 * kB * T)) * f_i * u"Å"
        noise = step_size * randn(3) * u"Å"
        proposal_positions[i] = proposal_positions[i] .+ drift .+ noise
    end
    
    # Update system with proposed positions

    set_position!(system, proposal_positions)

    
    # Compute proposed energy and forces
    U_proposal = ustrip(u"eV", potential_energy(system, model))
    f_proposal = forces(system, model)
    
    # Compute forward and reverse transition probabilities (log)
    # q(x'|x) ∝ exp(-|x' - x - drift_x|^2 / (2*step_size^2))
    # q(x|x') ∝ exp(-|x - x' - drift_x'|^2 / (2*step_size^2))
    
    log_q_forward = 0.0
    log_q_reverse = 0.0
    
    for i in 1:natoms
        # Forward: from x to x'
        f_i = ustrip.(u"eV/Å", f_current[i])
        drift_forward = (step_size^2 / (2 * kB * T)) * f_i
        diff_forward = ustrip.(u"Å", proposal_positions[i] - current_positions[i]) - drift_forward
        log_q_forward -= sum(diff_forward.^2) / (2 * step_size^2)
        
        # Reverse: from x' to x
        f_i_prop = ustrip.(u"eV/Å", f_proposal[i])
        drift_reverse = (step_size^2 / (2 * kB * T)) * f_i_prop
        diff_reverse = ustrip.(u"Å", current_positions[i] - proposal_positions[i]) - drift_reverse
        log_q_reverse -= sum(diff_reverse.^2) / (2 * step_size^2)
    end
    
    # Metropolis-Hastings acceptance criterion
    ΔU = U_proposal - U_current
    log_accept_prob = -(ΔU / (kB * T)) + log_q_reverse - log_q_forward
    
    if log(rand()) < log_accept_prob
        # Accept: keep new positions
        return system, true, ΔU, U_proposal, f_proposal
    else
        # Reject: restore old positions
        set_position!(system, current_positions)

        return system, false, ΔU, U_current, f_current
    end
end

"""
    run_mala_sampling(initial_system, model, n_samples, T; step_size=0.01, burnin=1000, thin=10)

Run Metropolis-Adjusted Langevin Algorithm (MALA) sampling from Gibbs-Boltzmann distribution

MALA is more efficient than random walk because it uses gradient information
to guide proposals toward favorable regions of configuration space.

# Parameters
- `initial_system`: starting configuration
- `model`: ACE potential defining the energy surface
- `n_samples`: number of samples to collect (after burnin and thinning)
- `T`: temperature (K)
- `step_size`: step size parameter (Å), default 0.01
- `burnin`: number of initial steps to discard, default 1000
- `thin`: keep every thin-th sample, default 10

# Returns
- `(samples, acceptance_rate, energies)`: collected samples, acceptance rate, energies
"""
function run_mala_sampling(initial_system, model, n_samples, T; step_size=0.01, burnin=1000, thin=10, collect_forces=false)
    samples = []
    if collect_forces
        traj = (energy = Float64[], forces = [])
    else
        traj = (energy=Float64[],)
    end
    current_system = deepcopy(initial_system)
    
    n_total = burnin + n_samples * thin
    n_accepted = 0
    
    println("Running Metropolis-Adjusted Langevin Algorithm (MALA) sampling...")
    println("Parameters: T=$T K, step_size=$step_size Å")
    println("Burnin: $burnin steps")
    println("Collecting $n_samples samples with thinning=$thin")
    
    for step in 1:n_total
        current_system, accepted, ΔU, U_current, f_current = mala_step!(current_system, model, T, step_size)
        
        if accepted
            n_accepted += 1
        end
        
        # Collect samples after burnin with thinning
        if step > burnin && (step - burnin) % thin == 0
            push!(samples, deepcopy(current_system))
            push!(traj.energy, U_current)
            if collect_forces
                push!(traj.forces, deepcopy(f_current))
            end
        end
        
        if step % 1000 == 0
            acc_rate = n_accepted / step
            println("Step $step / $n_total, Acceptance rate: $(round(acc_rate, digits=3))")
        end
    end
    
    acceptance_rate = n_accepted / n_total
    println("MALA sampling complete!")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    if collect_forces
        return samples, acceptance_rate, traj
    else
        return samples, acceptance_rate, traj
    end
end

"""
    hmc_step!(system, model, T, step_size, n_leapfrog)

Perform one Hamiltonian Monte Carlo (HMC) step.

HMC uses Hamiltonian dynamics with fictitious momentum variables to propose
new states. It simulates the dynamics using leapfrog integration, which is
symplectic and time-reversible, preserving the Hamiltonian.

The Hamiltonian is: H(x,p) = U(x) + K(p) where
- U(x) is the potential energy
- K(p) = p²/(2m) is the kinetic energy (we use m=1 for simplicity)

# Parameters
- `system`: AtomsBase system (modified in place)
- `model`: potential model
- `T`: temperature (K)
- `step_size`: leapfrog integration step size ε (Å)
- `n_leapfrog`: number of leapfrog steps L

# Returns
- `(system, accepted, ΔU, U_current, f_current)`: modified system, acceptance flag, energy change, current energy, forces

# Algorithm
1. Sample momentum from Gaussian: p ~ N(0, m*kB*T)
2. Compute current Hamiltonian: H = U(x) + K(p)
3. Simulate dynamics via leapfrog integration for L steps
4. Compute new Hamiltonian: H' = U(x') + K(p')
5. Accept/reject via Metropolis: α = min(1, exp(-(H' - H)/(kB*T)))
"""
function hmc_step!(system, model, T, step_size, n_leapfrog)
    kB = 8.617333262e-5  # eV/K
    natoms = length(system)
    
    # Get current positions and energy
    current_positions = deepcopy(position(system, :))
    U_current = ustrip(u"eV", potential_energy(system, model))
    
    # Sample momentum from Gaussian distribution
    # p ~ N(0, m*kB*T*I) where we use m=1 (unit mass)
    # Momentum has same units as sqrt(mass * energy) = sqrt(eV * mass)
    # For simplicity, we work in units where momentum ~ sqrt(kB*T)
    momentum = [sqrt(kB * T) * randn(3) for _ in 1:natoms]
    
    # Compute initial kinetic energy: K = sum(p²/(2m)) with m=1
    K_current = sum(sum(p.^2) for p in momentum) / 2.0
    
    # Current Hamiltonian
    H_current = U_current + K_current
    
    # Make copies for leapfrog integration
    x = deepcopy(current_positions)
    p = deepcopy(momentum)
    
    # Get initial forces (negative gradient of potential)
    f = forces(system, model)
    
    # Leapfrog integration
    # Half step for momentum
    for i in 1:natoms
        f_i = ustrip.(u"eV/Å", f[i])
        p[i] = p[i] .+ (step_size / 2.0) * f_i
    end
    
    # Alternate full steps for position and momentum
    for step in 1:n_leapfrog-1
        # Full step for position
        for i in 1:natoms
            x[i] = x[i] .+ step_size * p[i] * u"Å"
        end
        
        # Update system to get new forces
        set_position!(system, x)
        f = forces(system, model)
        
        # Full step for momentum
        for i in 1:natoms
            f_i = ustrip.(u"eV/Å", f[i])
            p[i] = p[i] .+ step_size * f_i
        end
    end
    
    # Final full step for position
    for i in 1:natoms
        x[i] = x[i] .+ step_size * p[i] * u"Å"
    end
    
    # Update system to get final energy and forces
    set_position!(system, x)
    U_proposal = ustrip(u"eV", potential_energy(system, model))
    f_proposal = forces(system, model)
    
    # Half step for momentum
    for i in 1:natoms
        f_i = ustrip.(u"eV/Å", f_proposal[i])
        p[i] = p[i] .+ (step_size / 2.0) * f_i
    end
    
    # Compute final kinetic energy
    K_proposal = sum(sum(p_i.^2) for p_i in p) / 2.0
    
    # Proposed Hamiltonian
    H_proposal = U_proposal + K_proposal
    
    # Metropolis acceptance criterion
    # Note: The Hamiltonian already includes kB*T scaling in kinetic energy
    # So we need to be careful about temperature scaling
    ΔH = H_proposal - H_current
    accept_prob = min(1.0, exp(-ΔH / (kB * T)))
    
    ΔU = U_proposal - U_current
    
    if rand() < accept_prob
        # Accept: keep new positions (already set)
        return system, true, ΔU, U_proposal, f_proposal
    else
        # Reject: restore old positions
        set_position!(system, current_positions)
        f_current = forces(system, model)
        return system, false, ΔU, U_current, f_current
    end
end

"""
    run_sampler(sampler, initial_system, model, T; n_samples=1000, burnin=1000, thin=10, collect_forces=false)

Run MCMC sampling using the specified sampler configuration.

This is a unified interface that works with both RWMCSampler and MALASampler.
The dispatch to specific algorithms happens at the step! level.

# Parameters
- `sampler`: Either RWMCSampler or MALASampler instance (contains step_size)
- `initial_system`: starting configuration
- `model`: potential model (e.g., ACE potential or HarmonicCalculator)
- `T`: temperature (K)
- `n_samples`: number of samples to collect (after burnin and thinning)
- `burnin`: number of initial steps to discard
- `thin`: keep every thin-th sample
- `collect_forces`: whether to collect forces in trajectory (only applies to MALA)

# Returns
- `(samples, acceptance_rate, traj)`: collected samples, acceptance rate, trajectory data

# Examples
```julia
# RWMC sampling
rwmc = RWMCSampler(step_size=0.1)
samples, acc_rate, traj = run_sampler(rwmc, system, model, 300.0; n_samples=5000, burnin=2000, thin=5, collect_forces=true)

# MALA sampling
mala = MALASampler(step_size=0.1)
samples, acc_rate, traj = run_sampler(mala, system, model, 300.0; n_samples=5000, burnin=2000, thin=5, collect_forces=true)
```
"""
function run_sampler(sampler, initial_system, model, T; n_samples::Int=1000, burnin::Int=1000, thin::Int=10, collect_forces::Bool=false)
    samples = []
    if collect_forces
        traj = (energy=Float64[], forces=[], acc_rate=Float64[])
    else
        traj = (energy=Float64[], acc_rate=Float64[])
    end
    current_system = deepcopy(initial_system)
    
    n_total = burnin + n_samples * thin
    n_accepted = 0
    n_accepted_since_last_sample = 0
    steps_since_last_sample = 0
    
    sampler_type = typeof(sampler)
    println("Running $(sampler_type) sampling...")
    println("Parameters: T=$T K, step_size=$(sampler.step_size) Å")
    println("Burnin: $burnin steps")
    println("Collecting $n_samples samples with thinning=$thin")
    if collect_forces && sampler isa MALASampler
        println("Force collection: enabled")
    end
    
    @showprogress for step_num in 1:n_total
        accepted, ΔU, U_current, f_current = step!(current_system, sampler, model, T)
        
        if accepted
            n_accepted += 1
            n_accepted_since_last_sample += 1
        end
        steps_since_last_sample += 1
        
        # Collect samples after burnin with thinning
        if step_num > burnin && (step_num - burnin) % thin == 0
            push!(samples, deepcopy(current_system))
            push!(traj.energy, U_current)
            
            # Compute acceptance rate since last sample
            local_acc_rate = steps_since_last_sample > 0 ? n_accepted_since_last_sample / steps_since_last_sample : 0.0
            push!(traj.acc_rate, local_acc_rate)
            
            # Reset counters
            n_accepted_since_last_sample = 0
            steps_since_last_sample = 0
            
            if collect_forces 
                if f_current !== nothing
                    push!(traj.forces, deepcopy(f_current))
                else
                    push!(traj.forces, forces(current_system, model))
                end
            end
        end
        
        if step_num % 1000 == 0
            acc_rate = n_accepted / step_num
            println("Step $step_num / $n_total, Acceptance rate: $(round(acc_rate, digits=3))")
        end
    end
    
    acceptance_rate = n_accepted / n_total
    println("$(sampler_type) sampling complete!")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate, traj
end

"""
    run_parallel_tempering(sampler, initial_system, model, n_replicas, T_min, T_max;
                          n_samples=1000, burnin=1000, thin=10, 
                          exchange_interval=10, collect_forces=false)

Run Parallel Tempering (Replica Exchange) Monte Carlo sampling.

Parallel tempering runs multiple replicas at different temperatures simultaneously
and periodically attempts to exchange configurations between neighboring temperature
levels. This helps overcome energy barriers and improves sampling efficiency.

The temperature schedule uses geometric spacing: T_i = T_min * (T_max/T_min)^(i/(n_replicas-1))

# Parameters
- `sampler`: Sampler configuration (RWMCSampler, MALASampler, or HMCSampler)
- `initial_system`: starting configuration (will be replicated for all temperatures)
- `model`: potential model
- `n_replicas`: number of temperature replicas
- `T_min`: minimum temperature (K)
- `T_max`: maximum temperature (K)
- `n_samples`: number of samples to collect per replica (after burnin and thinning)
- `burnin`: number of initial steps to discard
- `thin`: keep every thin-th sample
- `exchange_interval`: attempt replica exchange every N steps
- `collect_forces`: whether to collect forces in trajectory (only for MALA/HMC)

# Returns
- `replicas`: array of sample arrays, one per temperature
- `temperatures`: temperature values for each replica
- `acceptance_rates`: MCMC acceptance rate for each replica
- `exchange_rates`: exchange acceptance rate between neighboring replicas
- `trajectories`: trajectory data for each replica

# Example
```julia
sampler = HMCSampler(step_size=0.05, n_leapfrog=10)
replicas, temps, acc_rates, exch_rates, trajs = run_parallel_tempering(
    sampler, system, model, 8, 300.0, 1000.0;
    n_samples=5000, burnin=2000, thin=5, exchange_interval=10
)

# Get samples from target temperature (lowest)
target_samples = replicas[1]
```
"""
function run_parallel_tempering(sampler, initial_system, model, n_replicas::Int, 
                                T_min::Float64, T_max::Float64;
                                n_samples::Int=1000, burnin::Int=1000, thin::Int=10,
                                exchange_interval::Int=10, collect_forces::Bool=false)
    
    @assert n_replicas >= 2 "Need at least 2 replicas for parallel tempering"
    @assert T_max > T_min "T_max must be greater than T_min"
    @assert exchange_interval > 0 "exchange_interval must be positive"
    
    kB = 8.617333262e-5  # eV/K
    
    # Create temperature schedule (geometric spacing)
    if n_replicas == 1
        temperatures = [T_min]
    else
        temperatures = [T_min * (T_max/T_min)^((i-1)/(n_replicas-1)) for i in 1:n_replicas]
    end
    
    println("="^70)
    println("Parallel Tempering Monte Carlo")
    println("="^70)
    println("Number of replicas: $n_replicas")
    println("Temperature range: $(round(T_min, digits=1)) K to $(round(T_max, digits=1)) K")
    println("Temperature schedule:")
    for (i, T) in enumerate(temperatures)
        println("  Replica $i: $(round(T, digits=2)) K")
    end
    println("Sampler type: $(typeof(sampler))")
    println("Exchange interval: every $exchange_interval steps")
    println("="^70)
    
    # Initialize replicas (deep copy of initial system for each temperature)
    systems = [deepcopy(initial_system) for _ in 1:n_replicas]
    
    # Initialize storage for samples and trajectories
    all_samples = [[] for _ in 1:n_replicas]
    if collect_forces
        all_trajs = [(energy=Float64[], forces=[], acc_rate=Float64[]) for _ in 1:n_replicas]
    else
        all_trajs = [(energy=Float64[], acc_rate=Float64[]) for _ in 1:n_replicas]
    end
    
    # Track acceptance statistics
    n_accepted_mcmc = zeros(Int, n_replicas)
    n_accepted_since_last_sample = zeros(Int, n_replicas)
    steps_since_last_sample = zeros(Int, n_replicas)
    n_attempted_exchange = zeros(Int, n_replicas-1)
    n_accepted_exchange = zeros(Int, n_replicas-1)
    
    n_total = burnin + n_samples * thin
    
    println("\nRunning parallel tempering for $n_total steps...")
    println("Burnin: $burnin steps, then collecting $n_samples samples with thinning=$thin")
    
    # Main sampling loop
    for step_num in 1:n_total
        # Perform MCMC steps for all replicas
        for (i, T) in enumerate(temperatures)
            accepted, ΔU, U_current, f_current = step!(systems[i], sampler, model, T)
            
            if accepted
                n_accepted_mcmc[i] += 1
                n_accepted_since_last_sample[i] += 1
            end
            steps_since_last_sample[i] += 1
            
            # Collect samples after burnin with thinning
            if step_num > burnin && (step_num - burnin) % thin == 0
                push!(all_samples[i], deepcopy(systems[i]))
                push!(all_trajs[i].energy, U_current)
                
                # Compute acceptance rate since last sample
                local_acc_rate = steps_since_last_sample[i] > 0 ? n_accepted_since_last_sample[i] / steps_since_last_sample[i] : 0.0
                push!(all_trajs[i].acc_rate, local_acc_rate)
                
                # Reset counters
                n_accepted_since_last_sample[i] = 0
                steps_since_last_sample[i] = 0
                
                if collect_forces && f_current !== nothing
                    push!(all_trajs[i].forces, deepcopy(f_current))
                end
            end
        end
        
        # Attempt replica exchanges
        if step_num % exchange_interval == 0
            # Randomly choose whether to exchange even or odd pairs
            # This ensures detailed balance
            offset = rand(0:1)
            
            for i in (1+offset):2:(n_replicas-1)
                if i >= n_replicas
                    break
                end
                
                # Propose exchange between replica i and i+1
                n_attempted_exchange[i] += 1
                
                # Get energies
                U_i = ustrip(u"eV", potential_energy(systems[i], model))
                U_j = ustrip(u"eV", potential_energy(systems[i+1], model))
                
                T_i = temperatures[i]
                T_j = temperatures[i+1]
                
                # Metropolis criterion for exchange
                # P_accept = min(1, exp(ΔE * Δβ)) where ΔE = U_j - U_i, Δβ = 1/(kB*T_i) - 1/(kB*T_j)
                β_i = 1.0 / (kB * T_i)
                β_j = 1.0 / (kB * T_j)
                
                Δ = (β_i - β_j) * (U_j - U_i)
                accept_prob = min(1.0, exp(Δ))
                
                if rand() < accept_prob
                    # Accept exchange: swap the systems
                    systems[i], systems[i+1] = systems[i+1], systems[i]
                    n_accepted_exchange[i] += 1
                end
            end
        end
        
        # Progress reporting
        if step_num % 1000 == 0
            println("\nStep $step_num / $n_total:")
            for i in 1:n_replicas
                acc_rate = n_accepted_mcmc[i] / step_num
                println("  Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
                       "MCMC accept=$(round(acc_rate, digits=3))")
            end
            if step_num >= exchange_interval
                for i in 1:(n_replicas-1)
                    if n_attempted_exchange[i] > 0
                        exch_rate = n_accepted_exchange[i] / n_attempted_exchange[i]
                        println("  Exchange $i↔$(i+1): accept=$(round(exch_rate, digits=3))")
                    end
                end
            end
        end
    end
    
    # Calculate final statistics
    mcmc_acceptance_rates = n_accepted_mcmc ./ n_total
    exchange_acceptance_rates = zeros(n_replicas-1)
    for i in 1:(n_replicas-1)
        if n_attempted_exchange[i] > 0
            exchange_acceptance_rates[i] = n_accepted_exchange[i] / n_attempted_exchange[i]
        end
    end
    
    println("\n" * "="^70)
    println("Parallel Tempering Complete!")
    println("="^70)
    println("\nMCMC Acceptance Rates:")
    for i in 1:n_replicas
        println("  Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
               "$(round(mcmc_acceptance_rates[i], digits=3))")
    end
    println("\nExchange Acceptance Rates:")
    for i in 1:(n_replicas-1)
        println("  Exchange $i↔$(i+1) ($(round(temperatures[i], digits=1)) K ↔ " *
               "$(round(temperatures[i+1], digits=1)) K): " *
               "$(round(exchange_acceptance_rates[i], digits=3))")
    end
    println("="^70)
    
    return all_samples, temperatures, mcmc_acceptance_rates, exchange_acceptance_rates, all_trajs
end

"""
    run_parallel_tempering_distributed(sampler, initial_system, model, n_replicas, T_min, T_max;
                                       n_samples=1000, burnin=1000, thin=10, 
                                       exchange_interval=10, collect_forces=false)

Run Parallel Tempering (Replica Exchange) Monte Carlo with distributed computing.

This is a parallelized version of parallel tempering that uses Julia's Distributed
computing to run replicas on different workers/processors simultaneously. Each replica
evolves independently between exchange attempts, maximizing parallel efficiency.

**Requirements:** Must have workers available via `addprocs()` before calling this function.
The necessary modules must be loaded on all workers using `@everywhere`.

# Parameters
- `sampler`: Sampler configuration (RWMCSampler, MALASampler, or HMCSampler)
- `initial_system`: starting configuration (will be replicated for all temperatures)
- `model`: potential model
- `n_replicas`: number of temperature replicas
- `T_min`: minimum temperature (K)
- `T_max`: maximum temperature (K)
- `n_samples`: number of samples to collect per replica (after burnin and thinning)
- `burnin`: number of initial steps to discard
- `thin`: keep every thin-th sample
- `exchange_interval`: attempt replica exchange every N steps
- `collect_forces`: whether to collect forces in trajectory (only for MALA/HMC)

# Returns
- `replicas`: array of sample arrays, one per temperature
- `temperatures`: temperature values for each replica
- `acceptance_rates`: MCMC acceptance rate for each replica
- `exchange_rates`: exchange acceptance rate between neighboring replicas
- `trajectories`: trajectory data for each replica

# Example
```julia
using Distributed
addprocs(4)  # Add 4 worker processes

@everywhere using ACESIDopt
@everywhere using ACESIDopt.MSamplers

sampler = HMCSampler(step_size=0.05, n_leapfrog=10)
replicas, temps, acc_rates, exch_rates, trajs = run_parallel_tempering_distributed(
    sampler, system, model, 8, 300.0, 1000.0;
    n_samples=5000, burnin=2000, thin=5, exchange_interval=10
)
```
"""
function run_parallel_tempering_distributed(sampler, initial_system, model, n_replicas::Int, 
                                           T_min::Float64, T_max::Float64;
                                           n_samples::Int=1000, burnin::Int=1000, thin::Int=10,
                                           exchange_interval::Int=10, collect_forces::Bool=false)
    
    @assert n_replicas >= 2 "Need at least 2 replicas for parallel tempering"
    @assert T_max > T_min "T_max must be greater than T_min"
    @assert exchange_interval > 0 "exchange_interval must be positive"
    
    # Check if workers are available
    if nworkers() == 1
        @warn "No additional workers detected. Running on single process. Use addprocs() for parallelization."
    end
    
    kB = 8.617333262e-5  # eV/K
    
    # Create temperature schedule (geometric spacing)
    if n_replicas == 1
        temperatures = [T_min]
    else
        temperatures = [T_min * (T_max/T_min)^((i-1)/(n_replicas-1)) for i in 1:n_replicas]
    end
    
    println("="^70)
    println("Distributed Parallel Tempering Monte Carlo")
    println("="^70)
    println("Number of replicas: $n_replicas")
    println("Number of workers: $(nworkers())")
    println("Temperature range: $(round(T_min, digits=1)) K to $(round(T_max, digits=1)) K")
    println("Temperature schedule:")
    for (i, T) in enumerate(temperatures)
        println("  Replica $i: $(round(T, digits=2)) K")
    end
    println("Sampler type: $(typeof(sampler))")
    println("Exchange interval: every $exchange_interval steps")
    println("="^70)
    
    # Initialize replicas on all workers
    if length(initial_system) == 1
        systems = [deepcopy(initial_system) for _ in 1:n_replicas]
    elseif length(initial_system) == n_replicas
        systems = deepcopy(initial_system)
    else
        error("initial_system must be a single system or an array of systems with length equal to n_replicas")
    end
    
    # Initialize storage
    all_samples = [[] for _ in 1:n_replicas]
    if collect_forces
        all_trajs = [(energy=Float64[], forces=[], acc_rate=Float64[]) for _ in 1:n_replicas]
    else
        all_trajs = [(energy=Float64[], acc_rate=Float64[]) for _ in 1:n_replicas]
    end
    
    # Track acceptance statistics
    n_accepted_mcmc = zeros(Int, n_replicas)
    n_accepted_since_last_sample = zeros(Int, n_replicas)
    steps_since_last_sample = zeros(Int, n_replicas)
    n_attempted_exchange = zeros(Int, n_replicas-1)
    n_accepted_exchange = zeros(Int, n_replicas-1)
    
    n_total = burnin + n_samples * thin
    n_exchange_steps = div(n_total, exchange_interval)
    
    println("\nRunning distributed parallel tempering...")
    println("Total steps per replica: $n_total")
    println("Exchange attempts: $n_exchange_steps")
    println("Burnin: $burnin steps, then collecting $n_samples samples with thinning=$thin")
    
    # Main sampling loop with exchange synchronization
    for exchange_epoch in 1:n_exchange_steps
        step_start = (exchange_epoch - 1) * exchange_interval + 1
        step_end = min(exchange_epoch * exchange_interval, n_total)
        n_steps_this_epoch = step_end - step_start + 1
        
        # Evolve all replicas in parallel for exchange_interval steps
        results = @distributed (vcat) for i in 1:n_replicas
            # Each worker gets its replica and evolves it independently
            local_system = systems[i]
            local_samples = []
            local_energies = Float64[]
            local_acc_rates = Float64[]
            local_forces = collect_forces ? [] : nothing
            local_accepted = 0
            local_accepted_since_sample = 0
            local_steps_since_sample = 0
            
            T = temperatures[i]
            
            for local_step in 1:n_steps_this_epoch
                global_step = step_start + local_step - 1
                
                accepted, ΔU, U_current, f_current = step!(local_system, sampler, model, T)
                
                if accepted
                    local_accepted += 1
                    local_accepted_since_sample += 1
                end
                local_steps_since_sample += 1
                
                # Collect samples after burnin with thinning
                if global_step > burnin && (global_step - burnin) % thin == 0
                    push!(local_samples, deepcopy(local_system))
                    push!(local_energies, U_current)
                    
                    # Compute acceptance rate since last sample
                    local_acc = local_steps_since_sample > 0 ? local_accepted_since_sample / local_steps_since_sample : 0.0
                    push!(local_acc_rates, local_acc)
                    
                    # Reset counters
                    local_accepted_since_sample = 0
                    local_steps_since_sample = 0
                    
                    if collect_forces && f_current !== nothing
                        push!(local_forces, deepcopy(f_current))
                    end
                end
            end
            
            # Return results from this replica
            (replica_idx=i, system=local_system, samples=local_samples, 
             energies=local_energies, acc_rates=local_acc_rates, forces=local_forces, n_accepted=local_accepted)
        end
        
        # Update systems and collect samples from all replicas
        for result in results
            i = result.replica_idx
            systems[i] = result.system
            append!(all_samples[i], result.samples)
            append!(all_trajs[i].energy, result.energies)
            append!(all_trajs[i].acc_rate, result.acc_rates)
            if collect_forces && result.forces !== nothing
                append!(all_trajs[i].forces, result.forces)
            end
            n_accepted_mcmc[i] += result.n_accepted
        end
        
        # Attempt replica exchanges (sequential, but infrequent)
        offset = rand(0:1)
        for i in (1+offset):2:(n_replicas-1)
            if i >= n_replicas
                break
            end
            
            n_attempted_exchange[i] += 1
            
            # Get energies
            U_i = ustrip(u"eV", potential_energy(systems[i], model))
            U_j = ustrip(u"eV", potential_energy(systems[i+1], model))
            
            T_i = temperatures[i]
            T_j = temperatures[i+1]
            
            # Metropolis criterion for exchange
            β_i = 1.0 / (kB * T_i)
            β_j = 1.0 / (kB * T_j)
            
            Δ = (β_i - β_j) * (U_j - U_i)
            accept_prob = min(1.0, exp(Δ))
            
            if rand() < accept_prob
                # Accept exchange: swap the systems
                systems[i], systems[i+1] = systems[i+1], systems[i]
                n_accepted_exchange[i] += 1
            end
        end
        
        # Progress reporting
        if exchange_epoch % 10 == 0 || exchange_epoch == n_exchange_steps
            println("\nExchange epoch $exchange_epoch / $n_exchange_steps (step $step_end / $n_total):")
            for i in 1:n_replicas
                acc_rate = n_accepted_mcmc[i] / step_end
                println("  Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
                       "MCMC accept=$(round(acc_rate, digits=3))")
            end
            for i in 1:(n_replicas-1)
                if n_attempted_exchange[i] > 0
                    exch_rate = n_accepted_exchange[i] / n_attempted_exchange[i]
                    println("  Exchange $i↔$(i+1): accept=$(round(exch_rate, digits=3))")
                end
            end
        end
    end
    
    # Calculate final statistics
    mcmc_acceptance_rates = n_accepted_mcmc ./ n_total
    exchange_acceptance_rates = zeros(n_replicas-1)
    for i in 1:(n_replicas-1)
        if n_attempted_exchange[i] > 0
            exchange_acceptance_rates[i] = n_accepted_exchange[i] / n_attempted_exchange[i]
        end
    end
    
    println("\n" * "="^70)
    println("Distributed Parallel Tempering Complete!")
    println("="^70)
    println("\nMCMC Acceptance Rates:")
    for i in 1:n_replicas
        println("  Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
               "$(round(mcmc_acceptance_rates[i], digits=3))")
    end
    println("\nExchange Acceptance Rates:")
    for i in 1:(n_replicas-1)
        println("  Exchange $i↔$(i+1) ($(round(temperatures[i], digits=1)) K ↔ " *
               "$(round(temperatures[i+1], digits=1)) K): " *
               "$(round(exchange_acceptance_rates[i], digits=3))")
    end
    println("="^70)
    
    return all_samples, temperatures, mcmc_acceptance_rates, exchange_acceptance_rates, all_trajs
end

struct HALModel{T}
    model
    Σ::Matrix{T}
    Psqrt
    τ::T
end

function AtomsCalculators.potential_energy(atoms::AbstractSystem, bmodel::HALModel)
    uc =  .5 * bmodel.τ * sqrt(predictive_variance(bmodel.model, atoms, bmodel.Σ; Psqrt=bmodel.Psqrt))
    return potential_energy(atoms, bmodel.model) - uc * u"eV"
end


"""
    run_HAL_sampler(sampler, initial_system, model, T, Σ, Psqrt; τ=1.0, σ_stop=.1, n_samples::Int=1000, burnin::Int=1000, thin::Int=10, collect_forces::Bool=false)

Run MCMC sampling using the specified sampler configuration.

This is a unified interface that works with both RWMCSampler and MALASampler.
The dispatch to specific algorithms happens at the step! level.

# Parameters
- `sampler`: Either RWMCSampler or MALASampler instance (contains step_size)
- `initial_system`: starting configuration
- `model`: potential model (e.g., ACE potential or HarmonicCalculator)
- `T`: temperature (K)
- `Σ`: covariance matrix of the posterior distribution
- `Psqrt`: square root of the diagonal preconditioner matrix 
- `τ`: non-negative scaling parameter for the biasing strength of the HAL potential
- `σ_stop`: threshold for predictive standard deviation. Sampling terminates once this threshold has been reached.
- `n_samples`: number of samples to collect (after burnin and thinning)
- `burnin`: number of initial steps to discard
- `thin`: keep every thin-th sample
- `collect_forces`: whether to collect forces in trajectory 

# Returns
- `(samples, acceptance_rate, traj, system_max, std_max)`: collected samples, acceptance rate, trajectory data

# Examples
```julia
# RWMC sampling
rwmc = RWMCSampler(step_size=0.1)
samples, acc_rate, traj, system_max, std_max = run_HAL_sampler(rwmc, system, model, 300.0, Σ, Psqrt; τ=1.0, σ_stop=0.1, n_samples=5000, burnin=2000, thin=5, collect_forces=false)

# MALA sampling currently not supported with HAL potential
```
"""
function run_HAL_sampler(sampler, initial_system, model, T, Σ, Psqrt; τ=1.0, σ_stop=.1, n_samples::Int=1000, burnin::Int=1000, thin::Int=10, collect_forces::Bool=false)
    system_max = deepcopy(initial_system)
    std_max = sqrt(predictive_variance(model, system_max, Σ; Psqrt=Psqrt))
    
    samples = []
    if collect_forces
        traj = (energy=Float64[], forces=[], acc_rate=Float64[], std=Float64[])
    else
        traj = (energy=Float64[], acc_rate=Float64[], std=Float64[])
    end
    current_system = deepcopy(initial_system)
    
    n_total = burnin + n_samples * thin
    n_accepted = 0
    n_accepted_since_last_sample = 0
    steps_since_last_sample = 0
    
    sampler_type = typeof(sampler)
    println("Running $(sampler_type) sampling...")
    println("Parameters: T=$T K, step_size=$(sampler.step_size) Å")
    println("Burnin: $burnin steps")
    println("Collecting $n_samples samples with thinning=$thin")
    if collect_forces && sampler isa MALASampler
        println("Force collection: enabled")
    end
    @assert sampler isa RWMCSampler "HAL potential currently only supported with RWMCSampler"
    halmodel = HALModel(model, Σ, Psqrt, τ)
    @showprogress for step_num in 1:n_total
        accepted, ΔU, U_current, f_current = step!(current_system, sampler, halmodel, T)
        
        if accepted
            n_accepted += 1
            n_accepted_since_last_sample += 1
        end
        steps_since_last_sample += 1
        current_std = sqrt(predictive_variance(model, current_system, halmodel.Σ; Psqrt=halmodel.Psqrt))
        if current_std >= std_max
            system_max = deepcopy(current_system)
            std_max = current_std
            println("New maximum std reached: $(round(std_max, digits=4)) at step $step_num")
        end
        # Collect samples after burnin with thinning
        if step_num > burnin && (step_num - burnin) % thin == 0
            push!(samples, deepcopy(current_system))
            push!(traj.energy, U_current)
            push!(traj.std, current_std)
            
            # Compute acceptance rate since last sample
            local_acc_rate = steps_since_last_sample > 0 ? n_accepted_since_last_sample / steps_since_last_sample : 0.0
            push!(traj.acc_rate, local_acc_rate)
            
            # Reset counters
            n_accepted_since_last_sample = 0
            steps_since_last_sample = 0
            
            if collect_forces 
                if f_current !== nothing
                    push!(traj.forces, deepcopy(f_current))
                else
                    push!(traj.forces, forces(current_system, model))
                end
            end
        end
        
        if step_num % 1000 == 0
            acc_rate = n_accepted / step_num
            println("Step $step_num / $n_total, Acceptance rate: $(round(acc_rate, digits=3))")
        end
        if current_std > σ_stop
            println("Stopping criterion reached: predictive std $(round(current_std, digits=4)) > threshold $(round(σ_stop, digits=4)) at step $step_num")
            break
        end
    end
    
    acceptance_rate = n_accepted / n_total
    println("$(sampler_type) sampling complete!")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate, traj, system_max, std_max
end



end  # module MSamplers