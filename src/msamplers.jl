module MSamplers

using AtomsBase
using ExtXYZ: Atoms
using AtomsCalculators: potential_energy, forces
using Unitful: ustrip, @u_str

export rwmc_step!, run_rwmc_sampling, mala_step!, run_mala_sampling, set_position!, set_velocity!

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
    current_positions = position(system, :)
    U_current = ustrip(u"eV", potential_energy(system, model))
    
    # Get forces (negative gradient of potential)
    f_current = forces(system, model)
    
    # Propose new positions using Langevin dynamics
    # x' = x - (step_size^2/(2*kB*T)) * ∇U(x) + step_size * ξ
    # Note: forces = -∇U, so we add forces
    proposal_positions = similar(current_positions)
    for i in 1:natoms
        f_i = ustrip.(u"eV/Å", f_current[i])
        drift = (step_size^2 / (2 * kB * T)) * f_i * u"Å"
        noise = step_size * randn(3) * u"Å"
        proposal_positions[i] = current_positions[i] .+ drift .+ noise
    end
    
    # Update system with proposed positions
    for i in 1:natoms
        set_position!(system, i, proposal_positions[i])
    end
    
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
        for i in 1:natoms
            set_position!(system, i, current_positions[i])
        end
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

end  # module MSamplers