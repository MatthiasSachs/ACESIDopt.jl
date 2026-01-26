module QueryModels

using Random
using Statistics
using ACEpotentials
using ACEpotentials: make_atoms_data
using ACEfit
using LinearAlgebra: Diagonal
using AtomsCalculators: potential_energy, forces
using Unitful
using ExtXYZ
using Plots
using ACESIDopt.MSamplers: RWMCSampler, run_parallel_tempering_distributed, run_HAL_sampler
using ACESIDopt: add_energy, add_energy_forces, row_mapping, pred_variance, expected_red_variance, predictive_variance, convert_forces_to_svector
using ACESIDopt: generate_ptd_diagnostics_and_log, plot_energy_comparison, plot_forces_comparison
using Distributed
import AtomsCalculators

export query_TSSID, query_ABSID, query_US, query_TrainData, query_HAL

# export potential_energy

using AtomsBase: cell_vectors, AbstractSystem
"""
    query_TSSID(raw_data_train, model, ts_model, ref_model, Σ, α, Psqrt, my_weights;
                plots_dir, pt_diagnostics_dir, t,
                N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT)

Perform Thompson Sampling with Surrogate Informed Design (TSSID) to select the next
candidate system for active learning.

# Arguments
- `raw_data_train`: Current training data
- `model`: Fitted ACE model
- `ref_model`: Reference model for ground truth evaluation
- `Σ`: Covariance matrix from Bayesian regression
- `α`: Noise precision estimate
- `Psqrt`: Square root of prior precision matrix
- `my_weights`: Weight function for data assembly

# Keyword Arguments
- `plots_dir`: Directory for plots
- `pt_diagnostics_dir`: Directory for PT diagnostics
- `t`: Current iteration number
- `N_REPLICAS`: Number of PT replicas
- `T_MIN`, `T_MAX`: Temperature range for PT
- `N_SAMPLES_PT`: Number of samples per replica
- `BURNIN_PT`: Burn-in steps
- `THIN_PT`: Thinning factor
- `EXCHANGE_INTERVAL`: Exchange interval for PT
- `STEP_SIZE_PT`: Step size for MCMC sampler
- `R_CUT`: Cutoff radius for diagnostics
- `other_data_dir`: Directory for other data including Thompson sampling model

# Returns
- `selected_system`: The selected system with reference energy and forces
"""
function query_TSSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                     plots_dir, other_data_dir, pt_diagnostics_dir, t,
                     N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                     EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT)
    
    # Generate plots for test and training data
    println("\nGenerating comparison plots...")
    
    #%% Run distributed parallel tempering for surrogate and candidate sampling in parallel
    println("\nRunning Distributed Parallel Tempering for Surrogate Samples...")
    
    # Create Thompson sampling model
    ts_model = deepcopy(model)
    ts_model.ps = ts_model.co_ps[1]

    # Save the TS model
    ts_model_filename = joinpath(other_data_dir, "ts_model_iter_$(lpad(t, 3, '0')).ace")
    ACEpotentials.save_model(ts_model, ts_model_filename)
    println("Saved TS model: $ts_model_filename")

    # Load model on all workers
    @everywhere ts_model_filename = $ts_model_filename
    @everywhere ts_model = ACEpotentials.load_model(ts_model_filename)[1]
    
    # Sample initial systems for surrogate PT
    n_available = length(raw_data_train)
    sampled_indices_sur = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_sur) < N_REPLICAS
        # If not enough unique samples, repeat some
        while length(sampled_indices_sur) < N_REPLICAS
            push!(sampled_indices_sur, rand(1:n_available))
        end
    end
    initial_systems_sur = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_sur]
    
    # Sample initial systems for candidate PT
    sampled_indices_cand = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_cand) < N_REPLICAS
        while length(sampled_indices_cand) < N_REPLICAS
            push!(sampled_indices_cand, rand(1:n_available))
        end
    end
    initial_systems_cand = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_cand]
    
    # Create samplers
    sampler_sur = RWMCSampler(step_size=STEP_SIZE_PT)
    sampler_cand = RWMCSampler(step_size=STEP_SIZE_PT)
    
    # Run surrogate and candidate PT in parallel using @sync
    sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = nothing, nothing, nothing, nothing, nothing
    cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = nothing, nothing, nothing, nothing, nothing
    
    @sync begin
        @async begin
            println("  Starting surrogate PT...")
            sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = 
                run_parallel_tempering_distributed(
                    sampler_sur, initial_systems_sur, ts_model, N_REPLICAS, T_MIN, T_MAX;
                    n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                )
            println("  Surrogate PT complete!")
        end
        
        @async begin
            println("  Starting candidate PT...")
            cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = 
                run_parallel_tempering_distributed(
                    sampler_cand, initial_systems_cand, ts_model, N_REPLICAS, T_MIN, T_MAX;
                    n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                )
            println("  Candidate PT complete!")
        end
    end
    
    # Generate diagnostics for surrogate PT
    println("\nGenerating surrogate PT diagnostics...")
    sur_diag_dir = joinpath(pt_diagnostics_dir, "surrogate")
    mkpath(sur_diag_dir)
    generate_ptd_diagnostics_and_log(sur_replicas, sur_temperatures, sur_mcmc_rates, 
                                     sur_exchange_rates, sur_trajs, sur_diag_dir, t, R_CUT * u"Å")
    
    # Generate diagnostics for candidate PT
    println("\nGenerating candidate PT diagnostics...")
    cand_diag_dir = joinpath(pt_diagnostics_dir, "candidate")
    mkpath(cand_diag_dir)
    generate_ptd_diagnostics_and_log(cand_replicas, cand_temperatures, cand_mcmc_rates, 
                                     cand_exchange_rates, cand_trajs, cand_diag_dir, t, R_CUT * u"Å")
    
    # Extract samples from target temperature (lowest temperature, replica 1)
    sur_samples = sur_replicas[1]
    cand_samples = cand_replicas[1]
    
    # Add energies to surrogate samples and energy+forces to candidate samples
    raw_data_tsur_nf = [at for at in add_energy(sur_samples, model)]
    raw_data_tcand = [at for at in add_energy_forces(cand_samples, model)]
    
    data_tsur_nf = make_atoms_data(raw_data_tsur_nf, model; 
                                energy_key = "energy", 
                                force_key = nothing, 
                                virial_key = nothing, 
                                weights = my_weights())
    n_data_tsur_nf = length(raw_data_tsur_nf)
    
    data_tcand = make_atoms_data(raw_data_tcand, model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = my_weights())
    n_data_tcand = length(raw_data_tcand)
    
    # Assemble least squares systems
    A_tsur_nf, Y_tsur_nf, W_tsur_nf = ACEfit.assemble(data_tsur_nf, model)
    Awp_tsur_nf = Diagonal(W_tsur_nf) * (A_tsur_nf / Psqrt)
    
    A_tcand, Y_tcand, W_tcand = ACEfit.assemble(data_tcand, model)
    Awp_tcand = Diagonal(W_tcand) * (A_tcand / Psqrt)
    
    # Calculate predictive variance and expected variance reduction
    p_var = [pred_variance(Σ, Awp_tcand[i,:], α) for i in 1:n_data_tcand] 
    p_var_mean = mean(p_var)
    
    exp_red = [(p_var[j] >= p_var_mean) ? expected_red_variance(Σ, Awp_tsur_nf, Awp_tcand[j,:], α) : -1 for j in 1:n_data_tcand] 
    idx = argmax(exp_red)

    println("\nSelected candidate $idx with max expected variance reduction: $(exp_red[idx])")

    # Use reference model for ground truth evaluation
    println("Computing energy and forces with reference potential...")
    
    # Compute energy and forces with reference model
    selected_system = raw_data_tcand[idx]
    ref_energy = potential_energy(selected_system, ref_model)
    ref_forces = forces(selected_system, ref_model)
    
    add_energy_forces(selected_system, ref_energy, ref_forces)
    
    return selected_system
end

"""
    query_ABSID(raw_data_train, model, ts_model, ref_model, Σ, α, Psqrt, my_weights;
                plots_dir, pt_diagnostics_dir, t,
                N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT)

Perform Thompson Sampling with Surrogate Informed Design (TSSID) to select the next
candidate system for active learning.

# Arguments
- `raw_data_train`: Current training data
- `model`: Fitted ACE model
- `ref_model`: Reference model for ground truth evaluation
- `Σ`: Covariance matrix from Bayesian regression
- `α`: Noise precision estimate
- `Psqrt`: Square root of prior precision matrix
- `my_weights`: Weight function for data assembly

# Keyword Arguments
- `plots_dir`: Directory for plots
- `pt_diagnostics_dir`: Directory for PT diagnostics
- `t`: Current iteration number
- `N_REPLICAS`: Number of PT replicas
- `T_MIN`, `T_MAX`: Temperature range for PT
- `N_SAMPLES_PT`: Number of samples per replica
- `BURNIN_PT`: Burn-in steps
- `THIN_PT`: Thinning factor
- `EXCHANGE_INTERVAL`: Exchange interval for PT
- `STEP_SIZE_PT`: Step size for MCMC sampler
- `R_CUT`: Cutoff radius for diagnostics
- `other_data_dir`: Directory for other data including Thompson sampling model

# Returns
- `selected_system`: The selected system with reference energy and forces
"""
function query_ABSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                     plots_dir, other_data_dir, pt_diagnostics_dir, t,
                     N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                     EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT)
    
    # Generate plots for test and training data
    println("\nGenerating comparison plots...")
    
    #%% Run distributed parallel tempering for surrogate and candidate sampling in parallel
    println("\nRunning Distributed Parallel Tempering for Surrogate Samples...")
    
    # Create Thompson sampling model
    ts_model = deepcopy(model)
    ts_model.ps = ts_model.co_ps[1]

    # Save the TS model
    ts_model_filename = joinpath(other_data_dir, "ts_model_iter_$(lpad(t, 3, '0')).ace")
    ACEpotentials.save_model(ts_model, ts_model_filename)
    println("Saved TS model: $ts_model_filename")

    # # Load model on all workers
    #@everywhere loc_model = $model

    # Sample initial systems for surrogate PT
    n_available = length(raw_data_train)
    sampled_indices_sur = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_sur) < N_REPLICAS
        # If not enough unique samples, repeat some
        while length(sampled_indices_sur) < N_REPLICAS
            push!(sampled_indices_sur, rand(1:n_available))
        end
    end
    initial_systems_sur = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_sur]
    
    # Sample initial systems for candidate PT
    sampled_indices_cand = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_cand) < N_REPLICAS
        while length(sampled_indices_cand) < N_REPLICAS
            push!(sampled_indices_cand, rand(1:n_available))
        end
    end
    initial_systems_cand = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_cand]
    
    # Create samplers
    sampler_sur = RWMCSampler(step_size=STEP_SIZE_PT)
    sampler_cand = RWMCSampler(step_size=STEP_SIZE_PT)
    

    # Run surrogate and candidate PT in parallel using @sync
    sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = nothing, nothing, nothing, nothing, nothing
    cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = nothing, nothing, nothing, nothing, nothing
    
    @sync begin
        @async begin
            println("  Starting surrogate PT...")
            sur_replicas, sur_temperatures, sur_mcmc_rates, sur_exchange_rates, sur_trajs = 
                run_parallel_tempering_distributed(
                    sampler_sur, initial_systems_sur, model, N_REPLICAS, T_MIN, T_MAX;
                    n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                )
            println("  Surrogate PT complete!")
        end
        
        @async begin
            println("  Starting candidate PT...")
            bmodel = biasedATModel(model, Σ, Psqrt, T_MIN)
            cand_replicas, cand_temperatures, cand_mcmc_rates, cand_exchange_rates, cand_trajs = 
                run_parallel_tempering_distributed(
                    sampler_cand, initial_systems_cand, bmodel, N_REPLICAS, T_MIN, T_MAX;
                    n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                    exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                )
            println("  Candidate PT complete!")
        end
    end
    
    # Generate diagnostics for surrogate PT
    println("\nGenerating surrogate PT diagnostics...")
    sur_diag_dir = joinpath(pt_diagnostics_dir, "surrogate")
    mkpath(sur_diag_dir)
    generate_ptd_diagnostics_and_log(sur_replicas, sur_temperatures, sur_mcmc_rates, 
                                     sur_exchange_rates, sur_trajs, sur_diag_dir, t, R_CUT * u"Å")
    
    # Generate diagnostics for candidate PT
    println("\nGenerating candidate PT diagnostics...")
    cand_diag_dir = joinpath(pt_diagnostics_dir, "candidate")
    mkpath(cand_diag_dir)
    generate_ptd_diagnostics_and_log(cand_replicas, cand_temperatures, cand_mcmc_rates, 
                                     cand_exchange_rates, cand_trajs, cand_diag_dir, t, R_CUT * u"Å")
    
    # Extract samples from target temperature (lowest temperature, replica 1)
    sur_samples = sur_replicas[1]
    cand_samples = cand_replicas[1]
    
    # Add energies to surrogate samples and energy+forces to candidate samples
    raw_data_tsur_nf = [at for at in add_energy(sur_samples, model)]
    raw_data_tcand = [at for at in add_energy_forces(cand_samples, model)]
    
    data_tsur_nf = make_atoms_data(raw_data_tsur_nf, model; 
                                energy_key = "energy", 
                                force_key = nothing, 
                                virial_key = nothing, 
                                weights = my_weights())
    n_data_tsur_nf = length(raw_data_tsur_nf)
    
    data_tcand = make_atoms_data(raw_data_tcand, model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = my_weights())
    n_data_tcand = length(raw_data_tcand)
    
    # Assemble least squares systems
    A_tsur_nf, Y_tsur_nf, W_tsur_nf = ACEfit.assemble(data_tsur_nf, model)
    Awp_tsur_nf = Diagonal(W_tsur_nf) * (A_tsur_nf / Psqrt)
    
    A_tcand, Y_tcand, W_tcand = ACEfit.assemble(data_tcand, model)
    Awp_tcand = Diagonal(W_tcand) * (A_tcand / Psqrt)
    
    # Calculate predictive variance and expected variance reduction
    p_var = [pred_variance(Σ, Awp_tcand[i,:], α) for i in 1:n_data_tcand] 
    p_var_mean = mean(p_var)
    
    exp_red = [(p_var[j] >= p_var_mean) ? expected_red_variance(Σ, Awp_tsur_nf, Awp_tcand[j,:], α) : -1 for j in 1:n_data_tcand] 
    idx = argmax(exp_red)

    println("\nSelected candidate $idx with max expected variance reduction: $(exp_red[idx])")

    # Use reference model for ground truth evaluation
    println("Computing energy and forces with reference potential...")
    
    # Compute energy and forces with reference model
    selected_system = raw_data_tcand[idx]
    ref_energy = potential_energy(selected_system, ref_model)
    ref_forces = forces(selected_system, ref_model)
    
    add_energy_forces(selected_system, ref_energy, ref_forces)
    
    return selected_system
end


struct biasedATModel{T}
    model
    Σ::Matrix{T}
    Psqrt
    Temp
end

function AtomsCalculators.potential_energy(atoms::AbstractSystem, bmodel::biasedATModel)
    kB = 8.617333262e-5  # eV/K
    uc =  predictive_variance(bmodel.model, atoms, bmodel.Σ; Psqrt=bmodel.Psqrt)
    return potential_energy(atoms, bmodel.model) - .5 * uc/(kB * bmodel.Temp) * u"eV"
end

"""
    query_US(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights)

Perform Uniform Sampling (US) to select the next candidate system for active learning.
Randomly samples a configuration from training data, randomizes positions uniformly
in the simulation box, and evaluates with the reference model.

# Arguments
- `raw_data_train`: Current training data
- `model`: Fitted ACE model (unused, kept for signature compatibility)
- `ref_model`: Reference model for ground truth evaluation
- `Σ`, `α`, `Psqrt`, `my_weights`: Unused, kept for signature compatibility

# Returns
- `selected_system`: The selected system with randomized positions and reference energy/forces
"""
function query_US(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                  plots_dir=nothing, other_data_dir=nothing, pt_diagnostics_dir=nothing, t=nothing,
                  N_REPLICAS=nothing, T_MIN=nothing, T_MAX=nothing, N_SAMPLES_PT=nothing, BURNIN_PT=nothing, THIN_PT=nothing,
                  EXCHANGE_INTERVAL=nothing, STEP_SIZE_PT=nothing, R_CUT=nothing)
    
    println("\nUniform Sampling: Generating random configuration...")
    
    # Sample a random configuration from training data
    n_available = length(raw_data_train)
    random_idx = rand(1:n_available)
    selected_system = deepcopy(raw_data_train[random_idx])
    
    println("Sampled configuration $random_idx from training data")
    
    # Randomize positions uniformly in the simulation box
    randomize_positions!(selected_system)
    
    println("Randomized positions uniformly in simulation box")
    
    # Compute energy and forces with reference model
    println("Computing energy and forces with reference potential...")
    ref_energy = potential_energy(selected_system, ref_model)
    ref_forces = forces(selected_system, ref_model)
    
    add_energy_forces(selected_system, ref_energy, ref_forces)
    
    println("Selected random configuration with energy: $ref_energy")
    
    return selected_system
end

"""
    query_TrainData(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                   train_data_name)

Perform Random Data Sampling to select the next candidate system for active learning.
Loads data from an external extended XYZ file and randomly samples a configuration.

# Arguments
- `raw_data_train`, `model`, `Σ`, `α`, `Psqrt`, `my_weights`: Unused, kept for signature compatibility
- `ref_model`: Reference model for ground truth evaluation
- `train_data_name`: Path to extended XYZ file containing candidate configurations

# Returns
- `selected_system`: A randomly selected system from the file with reference energy/forces
"""
function query_TrainData(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                        train_data_name,
                        plots_dir=nothing, other_data_dir=nothing, pt_diagnostics_dir=nothing, t=nothing,
                        N_REPLICAS=nothing, T_MIN=nothing, T_MAX=nothing, N_SAMPLES_PT=nothing, BURNIN_PT=nothing, THIN_PT=nothing,
                        EXCHANGE_INTERVAL=nothing, STEP_SIZE_PT=nothing, R_CUT=nothing)
    
    println("\nRandom Data Sampling: Loading data from file...")
    
    # Load data from extended XYZ file
    println("Loading configurations from: $train_data_name")
    candidate_data = ExtXYZ.load(train_data_name)
    n_candidates = length(candidate_data)
    println("Loaded $n_candidates configurations")
    
    # Randomly sample one configuration
    random_idx = rand(1:n_candidates)
    selected_system = candidate_data[random_idx]
    
    println("Randomly selected configuration $random_idx from file")
    

    
    return convert_forces_to_svector(selected_system)
end

"""
    query_HAL(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
              TAU, SIGMA_STOP,
              plots_dir, t, T_MIN, N_SAMPLES_PT, BURNIN_PT, THIN_PT, STEP_SIZE_PT)

Perform Hamiltonian Annealed Learning (HAL) sampling to select the next candidate system.
Runs HAL sampler and generates diagnostic plots of the sampling trajectory.

# Arguments
- `raw_data_train`: Current training data
- `model`: Fitted ACE model
- `ref_model`: Reference model for ground truth evaluation
- `Σ`: Covariance matrix from Bayesian regression
- `α`, `Psqrt`, `my_weights`: Unused, kept for signature compatibility
- `TAU`: Temperature annealing parameter for HAL
- `SIGMA_STOP`: Stopping criterion based on predicted standard deviation
- `plots_dir`: Directory for plots
- `t`: Current iteration number
- `T_MIN`: Minimum temperature for sampling
- `N_SAMPLES_PT`: Number of samples
- `BURNIN_PT`: Burn-in steps
- `THIN_PT`: Thinning factor
- `STEP_SIZE_PT`: Step size for MCMC sampler

# Returns
- `system_max`: The selected system with maximum acquisition value and reference energy/forces
"""
function query_HAL(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                   TAU, SIGMA_STOP,
                   plots_dir, t, T_MIN, N_SAMPLES_PT, BURNIN_PT, THIN_PT, STEP_SIZE_PT,
                   other_data_dir=nothing, pt_diagnostics_dir=nothing,
                   N_REPLICAS=nothing, T_MAX=nothing, EXCHANGE_INTERVAL=nothing, R_CUT=nothing)
    
    println("\nHAL Sampling: Running Hamiltonian Annealed Learning...")
    println("  TAU: $TAU")
    println("  SIGMA_STOP: $SIGMA_STOP")
    
    # Sample initial system from training data
    n_available = length(raw_data_train)
    random_idx = rand(1:n_available)
    initial_system = deepcopy(raw_data_train[random_idx])
    println("  Initial configuration: $random_idx from training data")
    
    # Create sampler
    sampler = RWMCSampler(step_size=STEP_SIZE_PT)
    
    # Run HAL sampler
    println("\nRunning HAL sampler...")
    samples, acceptance_rate, traj, system_max, std_max = run_HAL_sampler(
        sampler, initial_system, model, T_MIN, Σ, Psqrt;
        τ=TAU, σ_stop=SIGMA_STOP,
        n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
        collect_forces=false
    )
    println("HAL sampling complete!")
    println("Maximum uncertainty reached: $(round(std_max, digits=4))")
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    # Extract trajectory information for plotting
    iterations = 1:length(traj.energy)
    potential_energies = traj.energy
    predicted_stds = traj.std
    # HAL energy = potential energy (for plotting purposes, they're from the HAL model)
    hal_energies = ustrip(potential_energies) .+ (0.5 .* TAU .* predicted_stds)
    
    # Create plots directory for HAL diagnostics
    hal_plots_dir = joinpath(plots_dir, "hal_diagnostics")
    mkpath(hal_plots_dir)
    
    # Plot 1: Iteration vs Potential Energy
    p1 = plot(iterations, potential_energies,
        xlabel="Iteration",
        ylabel="Potential Energy (eV)",
        title="HAL Sampling: Potential Energy Trajectory (Iter $t)",
        label="Potential Energy",
        marker=:circle,
        markersize=2,
        linewidth=1.5,
        legend=:best)
    savefig(p1, joinpath(hal_plots_dir, "potential_energy_iter_$(lpad(t, 3, '0')).png"))
    println("Saved potential energy plot")
    
    # Plot 2: Iteration vs Predicted Standard Deviation
    p2 = plot(iterations, predicted_stds,
        xlabel="Iteration",
        ylabel="Predicted Std Dev",
        title="HAL Sampling: Uncertainty Trajectory (Iter $t)",
        label="Predicted Std",
        marker=:circle,
        markersize=2,
        linewidth=1.5,
        legend=:best,
        color=:red)
    hline!([SIGMA_STOP], label="SIGMA_STOP", linestyle=:dash, color=:black, linewidth=2)
    savefig(p2, joinpath(hal_plots_dir, "predicted_std_iter_$(lpad(t, 3, '0')).png"))
    println("Saved predicted std plot")
    
    # Plot 3: Iteration vs HAL Energy
    p3 = plot(iterations, hal_energies,
        xlabel="Iteration",
        ylabel="HAL Energy",
        title="HAL Sampling: HAL Energy Trajectory (Iter $t)",
        label="HAL Energy",
        marker=:circle,
        markersize=2,
        linewidth=1.5,
        legend=:best,
        color=:green)
    savefig(p3, joinpath(hal_plots_dir, "hal_energy_iter_$(lpad(t, 3, '0')).png"))
    println("Saved HAL energy plot")
    
    # Combined plot with all three metrics
    p_combined = plot(
        plot(iterations, potential_energies, ylabel="Pot. Energy (eV)", label="", color=:blue, lw=1.5),
        plot(iterations, predicted_stds, ylabel="Pred. Std", label="", color=:red, lw=1.5),
        plot(iterations, hal_energies, ylabel="HAL Energy", xlabel="Iteration", label="", color=:green, lw=1.5),
        layout=(3,1),
        size=(800, 800),
        plot_title="HAL Sampling Diagnostics (Iter $t)"
    )
    savefig(p_combined, joinpath(hal_plots_dir, "combined_diagnostics_iter_$(lpad(t, 3, '0')).png"))
    println("Saved combined diagnostics plot")
    
    # Compute energy and forces with reference model
    println("\nComputing energy and forces with reference potential...")
    ref_energy = potential_energy(system_max, ref_model)
    ref_forces = forces(system_max, ref_model)
    
    add_energy_forces(system_max, ref_energy, ref_forces)
    
    println("Selected HAL configuration with energy: $ref_energy")
    
    return system_max
end

"""
    randomize_positions!(atoms)

Randomize atomic positions uniformly within the simulation box (in-place modification).
Positions are sampled as linear combinations of cell vectors with random coefficients in [0,1].

# Arguments
- `atoms`: An AtomsBase-compatible atomic system with periodic boundary conditions
"""
function randomize_positions!(atoms)
    
    cell = cell_vectors(atoms)
    n_atoms = length(atoms)
    
    # Generate new positions uniformly in the cell
    for i in 1:n_atoms
        # Position = sum of cell vectors with random coefficients [0,1]
        atoms.atom_data.position[i] = sum(cell[j] * rand() for j in 1:3)
    end
end

end # module
