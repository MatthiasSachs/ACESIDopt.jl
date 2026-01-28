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
using AtomsBase
import AtomsCalculators

export query_TSSID, query_ABSID, query_US, query_TrainData, query_HAL

# export potential_energy

using AtomsBase: cell_vectors, AbstractSystem
"""
    query_TSSID(raw_data_train, model, ts_model, ref_model, Σ, α, Psqrt, my_weights;
                plots_dir, pt_diagnostics_dir, t,
                N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT, constraint)

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
- `constraint`: Optional constraint function that takes a system and returns true if valid (default: nothing)

# Returns
- `selected_system`: The selected system with reference energy and forces
"""
function query_TSSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                     plots_dir, other_data_dir, pt_diagnostics_dir, t,
                     N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                     EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT, constraint=nothing)
    
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
    
    # Create samplers
    sampler_biased = RWMCSampler(step_size=STEP_SIZE_PT)
    sampler_gibbs = RWMCSampler(step_size=STEP_SIZE_PT)
    
    # Keep sampling until we get valid candidates with non-empty exp_red
    exp_red = Float64[]
    attempt = 0
    
    while isempty(exp_red)
        attempt += 1
        if attempt > 1
            println("\nAttempt $attempt: Previous sampling yielded no valid candidates, resampling...")
        end
        
        # Sample initial systems for biased PT
        sampled_indices_biased = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
        if length(sampled_indices_biased) < N_REPLICAS
            # If not enough unique samples, repeat some
            while length(sampled_indices_biased) < N_REPLICAS
                push!(sampled_indices_biased, rand(1:n_available))
            end
        end
        initial_systems_biased = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_biased]
        
        # Sample initial systems for candidate PT
        sampled_indices_gibbs = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
        if length(sampled_indices_gibbs) < N_REPLICAS
            while length(sampled_indices_gibbs) < N_REPLICAS
                push!(sampled_indices_gibbs, rand(1:n_available))
            end
        end
        initial_systems_gibbs = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_gibbs]
        
        # Run surrogate and candidate PT in parallel using @sync
        biased_replicas, biased_temperatures, biased_mcmc_rates, biased_exchange_rates, biased_trajs = nothing, nothing, nothing, nothing, nothing
        gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, gibbs_exchange_rates, gibbs_trajs = nothing, nothing, nothing, nothing, nothing
        
        @sync begin
            @async begin
                println("  Starting biased PT...")
                biased_replicas, biased_temperatures, biased_mcmc_rates, biased_exchange_rates, biased_trajs = 
                    run_parallel_tempering_distributed(
                        sampler_biased, initial_systems_biased, ts_model, N_REPLICAS, T_MIN, T_MAX;
                        n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                        exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                    )
                println("  Biased PT complete!")
            end
            
            @async begin
                println("  Starting gibbs PT...")
                gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, gibbs_exchange_rates, gibbs_trajs = 
                    run_parallel_tempering_distributed(
                        sampler_gibbs, initial_systems_gibbs, ts_model, N_REPLICAS, T_MIN, T_MAX;
                        n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                        exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                    )
                println("  Gibbs PT complete!")
            end
        end
        
        # Generate diagnostics for biased PT
        println("\nGenerating biased PT diagnostics...")
        biased_diag_dir = joinpath(pt_diagnostics_dir, "biased")
        mkpath(biased_diag_dir)
        generate_ptd_diagnostics_and_log(biased_replicas, biased_temperatures, biased_mcmc_rates, 
                                         biased_exchange_rates, biased_trajs, biased_diag_dir, t, R_CUT * u"Å")
        
        # Generate diagnostics for candidate PT
        println("\nGenerating gibbs PT diagnostics...")
        gibbs_diag_dir = joinpath(pt_diagnostics_dir, "gibbs")
        mkpath(gibbs_diag_dir)
        generate_ptd_diagnostics_and_log(gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, 
                                         gibbs_exchange_rates, gibbs_trajs, gibbs_diag_dir, t, R_CUT * u"Å")
        
        # Extract samples from target temperature (lowest temperature, replica 1)
        biased_samples = biased_replicas[1]
        gibbs_samples = gibbs_replicas[1]
        
        # Add energies to surrogate samples and energy+forces to candidate samples
        if constraint === nothing
            raw_data_tbiased = [at for at in add_energy(biased_samples, model)]
            raw_data_tgibbs = [at for at in add_energy_forces(gibbs_samples, model)]
        else
            raw_data_tbiased = [at for at in add_energy(biased_samples, model) if constraint(at)]
            raw_data_tgibbs = [at for at in add_energy_forces(gibbs_samples, model) if constraint(at)]
        end
        
        println("After constraint filtering: $(length(raw_data_tbiased)) biased samples, $(length(raw_data_tgibbs)) gibbs samples")
        
        if isempty(raw_data_tbiased) || isempty(raw_data_tgibbs)
            println("Warning: No valid samples after constraint filtering, resampling...")
            exp_red = Float64[]
            continue
        end
        
        data_tbiased = make_atoms_data(raw_data_tbiased, model; 
                                    energy_key = "energy", 
                                    force_key = nothing, 
                                    virial_key = nothing, 
                                    weights = my_weights())
        n_data_tbiased = length(raw_data_tbiased)
        
        data_tgibbs = make_atoms_data(raw_data_tgibbs, model; 
                                    energy_key = "energy", 
                                    force_key = "forces", 
                                    virial_key = nothing, 
                                    weights = my_weights())
        n_data_tgibbs = length(raw_data_tgibbs)
        
        # Assemble least squares systems
        A_tbiased, Y_tbiased, W_tbiased = ACEfit.assemble(data_tbiased, model)
        Awp_tbiased = Diagonal(W_tbiased) * (A_tbiased / Psqrt)
        
        A_tgibbs, Y_tgibbs, W_tgibbs = ACEfit.assemble(data_tgibbs, model)
        Awp_tgibbs = Diagonal(W_tgibbs) * (A_tgibbs / Psqrt)
        
        # Calculate predictive variance and expected variance reduction
        p_var = [pred_variance(Σ, Awp_tgibbs[i,:], α) for i in 1:n_data_tgibbs] 
        p_var_mean = mean(p_var)
        
        exp_red = [(p_var[j] >= p_var_mean) ? expected_red_variance(Σ, Awp_tbiased, Awp_tgibbs[j,:], α) : -1 for j in 1:n_data_tgibbs]
        
        if !isempty(exp_red)
            idx = argmax(exp_red)
            println("\nSelected candidate $idx with max expected variance reduction: $(exp_red[idx])")

            # Use reference model for ground truth evaluation
            println("Computing energy and forces with reference potential...")
            
            # Compute energy and forces with reference model
            selected_system = raw_data_tgibbs[idx]
            ref_energy = potential_energy(selected_system, ref_model)
            ref_forces = forces(selected_system, ref_model)
            
            selected_system = add_energy_forces(selected_system, ref_energy, ref_forces)
            
            return selected_system
        else
            println("Warning: exp_red is empty, no valid candidates found, resampling...")
        end
    end
end

"""
    query_ABSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                plots_dir, other_data_dir, pt_diagnostics_dir, t,
                N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT, constraint)

Perform Acquisition-Based Surrogate Informed Design (ABSID) to select the next
candidate system for active learning. Uses biased sampling with predictive variance
to explore high-uncertainty regions, combined with Gibbs sampling for candidate generation.

The method runs two parallel tempering chains:
1. Biased sampling: Samples from a modified potential that favors high-uncertainty regions
2. Gibbs sampling: Samples from the learned ACE model potential

Candidates are selected by maximizing expected variance reduction over the biased samples.

# Arguments
- `raw_data_train`: Current training data
- `model`: Fitted ACE model with committee for uncertainty quantification
- `ref_model`: Reference model for ground truth evaluation
- `Σ`: Covariance matrix from Bayesian regression
- `α`: Noise precision estimate
- `Psqrt`: Square root of prior precision matrix
- `my_weights`: Weight function for data assembly

# Keyword Arguments
- `plots_dir`: Directory for plots
- `other_data_dir`: Directory for other data including Thompson sampling model
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
- `constraint`: Constraint function that takes a system and returns true if valid

# Returns
- `selected_system`: The selected system with reference energy and forces
"""
function query_ABSID(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                     plots_dir, other_data_dir, pt_diagnostics_dir, t,
                     N_REPLICAS, T_MIN, T_MAX, N_SAMPLES_PT, BURNIN_PT, THIN_PT,
                     EXCHANGE_INTERVAL, STEP_SIZE_PT, R_CUT, constraint)
    
    # Generate plots for test and training data
    println("\nGenerating comparison plots...")
    
    #%% Run distributed parallel tempering for biased and gibbs sampling in parallel
    println("\nRunning Distributed Parallel Tempering for biased Samples...")
    
    # Create Thompson sampling model
    ts_model = deepcopy(model)
    ts_model.ps = ts_model.co_ps[1]

    # Save the TS model
    ts_model_filename = joinpath(other_data_dir, "ts_model_iter_$(lpad(t, 3, '0')).ace")
    ACEpotentials.save_model(ts_model, ts_model_filename)
    println("Saved TS model: $ts_model_filename")

    # # Load model on all workers
    #@everywhere loc_model = $model

    # Sample initial systems for biasedrogate PT
    n_available = length(raw_data_train)
    sampled_indices_biased = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_biased) < N_REPLICAS
        # If not enough unique samples, repeat some
        while length(sampled_indices_biased) < N_REPLICAS
            push!(sampled_indices_biased, rand(1:n_available))
        end
    end
    initial_systems_biased = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_biased]
    
    # Sample initial systems for gibbs PT
    sampled_indices_gibbs = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
    if length(sampled_indices_gibbs) < N_REPLICAS
        while length(sampled_indices_gibbs) < N_REPLICAS
            push!(sampled_indices_gibbs, rand(1:n_available))
        end
    end
    initial_systems_gibbs = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_gibbs]
    
    # Create samplers
    sampler_biased = RWMCSampler(step_size=STEP_SIZE_PT)
    sampler_gibbs = RWMCSampler(step_size=STEP_SIZE_PT)
    
    # Keep sampling until we get valid candidates with non-empty exp_red
    exp_red = Float64[]
    attempt = 0
    
    while isempty(exp_red)
        attempt += 1
        if attempt > 1
            println("\nAttempt $attempt: Previous sampling yielded no valid candidates, resampling...")
        end
        
        # Sample initial systems for biased PT
        sampled_indices_biased = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
        if length(sampled_indices_biased) < N_REPLICAS
            # If not enough unique samples, repeat some
            while length(sampled_indices_biased) < N_REPLICAS
                push!(sampled_indices_biased, rand(1:n_available))
            end
        end
        initial_systems_biased = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_biased]
        
        # Sample initial systems for gibbs PT
        sampled_indices_gibbs = Random.randperm(n_available)[1:min(N_REPLICAS, n_available)]
        if length(sampled_indices_gibbs) < N_REPLICAS
            while length(sampled_indices_gibbs) < N_REPLICAS
                push!(sampled_indices_gibbs, rand(1:n_available))
            end
        end
        initial_systems_gibbs = [deepcopy(raw_data_train[idx]) for idx in sampled_indices_gibbs]

        # Run biased and gibbs PT in parallel using @sync
        biased_replicas, biased_temperatures, biased_mcmc_rates, biased_exchange_rates, biased_trajs = nothing, nothing, nothing, nothing, nothing
        gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, gibbs_exchange_rates, gibbs_trajs = nothing, nothing, nothing, nothing, nothing
        
        @sync begin
            @async begin
                println("  Starting biasedrogate PT...")
                bmodel = biasedATModel(model, Σ, Psqrt, T_MIN)
                biased_replicas, biased_temperatures, biased_mcmc_rates, biased_exchange_rates, biased_trajs = 
                    run_parallel_tempering_distributed(
                        sampler_biased, initial_systems_biased, bmodel, N_REPLICAS, T_MIN, T_MAX;
                        n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                        exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                    )
                println("  biased PT complete!")
            end
            
            @async begin
                println("  Starting gibbs PT...")
                # bmodel = biasedATModel(model, Σ, Psqrt, T_MIN)
                gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, gibbs_exchange_rates, gibbs_trajs = 
                    run_parallel_tempering_distributed(
                        sampler_gibbs, initial_systems_gibbs, model, N_REPLICAS, T_MIN, T_MAX;
                        n_samples=N_SAMPLES_PT, burnin=BURNIN_PT, thin=THIN_PT,
                        exchange_interval=EXCHANGE_INTERVAL, collect_forces=false
                    )
                println("  gibbs PT complete!")
            end
        end
        
        # Generate diagnostics for biased PT
        println("\nGenerating biased PT diagnostics...")
        biased_diag_dir = joinpath(pt_diagnostics_dir, "biased")
        mkpath(biased_diag_dir)
        generate_ptd_diagnostics_and_log(biased_replicas, biased_temperatures, biased_mcmc_rates, 
                                         biased_exchange_rates, biased_trajs, biased_diag_dir, t, R_CUT * u"Å")
        
        # Generate diagnostics for gibbs PT
        println("\nGenerating gibbs PT diagnostics...")
        gibbs_diag_dir = joinpath(pt_diagnostics_dir, "gibbs")
        mkpath(gibbs_diag_dir)
        generate_ptd_diagnostics_and_log(gibbs_replicas, gibbs_temperatures, gibbs_mcmc_rates, 
                                         gibbs_exchange_rates, gibbs_trajs, gibbs_diag_dir, t, R_CUT * u"Å")
        
        # Extract samples from target temperature (lowest temperature, replica 1)
        biased_samples = biased_replicas[1]
        gibbs_samples = gibbs_replicas[1]
        
        # Add energies to biased samples and energy+forces to gibbs samples
        raw_data_tbiased = [at for at in add_energy(biased_samples, model) if constraint(at)]
        raw_data_tgibbs = [at for at in add_energy_forces(gibbs_samples,model) if constraint(at)]
        
        println("After constraint filtering: $(length(raw_data_tbiased)) biased samples, $(length(raw_data_tgibbs)) gibbs samples")
        
        if isempty(raw_data_tbiased) || isempty(raw_data_tgibbs)
            println("Warning: No valid samples after constraint filtering, resampling...")
            exp_red = Float64[]
            continue
        end
        
        data_tbiased = make_atoms_data(raw_data_tbiased, model; 
                                    energy_key = "energy", 
                                    force_key = "forces", 
                                    virial_key = nothing, 
                                    weights = my_weights())
        n_data_tbiased = length(raw_data_tbiased)
        
        data_tgibbs = make_atoms_data(raw_data_tgibbs, model; 
                                    energy_key = "energy", 
                                    force_key = nothing, 
                                    virial_key = nothing, 
                                    weights = my_weights())
        n_data_tgibbs = length(raw_data_tgibbs)
        
        # Assemble least squares systems
        A_tbiased, Y_tbiased, W_tbiased = ACEfit.assemble(data_tbiased, model)
        Awp_tbiased = Diagonal(W_tbiased) * (A_tbiased / Psqrt)
        
        A_tgibbs, Y_tgibbs, W_tgibbs = ACEfit.assemble(data_tgibbs, model)
        Awp_tgibbs = Diagonal(W_tgibbs) * (A_tgibbs / Psqrt)
        
        # Calculate predictive variance and expected variance reduction
        p_var = [pred_variance(Σ, Awp_tbiased[i,:], α) for i in 1:n_data_tbiased] 
        p_var_mean = mean(p_var)
        
        exp_red = [(p_var[j] >= p_var_mean) ? expected_red_variance(Σ, Awp_tbiased, Awp_tgibbs[j,:], α) : -1 for j in 1:n_data_tgibbs]
        
        if !isempty(exp_red)
            idx = argmax(exp_red)
            println("\nSelected gibbs $idx with max expected variance reduction: $(exp_red[idx])")
            
            # Use reference model for ground truth evaluation
            println("Computing energy and forces with reference potential...")
            
            # Compute energy and forces with reference model
            selected_system = raw_data_tbiased[idx]
            ref_energy = potential_energy(selected_system, ref_model)
            ref_forces = forces(selected_system, ref_model)
            
            selected_system = add_energy_forces(selected_system, ref_energy, ref_forces)
            
            return selected_system
        else
            println("Warning: exp_red is empty, no valid candidates found, resampling...")
        end
    end
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

# Keyword Arguments
- `constraint`: Optional constraint function that takes a system and returns true if valid (default: nothing)

# Returns
- `selected_system`: The selected system with randomized positions and reference energy/forces
"""
function query_US(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                  plots_dir=nothing, other_data_dir=nothing, pt_diagnostics_dir=nothing, t=nothing,
                  N_REPLICAS=nothing, T_MIN=nothing, T_MAX=nothing, N_SAMPLES_PT=nothing, BURNIN_PT=nothing, THIN_PT=nothing,
                  EXCHANGE_INTERVAL=nothing, STEP_SIZE_PT=nothing, R_CUT=nothing, constraint=nothing)
    println("\nUniform Sampling: Generating random configuration...")
    
    # Sample a random configuration from training data
    selected_system = deepcopy(raw_data_train[1])
    
    attempt = 0
    while true
        attempt += 1
        if attempt > 1
            println("Attempt $attempt: Previous configuration did not satisfy constraint, resampling...")
        end
        
        # Randomize positions uniformly in the simulation box
        randomize_positions!(selected_system)
        
        # Compute energy with reference model
        ref_energy = potential_energy(selected_system, ref_model)
        
        # Compute forces with reference model
        println("Computing forces with reference potential...")
        ref_forces = forces(selected_system, ref_model)
        
        selected_system = add_energy_forces(selected_system, ref_energy, ref_forces)
        
        # Check constraint
        if constraint === nothing || constraint(selected_system)
            println("Selected random configuration with energy: $ref_energy")
            return selected_system
        else
            println("Configuration rejected by constraint (energy: $ref_energy), resampling...")
        end
    end
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

# Keyword Arguments
- `constraint`: Optional constraint function that takes a system and returns true if valid (default: nothing)

# Returns
- `selected_system`: A randomly selected system from the file with reference energy/forces
"""
function query_TrainData(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                        train_data_name,
                        plots_dir=nothing, other_data_dir=nothing, pt_diagnostics_dir=nothing, t=nothing,
                        N_REPLICAS=nothing, T_MIN=nothing, T_MAX=nothing, N_SAMPLES_PT=nothing, BURNIN_PT=nothing, THIN_PT=nothing,
                        EXCHANGE_INTERVAL=nothing, STEP_SIZE_PT=nothing, R_CUT=nothing, constraint=nothing)
    
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
    

    selected_system = add_energy_forces(selected_system, potential_energy(selected_system, ref_model), forces(selected_system, ref_model))
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

# Keyword Arguments
- `constraint`: Optional constraint function that takes a system and returns true if valid (default: nothing)

# Returns
- `system_max`: The selected system with maximum acquisition value and reference energy/forces
"""
function query_HAL(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights;
                   TAU, SIGMA_STOP,
                   plots_dir, t, T_MIN, N_SAMPLES_PT, BURNIN_PT, THIN_PT, STEP_SIZE_PT,
                   other_data_dir=nothing, pt_diagnostics_dir=nothing,
                   N_REPLICAS=nothing, T_MAX=nothing, EXCHANGE_INTERVAL=nothing, R_CUT=nothing, constraint=nothing)
    
    println("\nHAL Sampling: Running Hamiltonian Annealed Learning...")
    println("  TAU: $TAU")
    println("  SIGMA_STOP: $SIGMA_STOP")
    
    # Create sampler
    sampler = RWMCSampler(step_size=STEP_SIZE_PT)
    
    attempt = 0
    while true
        attempt += 1
        if attempt > 1
            println("\nAttempt $attempt: Previous HAL sample did not satisfy constraint, resampling...")
        end
        
        # Sample initial system from training data
        n_available = length(raw_data_train)
        random_idx = rand(1:n_available)
        initial_system = deepcopy(raw_data_train[random_idx])
        println("  Initial configuration: $random_idx from training data")
        
        # Run HAL sampler
        println("\nRunning HAL sampler...")
        samples, acceptance_rate, traj, system_max, std_max = run_HAL_sampler(
            sampler, initial_system, model, T_MIN, Σ, Psqrt;
            τ=TAU, σ_stop=SIGMA_STOP,
            n_samples=N_SAMPLES_PT, burnin=0, thin=THIN_PT,
            collect_forces=false
        )
        println("HAL sampling complete!")
        println("Maximum uncertainty reached: $(round(std_max, digits=4))")
        println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
        
        # Compute energy and forces with reference model
        println("\nComputing energy and forces with reference potential...")
        ref_energy = potential_energy(system_max, ref_model)
        ref_forces = forces(system_max, ref_model)
        
        system_max = add_energy_forces(system_max, ref_energy, ref_forces)
        
        # Check constraint
        if constraint === nothing || constraint(system_max)
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
            
            println("Selected HAL configuration with energy: $ref_energy")
            
            return system_max
        else
            println("Configuration rejected by constraint (energy: $ref_energy), resampling...")
        end
    end
end

"""
    randomize_positions!(atoms)

Randomize atomic positions uniformly within the simulation box (in-place modification).
Positions are sampled as linear combinations of cell vectors with random coefficients in [0,1].

# Arguments
- `atoms`: An AtomsBase-compatible atomic system with periodic boundary conditions
"""
function randomize_positions!(atoms)
    
    cell = AtomsBase.cell_vectors(atoms)
    n_atoms = length(atoms)
    
    # Generate new positions uniformly in the cell
    for i in 1:n_atoms
        # Position = sum of cell vectors with random coefficients [0,1]
        atoms.atom_data.position[i] = sum(cell[j] * rand() for j in 1:3)
    end
end

end # module
