using Plots
using AtomsCalculators: potential_energy, forces
using Unitful: ustrip


using Statistics
using AtomsBase: FlexibleSystem
using ACEpotentials

function plot_forces_comparison(raw_data, model, filename; marked=Int[])
    """
    Plot true vs predicted forces for a dataset and save to file.
    
    Arguments:
    - raw_data: Array of atomic structures with true forces
    - model: ACE model for predictions
    - filename: Path where the plot will be saved
    - marked: Indices of structures to highlight in red
    """
    
    # Extract true forces (flatten all force components) and track structure indices
    true_forces_all = Float64[]
    pred_forces_all = Float64[]
    structure_indices = Int[]  # Track which structure each force component belongs to
    
    for (struct_idx, d) in enumerate(raw_data)
        true_f = d.atom_data.forces
        pred_f = forces(d, model)
        
        for i in 1:length(true_f)
            if isa(true_f[i], Vector)
                append!(true_forces_all, true_f[i])
                append!(structure_indices, fill(struct_idx, length(true_f[i])))
            else
                append!(true_forces_all, [true_f[i][1], true_f[i][2], true_f[i][3]])
                append!(structure_indices, fill(struct_idx, 3))
            end
            append!(pred_forces_all, ustrip.(u"eV/Å", pred_f[i]))
        end
    end
    
    # Separate marked and unmarked points
    unmarked_mask = [!(idx in marked) for idx in structure_indices]
    marked_mask = [idx in marked for idx in structure_indices]
    
    # Create scatter plot - unmarked points first
    p_forces = scatter(true_forces_all[unmarked_mask], pred_forces_all[unmarked_mask],
           xlabel="True Force Component (eV/Å)",
           ylabel="Predicted Force Component (eV/Å)",
           title="True vs Predicted Forces",
           label="Unmarked",
           alpha=0.4,
           markersize=2,
           color=:blue)
    
    # Add marked points in red
    if any(marked_mask)
        scatter!(p_forces, true_forces_all[marked_mask], pred_forces_all[marked_mask],
               label="Marked",
               alpha=0.6,
               markersize=3,
               color=:red)
    end
    
    # Add diagonal line for perfect predictions
    min_f, max_f = extrema([true_forces_all; pred_forces_all])
    plot!(p_forces, [min_f, max_f], [min_f, max_f], 
          linestyle=:dash, 
          linewidth=2, 
          color=:black,
          label="Perfect prediction")
    
    # Save plot to file
    savefig(p_forces, filename)
    println("Saved forces plot: $filename")
    
    return p_forces
end

function plot_energy_comparison(raw_data, model, filename; marked=Int[])
    """
    Plot true vs predicted energies for a dataset and save to file.
    
    Arguments:
    - raw_data: Array of atomic structures (with true energies)
    - model: ACE model for predictions
    - filename: Path where the plot will be saved
    - marked: Indices of structures to highlight in red
    """
    
    # Extract true energies from the structures
    true_energies = ustrip.([at.system_data.energy for at in raw_data])
    
    # Compute predicted energies
    pred_energies = ustrip.([potential_energy(at, model) for at in raw_data])
    
    # Separate marked and unmarked points
    all_indices = 1:length(raw_data)
    unmarked_indices = [i for i in all_indices if !(i in marked)]
    marked_indices = [i for i in all_indices if i in marked]
    
    # Create scatter plot - unmarked points first
    p = scatter(true_energies[unmarked_indices], pred_energies[unmarked_indices],
           xlabel="True Energy (eV)",
           ylabel="Predicted Energy (eV)",
           title="True vs Predicted Energies",
           label="Unmarked",
           alpha=0.6,
           markersize=4,
           color=:blue)
    
    # Add marked points in red
    if !isempty(marked_indices)
        scatter!(p, true_energies[marked_indices], pred_energies[marked_indices],
               label="Marked",
               alpha=0.8,
               markersize=6,
               color=:red)
    end
    
    # Add diagonal line for perfect predictions
    min_e, max_e = extrema([true_energies; pred_energies])
    plot!(p, [min_e, max_e], [min_e, max_e], 
          linestyle=:dash, 
          linewidth=2, 
          color=:black,
          label="Perfect prediction")
    
    # Save plot to file
    savefig(p, filename)
    println("Saved energy plot: $filename")
    
    return p
end

"""
    generate_ptd_diagnostics_and_log(replicas, temperatures, mcmc_rates, exchange_rates, trajs, 
                                      output_dir, iteration, r_cut)

Generate comprehensive diagnostic plots and save terminal output for distributed parallel tempering.
Creates a 9-panel figure similar to the testing diagnostics and saves detailed logs.

# Arguments
- `replicas`: Array of sample arrays, one per temperature
- `temperatures`: Temperature values for each replica
- `mcmc_rates`: MCMC acceptance rate for each replica  
- `exchange_rates`: Exchange acceptance rate between neighboring replicas
- `trajs`: Trajectory data for each replica (energy, acc_rate)
- `output_dir`: Directory where plots and logs will be saved
- `iteration`: Current active learning iteration number
- `r_cut`: Cutoff radius for RDF/ADF calculations (with units)

# Returns
- Nothing (saves files to disk)
"""
function generate_ptd_diagnostics_and_log(replicas, temperatures, mcmc_rates, exchange_rates, trajs, 
                                          output_dir, iteration, r_cut)
    
    n_replicas = length(replicas)
    
    # Prepare log content
    log_content = String[]
    push!(log_content, "="^70)
    push!(log_content, "Parallel Tempering Diagnostics - Iteration $iteration")
    push!(log_content, "="^70)
    push!(log_content, "")
    
    # Sampling results
    push!(log_content, "--- Sampling Results ---")
    for i in 1:n_replicas
        push!(log_content, "Replica $i (T=$(round(temperatures[i], digits=1)) K): " *
               "$(length(replicas[i])) samples, " *
               "MCMC acceptance: $(round(mcmc_rates[i], digits=3))")
    end
    push!(log_content, "")
    
    # MCMC acceptance rate quality
    push!(log_content, "--- MCMC Acceptance Rate Quality ---")
    for i in 1:n_replicas
        acc_rate = mcmc_rates[i]
        status = if 0.15 < acc_rate < 0.4
            "✓ Good"
        elseif 0.1 < acc_rate < 0.5
            "✓ OK"
        else
            "⚠ Check"
        end
        push!(log_content, "  Replica $i: $status ($(round(acc_rate, digits=3)))")
    end
    push!(log_content, "")
    
    # Exchange rates
    push!(log_content, "--- Exchange Acceptance Rates ---")
    for i in 1:(n_replicas-1)
        rate = exchange_rates[i]
        status = 0.1 < rate < 0.5 ? "✓" : "⚠"
        push!(log_content, "  $(round(temperatures[i], digits=1)) K ↔ $(round(temperatures[i+1], digits=1)) K: " *
               "$(round(rate, digits=3)) $status")
    end
    push!(log_content, "")
    
    # Energy statistics per replica
    push!(log_content, "--- Energy Statistics per Replica ---")
    mean_energies = Float64[]
    for i in 1:n_replicas
        energies = trajs[i].energy
        mean_energy = mean(energies)
        std_energy = std(energies)
        n_atoms = length(replicas[i][1])
        mean_energy_per_atom = mean_energy / n_atoms
        std_energy_per_atom = std_energy / n_atoms
        push!(mean_energies, mean_energy_per_atom)
        
        push!(log_content, "Replica $i (T=$(round(temperatures[i], digits=1)) K):")
        push!(log_content, "  Mean energy: $(round(mean_energy, digits=6)) eV ($(round(mean_energy_per_atom, digits=6)) eV/atom)")
        push!(log_content, "  Std energy: $(round(std_energy, digits=6)) eV ($(round(std_energy_per_atom, digits=6)) eV/atom)")
    end
    push!(log_content, "")
    
    # Structural analysis
    push!(log_content, "--- Structural Analysis ---")
    all_rdf_distances = Vector{Vector{Float64}}(undef, n_replicas)
    all_adf_angles_deg = Vector{Vector{Float64}}(undef, n_replicas)
    
    for i in 1:n_replicas
        samples_vec = FlexibleSystem.(replicas[i])
        rdf_data = ACEpotentials.get_rdf(samples_vec, r_cut; rescale=true)
        all_rdf_distances[i] = rdf_data[(:Si, :Si)]
        adf_data = ACEpotentials.get_adf(samples_vec, r_cut)
        all_adf_angles_deg[i] = rad2deg.(adf_data)
        
        push!(log_content, "  Replica $i (T=$(round(temperatures[i], digits=0)) K): " *
               "$(length(all_rdf_distances[i])) Si-Si pairs, " *
               "$(length(all_adf_angles_deg[i])) Si-Si-Si triplets")
    end
    push!(log_content, "")
    push!(log_content, "="^70)
    
    # Save log file
    log_filename = joinpath(output_dir, "ptd_diagnostics_iter_$(lpad(iteration, 3, '0')).log")
    open(log_filename, "w") do io
        for line in log_content
            println(io, line)
        end
    end
    println("Saved PT diagnostics log: $log_filename")
    
    # Generate 9-panel diagnostic plot
    # Plot 1: Energy trajectories for all replicas
    p1 = plot(title="Energy Trajectories (All Replicas)", 
             xlabel="Sample", ylabel="Energy per atom (eV)", 
             legend=:outerright, size=(800, 400))
    n_atoms = length(replicas[1][1])
    for i in 1:n_replicas
        plot!(p1, trajs[i].energy ./ n_atoms, 
             label="T=$(round(temperatures[i], digits=0)) K",
             alpha=0.7, linewidth=1)
    end
    
    # Plot 2: Energy distributions for all replicas
    p2 = plot(title="Energy Distributions", 
             xlabel="Energy per atom (eV)", ylabel="Density",
             legend=:outerright, size=(800, 400))
    for i in 1:n_replicas
        histogram!(p2, trajs[i].energy ./ n_atoms, 
                  bins=30, 
                  normalize=:pdf, 
                  alpha=0.5,
                  label="T=$(round(temperatures[i], digits=0)) K")
    end
    
    # Plot 3: Mean energy vs temperature
    p3 = scatter(temperatures, mean_energies,
                xlabel="Temperature (K)", 
                ylabel="Mean Energy per atom (eV)",
                title="Energy-Temperature Scaling",
                label="Sampled",
                markersize=8,
                color=:blue,
                size=(800, 400))
    plot!(p3, temperatures, mean_energies,
         linewidth=2,
         linestyle=:dash,
         label="Trend",
         color=:red)
    
    # Plot 4: Exchange rate matrix
    exchange_matrix = zeros(n_replicas, n_replicas)
    for i in 1:(n_replicas-1)
        exchange_matrix[i, i+1] = exchange_rates[i]
        exchange_matrix[i+1, i] = exchange_rates[i]
    end
    
    p4 = heatmap(1:n_replicas, 1:n_replicas, exchange_matrix,
                xlabel="Replica Index",
                ylabel="Replica Index",
                title="Exchange Rate Matrix",
                c=:viridis,
                clims=(0, 1),
                colorbar_title="Accept Rate",
                aspect_ratio=:equal,
                size=(800, 400))
    
    for i in 1:(n_replicas-1)
        annotate!(p4, i+1, i, text("$(round(exchange_rates[i], digits=2))", 10, :white))
        annotate!(p4, i, i+1, text("$(round(exchange_rates[i], digits=2))", 10, :white))
    end
    
    # Plot 5: RDF for target temperature
    si_si_distances = all_rdf_distances[1]
    p5 = histogram(si_si_distances,
                  bins=100,
                  normalize=:pdf,
                  xlabel="Distance (Å)",
                  ylabel="g(r)",
                  title="RDF (Si-Si) at T=$(round(temperatures[1], digits=0)) K",
                  label="",
                  alpha=0.7,
                  color=:orange,
                  xlims=(0, ustrip(u"Å", r_cut)),
                  size=(800, 400))
    vline!(p5, [2.35], linestyle=:dash, color=:red, linewidth=1, label="1st shell (~2.35 Å)")
    
    # Plot 6: ADF for target temperature
    si_si_si_angles_deg = all_adf_angles_deg[1]
    p6 = histogram(si_si_si_angles_deg,
                  bins=100,
                  normalize=:pdf,
                  xlabel="Angle (degrees)",
                  ylabel="P(θ)",
                  title="ADF (Si-Si-Si) at T=$(round(temperatures[1], digits=0)) K",
                  label="",
                  alpha=0.7,
                  color=:purple,
                  xlims=(0, 180),
                  size=(800, 400))
    vline!(p6, [109.47], linestyle=:dash, color=:red, linewidth=1, label="Tetrahedral (109.47°)")
    
    # Plot 7: RDF for all temperatures
    p7 = plot(title="Radial Distribution Functions (All Temperatures)", 
             xlabel="Distance (Å)", 
             ylabel="g(r)",
             legend=:outerright,
             size=(800, 400))
    for i in 1:n_replicas
        histogram!(p7, all_rdf_distances[i],
                  bins=100,
                  normalize=:pdf,
                  alpha=0.5,
                  label="T=$(round(temperatures[i], digits=0)) K",
                  xlims=(0, ustrip(u"Å", r_cut)))
    end
    vline!(p7, [2.35], linestyle=:dash, color=:black, linewidth=2, label="1st shell (~2.35 Å)")
    
    # Plot 8: ADF for all temperatures
    p8 = plot(title="Angular Distribution Functions (All Temperatures)", 
             xlabel="Angle (degrees)", 
             ylabel="P(θ)",
             legend=:outerright,
             size=(800, 400))
    for i in 1:n_replicas
        histogram!(p8, all_adf_angles_deg[i],
                  bins=100,
                  normalize=:pdf,
                  alpha=0.5,
                  label="T=$(round(temperatures[i], digits=0)) K",
                  xlims=(0, 180))
    end
    vline!(p8, [109.47], linestyle=:dash, color=:black, linewidth=2, label="Tetrahedral (109.47°)")
    
    # Plot 9: Acceptance rates as function of iterations
    p9 = plot(title="Acceptance Rates vs Iteration", xlabel="Sample", 
             ylabel="Acceptance Rate", legend=:outerright, 
             size=(800, 400), ylims=(0, 1))
    for i in 1:n_replicas
        acc_rates = trajs[i].acc_rate
        # Plot instantaneous acceptance rates
        plot!(p9, 1:length(acc_rates), acc_rates, 
              label="T=$(round(temperatures[i], digits=0)) K", 
              alpha=0.5, linewidth=1, color=i)
        
        # Compute and plot cumulative average
        cumulative_avg = cumsum(acc_rates) ./ (1:length(acc_rates))
        plot!(p9, 1:length(cumulative_avg), cumulative_avg,
              label="T=$(round(temperatures[i], digits=0)) K (cumul.)",
              linewidth=2.5, color=i, linestyle=:solid)
        
        # Add reference line at overall acceptance rate
        hline!(p9, [mcmc_rates[i]], linestyle=:dash, color=i, alpha=0.3, label="")
    end
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9,
                        layout=(5,2), size=(1600, 2200), margin=5Plots.mm)
    
    # Save plot
    plot_filename = joinpath(output_dir, "ptd_diagnostics_iter_$(lpad(iteration, 3, '0')).png")
    savefig(combined_plot, plot_filename)
    println("Saved PT diagnostics plot: $plot_filename")
    
    return nothing
end