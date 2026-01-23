using Plots
using AtomsCalculators: potential_energy, forces
using Unitful: ustrip

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