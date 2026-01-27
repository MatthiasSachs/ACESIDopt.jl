# Minimal 2-atom diamond-Si primitive cell
# Compatible with ACEpotentials.jl ecosystem (AtomsBase / AtomsCalculators)

using AtomsBase
using Unitful
using LinearAlgebra
using ExtXYZ

# --- lattice constant (use your DFT-relaxed value in practice)
a0 = 6.43u"Å"

# --- primitive diamond lattice vectors (rhombohedral)
a1 = a0/2 .* [0.0, 1.0, 1.0]
a2 = a0/2 .* [1.0, 0.0, 1.0]
a3 = a0/2 .* [1.0, 1.0, 0.0]

bounding_box = hcat(a1, a2, a3)

# --- atomic basis (fractional coordinates)
species = [:Si, :Si]
positions_frac = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
]

# --- build periodic system (2 atoms)
# Convert fractional to Cartesian coordinates  
positions_cart = [bounding_box * pos for pos in positions_frac]


# Create periodic system with cell vectors
ps_atoms= periodic_system(
    [Atom(species[i], positions_cart[i]) for i in 1:length(species)],
    [bounding_box[:,1], bounding_box[:,2], bounding_box[:,3]]
)

atoms = ExtXYZ.Atoms(ps_atoms)
#atoms.atom_data.position[2] =  
bounding_box * [.1,.1,.1]
atoms.atom_data.position
#%%
using ProgressMeter
using AtomsCalculators: potential_energy
using Statistics
model = EmpiricalPotentials.StillingerWeber()
n_points_per_side = 100
dh = 1.0/n_points_per_side
n_steps = Int(1.0 / dh) + 1
E = zeros(n_steps, n_steps, n_steps)
@showprogress for (i,x) in enumerate(0:dh:1.0)
    for (j,y) in enumerate(0:dh:1.0)
        for (k,z) in enumerate(0:dh:1.0)
            atoms.atom_data.position[2] = bounding_box * [x,y,z]
            E[i,j,k] = ustrip(potential_energy(atoms,model))
        end
    end
end


#%%
using StatsPlots
using AtomsBase: cell_vectors

# Function to randomize positions uniformly in the cell
function randomize_positions!(atoms)
    cell = cell_vectors(atoms)
    n_atoms = length(atoms)
    
    # Generate new positions uniformly in the cell
    for i in 1:n_atoms
        # Position = sum of cell vectors with random coefficients [0,1]
        atoms.atom_data.position[i] = sum(cell[j] * rand() for j in 1:3)
    end
end

# Resample 10000 configurations and compute their energies
n_samples = 100000
E_random = zeros(n_samples)

println("\nResampling $n_samples random configurations...")
@showprogress for i in 1:n_samples
    randomize_positions!(atoms)
    E_random[i] = ustrip(potential_energy(atoms, model))
end

# Generate histogram of random configuration energies
p_random = histogram(E_random, xlabel="Energy (eV)", ylabel="Frequency", 
                     title="Energy Distribution from Random Configurations (n=$n_samples)",
                     bins=50, legend=false)

# Print statistics for random configurations
println("\nRandom Configuration Energy Statistics:")
println("  Min: $(minimum(E_random)) eV")
println("  Max: $(maximum(E_random)) eV")
println("  Mean: $(mean(E_random)) eV")
println("  Median: $(median(E_random)) eV")
println("  Std: $(std(E_random)) eV")
println("  Fraction with E < 1000 eV: $(sum(E_random .< 1000) / n_samples * 100)%")

# Save the histogram
savefig(p_random, joinpath(dirname(@__FILE__), "energy_histogram_random.png"))
println("Histogram saved to energy_histogram_random.png")

display(p_random)

# Original grid-based histogram
p = histogram(E[E.<1E3], xlabel="Energy (eV)", ylabel="Frequency", 
              title="Energy Distribution over Position Grid",
              bins=1000, legend=false)

println("\nGrid-based Energy Statistics:")
println("  Min: $(minimum(E)) eV")
println("  Max: $(maximum(E)) eV")
println("  Mean: $(mean(E)) eV")
println("  Median: $(median(E)) eV")
println("  Std: $(std(E)) eV")

savefig(p, joinpath(dirname(@__FILE__), "energy_histogram_grid.png"))
println("Histogram saved to energy_histogram_grid.png")

display(p)




# # --- Save to ExtXYZ file
# output_file = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-2atom.xyz")
# println("\nSaving configuration to: $output_file")
# ExtXYZ.save(output_file, atoms)
# println("Configuration saved successfully!")
