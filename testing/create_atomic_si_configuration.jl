# Minimal 2-atom diamond-Si primitive cell
# Compatible with ACEpotentials.jl ecosystem (AtomsBase / AtomsCalculators)

using AtomsBase
using Unitful
using LinearAlgebra
using ExtXYZ

# --- lattice constant (use your DFT-relaxed value in practice)
a0 = 5.43u"Å"

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

# Create Atom objects
atoms = [Atom(species[i], positions_cart[i]) for i in 1:length(species)]

# Create periodic system with cell vectors
si_2 = periodic_system(
    atoms,
    [bounding_box[:,1], bounding_box[:,2], bounding_box[:,3]]
)

# --- sanity checks / inspection
println("Number of atoms: ", length(si_2))
println("Cell vectors (Å):")
println(bounding_box)
println("Cell lengths (Å): ", [norm(bounding_box[:,i]) for i in 1:3])
println("Cell angles (deg): ",
    [acosd(dot(bounding_box[:,i], bounding_box[:,j]) /
           (norm(bounding_box[:,i])*norm(bounding_box[:,j])))
     for (i,j) in ((1,2),(1,3),(2,3))])

# --- Save to ExtXYZ file
output_file = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-2atom.xyz")
println("\nSaving configuration to: $output_file")
ExtXYZ.save(output_file, si_2)
println("Configuration saved successfully!")


# Alternative version with slightly larger box
println("\nCreating alternative configuration with larger lattice constant...")
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

# Create Atom objects
atoms = [Atom(species[i], positions_cart[i]) for i in 1:length(species)]

# Create periodic system with cell vectors
si_2 = periodic_system(
    atoms,
    [bounding_box[:,1], bounding_box[:,2], bounding_box[:,3]]
)

# --- sanity checks / inspection
println("Number of atoms: ", length(si_2))
println("Cell vectors (Å):")
println(bounding_box)
println("Cell lengths (Å): ", [norm(bounding_box[:,i]) for i in 1:3])
println("Cell angles (deg): ",
    [acosd(dot(bounding_box[:,i], bounding_box[:,j]) /
           (norm(bounding_box[:,i])*norm(bounding_box[:,j])))
     for (i,j) in ((1,2),(1,3),(2,3))])

# --- Save to ExtXYZ file
output_file = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-2atom-large.xyz")
println("\nSaving configuration to: $output_file")
ExtXYZ.save(output_file, si_2)
println("Configuration saved successfully!")