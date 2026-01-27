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


#%%
# Alternative version with slightly larger box
println("\nCreating alternative configuration with larger lattice constant...")
# --- lattice constant (use your DFT-relaxed value in practice)
a0 = 7.43u"Å"

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
output_file = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-2atom-very-large.xyz")
println("\nSaving configuration to: $output_file")
ExtXYZ.save(output_file, si_2)
println("Configuration saved successfully!")


#%%
# Alternative version with slightly larger box
println("\nCreating alternative configuration with larger lattice constant...")
# --- lattice constant (use your DFT-relaxed value in practice)
a0 = 8.43u"Å"

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
output_file = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-2atom-very-very-large.xyz")
println("\nSaving configuration to: $output_file")
ExtXYZ.save(output_file, si_2)
println("Configuration saved successfully!")


#%%
# Alternative version with 4 atoms
using StaticArrays
"""
Build an n×m×l supercell from the primitive diamond-Si cell, using AtomsBase.

For the recommended 2×2×2:
  sys8 = diamond_si_supercell(2,2,2; a=5.431)
"""
function diamond_si_supercell(n1::Int, n2::Int, n3::Int; a::Real = 5.431)
    @assert n1 ≥ 1 && n2 ≥ 1 && n3 ≥ 1

    # Primitive vectors
    a1 = (a/2) .* SVector(0.0, 1.0, 1.0) .* u"Å"
    a2 = (a/2) .* SVector(1.0, 0.0, 1.0) .* u"Å"
    a3 = (a/2) .* SVector(1.0, 1.0, 0.0) .* u"Å"

    # Supercell vectors
    A1, A2, A3 = n1*a1, n2*a2, n3*a3
    cellvecs_super = (A1, A2, A3)

    # Primitive basis in fractional coordinates (w.r.t. a1,a2,a3)
    basis_fracs = (
        SVector(0.0, 0.0, 0.0),
        SVector(0.25, 0.25, 0.25),
    )

    atoms = Atom[]
    for i in 0:(n1-1), j in 0:(n2-1), k in 0:(n3-1)
        shift = SVector(Float64(i), Float64(j), Float64(k))
        for f in basis_fracs
            # Cartesian position: (f + shift) in primitive fractional coords
            r = (f[1] + shift[1]) * a1 +
                (f[2] + shift[2]) * a2 +
                (f[3] + shift[3]) * a3
            push!(atoms, Atom(:Si, r))
        end
    end

    return periodic_system(atoms, cellvecs_super)
end

# --- Example: 2×2×2 supercell (8 atoms) ---
sys8 = diamond_si_supercell(2, 2, 2; a=5.431)
println("Number of atoms: ", length(sys8))
println(sys8)
output_file = joinpath(dirname(@__FILE__), "..", "data", "Si-diamond-primitive-8atoms.xyz")
println("\nSaving configuration to: $output_file")
ExtXYZ.save(output_file, sys8)