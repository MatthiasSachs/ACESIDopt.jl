"""
    HarmonicCalculator

A simple harmonic oscillator potential centered at the origin [0,0,0].
Each atom experiences a harmonic restoring force based on its element type.

The potential energy for atom i is:
    U_i = 0.5 * k_i * |r_i|^2

where k_i is the spring constant for the element type of atom i.

# Fields
- `spring_constants::Dict{Symbol, Float64}`: Spring constants (eV/Å²) for each element type

# Example
```julia
using AtomsBase, Unitful
using AtomsCalculators: potential_energy, forces

# Create calculator with spring constants
calc = HarmonicCalculator(Dict(:Si => 1.0, :O => 2.0))

# Use with an AtomsBase system
energy = potential_energy(system, calc)
f = forces(system, calc)
```
"""
struct HarmonicCalculator
    spring_constants::Dict{Symbol, Float64}
    
    function HarmonicCalculator(spring_constants::Dict{Symbol, Float64})
        # Validate that all spring constants are positive
        for (elem, k) in spring_constants
            if k <= 0
                throw(ArgumentError("Spring constant for element $elem must be positive, got $k"))
            end
        end
        new(spring_constants)
    end
end

# Convenience constructor for single element type
HarmonicCalculator(element::Symbol, k::Float64) = HarmonicCalculator(Dict(element => k))

# Import required functions from AtomsCalculators
import AtomsCalculators
import AtomsCalculators: potential_energy, forces, virial, energy_unit, length_unit
using Unitful: ustrip, @u_str
using AtomsBase: position, atomic_symbol
using StaticArrays: SVector, SMatrix

"""
    potential_energy(system, calc::HarmonicCalculator; kwargs...)

Calculate the total potential energy of the system under the harmonic potential.

U_total = Σᵢ 0.5 * k_i * |r_i|²

where the sum is over all atoms and k_i is the spring constant for atom i's element type.
"""
function AtomsCalculators.potential_energy(system, calc::HarmonicCalculator; kwargs...)
    total_energy = 0.0
    
    for i in 1:length(system)
        # Get atom position (with units)
        pos = position(system, i)
        pos_vec = ustrip.(u"Å", pos)  # Convert to Å and strip units
        
        # Get element symbol
        elem = Symbol(atomic_symbol(system, i))
        
        # Get spring constant for this element
        if !haskey(calc.spring_constants, elem)
            throw(ArgumentError("No spring constant defined for element $elem"))
        end
        k = calc.spring_constants[elem]
        
        # Calculate harmonic potential: U = 0.5 * k * r²
        r_squared = sum(pos_vec.^2)
        total_energy += 0.5 * k * r_squared
    end
    
    return total_energy * u"eV"
end

"""
    forces(system, calc::HarmonicCalculator; kwargs...)

Calculate forces on all atoms under the harmonic potential.

F_i = -∇U_i = -k_i * r_i

where k_i is the spring constant for atom i's element type and r_i is the position vector.
"""
function AtomsCalculators.forces(system, calc::HarmonicCalculator; kwargs...)
    natoms = length(system)
    force_array = Vector{SVector{3, typeof(1.0u"eV/Å")}}(undef, natoms)
    
    for i in 1:natoms
        # Get atom position (with units)
        pos = position(system, i)
        pos_vec = ustrip.(u"Å", pos)  # Convert to Å and strip units
        
        # Get element symbol
        elem = Symbol(atomic_symbol(system, i))
        
        # Get spring constant for this element
        if !haskey(calc.spring_constants, elem)
            throw(ArgumentError("No spring constant defined for element $elem"))
        end
        k = calc.spring_constants[elem]
        
        # Calculate force: F = -k * r (restoring force toward origin)
        force_vec = -k * pos_vec
        force_array[i] = SVector{3}(force_vec) * u"eV/Å"
    end
    
    return force_array
end

"""
    energy_unit(calc::HarmonicCalculator)

Return the energy unit used by the calculator.
"""
function AtomsCalculators.energy_unit(calc::HarmonicCalculator)
    return u"eV"
end

"""
    length_unit(calc::HarmonicCalculator)

Return the length unit used by the calculator.
"""
function AtomsCalculators.length_unit(calc::HarmonicCalculator)
    return u"Å"
end

"""
    virial(system, calc::HarmonicCalculator; kwargs...)

Calculate the virial tensor of the system under the harmonic potential.

For a harmonic potential centered at the origin, the virial is:
W = -Σᵢ rᵢ ⊗ Fᵢ = Σᵢ k_i * rᵢ ⊗ rᵢ

Returns a 3×3 symmetric matrix with energy units.
"""
function AtomsCalculators.virial(system, calc::HarmonicCalculator; kwargs...)
    # Initialize virial as a 3×3 matrix
    W = zeros(3, 3)
    
    for i in 1:length(system)
        # Get atom position (with units)
        pos = position(system, i)
        pos_vec = ustrip.(u"Å", pos)  # Convert to Å and strip units
        
        # Get element symbol
        elem = Symbol(atomic_symbol(system, i))
        
        # Get spring constant for this element
        if !haskey(calc.spring_constants, elem)
            throw(ArgumentError("No spring constant defined for element $elem"))
        end
        k = calc.spring_constants[elem]
        
        # Calculate virial contribution: W += k * r ⊗ r
        # This is equivalent to -r ⊗ F where F = -k * r
        for j in 1:3
            for l in 1:3
                W[j, l] += k * pos_vec[j] * pos_vec[l]
            end
        end
    end
    
    return SMatrix{3, 3}(W) * u"eV"
end

export HarmonicCalculator
