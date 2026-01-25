"""
Test suite for HarmonicCalculator

Tests the implementation of the harmonic oscillator calculator
with various molecular systems and configurations.
"""

using AtomsBase
using AtomsBase: isolated_system, Atom
using Unitful
using LinearAlgebra
using StaticArrays
using AtomsCalculators
using AtomsCalculators: potential_energy, forces, virial, energy_unit, length_unit
using ACESIDopt: HarmonicCalculator
using ACESIDopt
# Load the harmonic calculator
#  include("../src/harmonicCalculator.jl")

println("="^60)
println("Testing HarmonicCalculator Implementation")
println("="^60)

# Test 1: Unit functions
println("\n--- Test 1: Unit Functions ---")
calc = HarmonicCalculator(:Si, 1.0)
println("Energy unit: $(energy_unit(calc))")
println("Length unit: $(length_unit(calc))")
@assert energy_unit(calc) == u"eV"
@assert length_unit(calc) == u"Å"
println("✓ Unit functions work correctly")

# Test 2: Single atom at origin
println("\n--- Test 2: Single Atom at Origin ---")
system = isolated_system([
    Atom(:Si, [0.0, 0.0, 0.0]u"Å")
])
calc = HarmonicCalculator(:Si, 1.0)

E = potential_energy(system, calc)
F = forces(system, calc)
V = virial(system, calc)

println("Energy: $E (expected: 0.0 eV)")
println("Forces: $F (expected: [0, 0, 0] eV/Å)")
println("Virial: $V (expected: zeros(3,3) eV)")

@assert abs(ustrip(u"eV", E)) < 1e-10
@assert all(abs.(ustrip.(u"eV/Å", F[1])) .< 1e-10)
@assert all(abs.(ustrip.(u"eV", V)) .< 1e-10)
println("✓ Single atom at origin has zero energy, forces, and virial")

# Test 3: Single atom displaced along x-axis
println("\n--- Test 3: Single Atom Displaced Along X-axis ---")
system = isolated_system([
    Atom(:Si, [2.0, 0.0, 0.0]u"Å")
])
calc = HarmonicCalculator(:Si, 1.0)

E = potential_energy(system, calc)
F = forces(system, calc)
V = virial(system, calc)

# Expected: E = 0.5 * k * r^2 = 0.5 * 1.0 * 4.0 = 2.0 eV
# Expected: F = -k * r = -1.0 * [2, 0, 0] = [-2, 0, 0] eV/Å
# Expected: V = k * r ⊗ r = [[4, 0, 0], [0, 0, 0], [0, 0, 0]] eV

println("Energy: $E (expected: 2.0 eV)")
println("Forces: $F (expected: [-2.0, 0.0, 0.0] eV/Å)")
println("Virial diagonal: [$(V[1,1]), $(V[2,2]), $(V[3,3])]")

@assert abs(ustrip(u"eV", E) - 2.0) < 1e-10
@assert abs(ustrip(u"eV/Å", F[1][1]) - (-2.0)) < 1e-10
@assert abs(ustrip(u"eV/Å", F[1][2])) < 1e-10
@assert abs(ustrip(u"eV/Å", F[1][3])) < 1e-10
@assert abs(ustrip(u"eV", V[1,1]) - 4.0) < 1e-10
@assert abs(ustrip(u"eV", V[2,2])) < 1e-10
@assert abs(ustrip(u"eV", V[3,3])) < 1e-10
println("✓ Single displaced atom has correct energy, forces, and virial")

# Test 4: Two atoms of same element
println("\n--- Test 4: Two Atoms of Same Element ---")
system = isolated_system([
    Atom(:Si, [1.0, 0.0, 0.0]u"Å"),
    Atom(:Si, [0.0, 1.0, 0.0]u"Å")
])
calc = HarmonicCalculator(:Si, 2.0)

E = potential_energy(system, calc)
F = forces(system, calc)

# Expected: E = 0.5 * 2.0 * (1.0^2 + 1.0^2) = 2.0 eV
# Expected: F1 = -2.0 * [1, 0, 0] = [-2, 0, 0] eV/Å
# Expected: F2 = -2.0 * [0, 1, 0] = [0, -2, 0] eV/Å

println("Energy: $E (expected: 2.0 eV)")
println("Force on atom 1: $(F[1]) (expected: [-2.0, 0.0, 0.0] eV/Å)")
println("Force on atom 2: $(F[2]) (expected: [0.0, -2.0, 0.0] eV/Å)")

@assert abs(ustrip(u"eV", E) - 2.0) < 1e-10
@assert abs(ustrip(u"eV/Å", F[1][1]) - (-2.0)) < 1e-10
@assert abs(ustrip(u"eV/Å", F[2][2]) - (-2.0)) < 1e-10
println("✓ Two atoms of same element work correctly")

# Test 5: Mixed element types
println("\n--- Test 5: Mixed Element Types ---")
system = isolated_system([
    Atom(:Si, [1.0, 0.0, 0.0]u"Å"),
    Atom(:O, [0.0, 2.0, 0.0]u"Å")
])
calc = HarmonicCalculator(Dict(:Si => 1.0, :O => 0.5))

E = potential_energy(system, calc)
F = forces(system, calc)

# Expected: E = 0.5 * 1.0 * 1.0 + 0.5 * 0.5 * 4.0 = 0.5 + 1.0 = 1.5 eV
# Expected: F_Si = -1.0 * [1, 0, 0] = [-1, 0, 0] eV/Å
# Expected: F_O = -0.5 * [0, 2, 0] = [0, -1, 0] eV/Å

println("Energy: $E (expected: 1.5 eV)")
println("Force on Si: $(F[1]) (expected: [-1.0, 0.0, 0.0] eV/Å)")
println("Force on O: $(F[2]) (expected: [0.0, -1.0, 0.0] eV/Å)")

@assert abs(ustrip(u"eV", E) - 1.5) < 1e-10
@assert abs(ustrip(u"eV/Å", F[1][1]) - (-1.0)) < 1e-10
@assert abs(ustrip(u"eV/Å", F[2][2]) - (-1.0)) < 1e-10
println("✓ Mixed element types work correctly")

# Test 6: 3D displacement
println("\n--- Test 6: 3D Displacement ---")
system = isolated_system([
    Atom(:Si, [1.0, 1.0, 1.0]u"Å")
])
calc = HarmonicCalculator(:Si, 1.0)

E = potential_energy(system, calc)
F = forces(system, calc)
V = virial(system, calc)

# Expected: E = 0.5 * 1.0 * (1^2 + 1^2 + 1^2) = 1.5 eV
# Expected: F = -1.0 * [1, 1, 1] = [-1, -1, -1] eV/Å
# Expected: V = 1.0 * [1, 1, 1] ⊗ [1, 1, 1] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]] eV

println("Energy: $E (expected: 1.5 eV)")
println("Forces: $F (expected: [-1.0, -1.0, -1.0] eV/Å)")
println("Virial (should be symmetric): ")
println("  [$(V[1,1]), $(V[1,2]), $(V[1,3])]")
println("  [$(V[2,1]), $(V[2,2]), $(V[2,3])]")
println("  [$(V[3,1]), $(V[3,2]), $(V[3,3])]")

@assert abs(ustrip(u"eV", E) - 1.5) < 1e-10
@assert all(abs.(ustrip.(u"eV/Å", F[1]) .+ 1.0) .< 1e-10)
@assert all(abs.(ustrip.(u"eV", V) .- 1.0) .< 1e-10)
println("✓ 3D displacement works correctly")

# Test 7: Virial-force consistency
println("\n--- Test 7: Virial-Force Consistency ---")
# For harmonic oscillator: W = -r ⊗ F
system = isolated_system([
    Atom(:Si, [2.0, 1.0, 0.5]u"Å")
])
calc = HarmonicCalculator(:Si, 1.5)

F = forces(system, calc)
V = virial(system, calc)

pos = [2.0, 1.0, 0.5]
force = ustrip.(u"eV/Å", F[1])

# Manual calculation: W_ij = -r_i * F_j
W_manual = zeros(3, 3)
for i in 1:3
    for j in 1:3
        W_manual[i, j] = -pos[i] * force[j]
    end
end

println("Virial from calculator:")
println("  [$(V[1,1]), $(V[1,2]), $(V[1,3])]")
println("  [$(V[2,1]), $(V[2,2]), $(V[2,3])]")
println("  [$(V[3,1]), $(V[3,2]), $(V[3,3])]")
println("Manual W = -r ⊗ F:")
println("  [$(W_manual[1,1]), $(W_manual[1,2]), $(W_manual[1,3])]")
println("  [$(W_manual[2,1]), $(W_manual[2,2]), $(W_manual[2,3])]")
println("  [$(W_manual[3,1]), $(W_manual[3,2]), $(W_manual[3,3])]")

for i in 1:3, j in 1:3
    @assert abs(ustrip(u"eV", V[i,j]) - W_manual[i,j]) < 1e-10
end
println("✓ Virial is consistent with forces")

# Test 8: Error handling - missing element
println("\n--- Test 8: Error Handling ---")
system = isolated_system([
    Atom(:C, [1.0, 0.0, 0.0]u"Å")
])
calc = HarmonicCalculator(:Si, 1.0)  # No spring constant for C

println("Testing error handling for undefined element...")
try
    E = potential_energy(system, calc)
    println("✗ Should have thrown an error for undefined element")
catch e
    println("✓ Correctly threw error: $(typeof(e))")
end

# Test 9: Validation - negative spring constant
println("\n--- Test 9: Validation ---")
println("Testing validation for negative spring constant...")
try
    calc = HarmonicCalculator(:Si, -1.0)
    println("✗ Should have thrown an error for negative spring constant")
catch e
    println("✓ Correctly threw error: $(typeof(e))")
end

# Test 10: Keyword arguments acceptance
println("\n--- Test 10: Keyword Arguments ---")
system = isolated_system([
    Atom(:Si, [1.0, 0.0, 0.0]u"Å")
])
calc = HarmonicCalculator(:Si, 1.0)

# Test that kwargs are accepted (even if ignored)
E1 = potential_energy(system, calc)
E2 = potential_energy(system, calc, domain=1:1)
F1 = forces(system, calc)
F2 = forces(system, calc, executor=:serial)
V1 = virial(system, calc)
V2 = virial(system, calc, nlist=nothing)

@assert E1 == E2
@assert F1 == F2
@assert V1 == V2
println("✓ Keyword arguments are accepted")

println("\n" * "="^60)
println("All tests passed! ✓")
println("="^60)
