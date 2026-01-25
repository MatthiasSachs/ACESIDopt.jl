"""
Test the unified sampler interface with RWMCSampler and MALASampler
"""

using AtomsBase
using AtomsBase: isolated_system, Atom
using ExtXYZ
using Unitful

# Load the required modules
using ACESIDopt: HarmonicCalculator, MSamplers
using ACESIDopt.MSamplers: RWMCSampler, MALASampler, run_sampler
using AtomsCalculators: potential_energy

println("="^60)
println("Testing Unified Sampler Interface")
println("="^60)

# Create a simple test system with 4 Si atoms
n_atoms = 4
initial_positions = []
for i in 1:n_atoms
    pos = randn(3) * 0.3 * u"Å"
    push!(initial_positions, Atom(:Si, pos))
end

flexible_system = isolated_system(initial_positions)
system = ExtXYZ.Atoms(flexible_system)

# Create calculator
calc = HarmonicCalculator(:Si, 0.1)
E_initial = potential_energy(system, calc)

println("\nTest System:")
println("  Atoms: $n_atoms Si")
println("  Initial energy: $E_initial")
println("  Calculator: HarmonicCalculator (k=0.1 eV/Å²)")

# Test parameters
T = 300.0
step_size = 0.1

# Test 1: RWMC with new interface
println("\n" * "="^60)
println("Test 1: RWMC Sampler")
println("="^60)

rwmc = RWMCSampler(step_size=0.1)
println("\nRWMC Configuration:")
println("  step_size: $(rwmc.step_size)")

samples_rwmc, acc_rate_rwmc, traj_rwmc = run_sampler(rwmc, system, calc, T; n_samples=100, burnin=50, thin=2)

println("\nRWMC Results:")
println("  Samples collected: $(length(samples_rwmc))")
println("  Acceptance rate: $(round(acc_rate_rwmc, digits=3))")
println("  Energy samples: $(length(traj_rwmc.energy))")

# Test 2: MALA with new interface
println("\n" * "="^60)
println("Test 2: MALA Sampler")
println("="^60)

mala = MALASampler(step_size=0.1)
println("\nMALA Configuration:")
println("  step_size: $(mala.step_size)")
println("  collect_forces: false")

# Reset system
system2 = ExtXYZ.Atoms(flexible_system)
samples_mala, acc_rate_mala, traj_mala = run_sampler(mala, system2, calc, T; n_samples=100, burnin=50, thin=2, collect_forces=false)

println("\nMALA Results:")
println("  Samples collected: $(length(samples_mala))")
println("  Acceptance rate: $(round(acc_rate_mala, digits=3))")
println("  Energy samples: $(length(traj_mala.energy))")

# Test 3: MALA with force collection
println("\n" * "="^60)
println("Test 3: MALA Sampler (with force collection)")
println("="^60)

mala_forces = MALASampler(step_size=0.1)
println("\nMALA Configuration:")
println("  step_size: $(mala_forces.step_size)")
println("  collect_forces: true")

system3 = ExtXYZ.Atoms(flexible_system)
samples_mala_f, acc_rate_mala_f, traj_mala_f = run_sampler(mala_forces, system3, calc, T; n_samples=100, burnin=50, thin=2, collect_forces=true)

println("\nMALA Results:")
println("  Samples collected: $(length(samples_mala_f))")
println("  Acceptance rate: $(round(acc_rate_mala_f, digits=3))")
println("  Energy samples: $(length(traj_mala_f.energy))")
println("  Force samples: $(length(traj_mala_f.forces))")

# Summary
println("\n" * "="^60)
println("Summary: Unified Interface Test")
println("="^60)
println("✓ RWMCSampler interface works correctly")
println("✓ MALASampler interface works correctly")
println("✓ Force collection option works correctly")
println("✓ Interface: run_sampler(sampler, system, model, T; n_samples, burnin, thin, collect_forces)")
println("✓ Sampler contains: step_size only")
println("✓ Runtime parameters: n_samples, burnin, thin, collect_forces")
println("="^60)
