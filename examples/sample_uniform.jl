# Generate dataset using query_US and plot histogram of energies

using AtomsBase
using Unitful
using LinearAlgebra
using ExtXYZ
using Statistics
using Plots
using ProgressMeter
using EmpiricalPotentials
using ACEpotentials
using ACESIDopt
using ACESIDopt.QueryModels: query_US
using Random

println("="^70)
println("Generating Dataset with query_US")
println("="^70)

# Set random seed for reproducibility
Random.seed!(1234)

# Create initial configuration (2-atom diamond-Si primitive cell)
a0 = 5.43u"Å"

# Primitive diamond lattice vectors
a1 = a0/2 .* [0.0, 1.0, 1.0]
a2 = a0/2 .* [1.0, 0.0, 1.0]
a3 = a0/2 .* [1.0, 1.0, 0.0]

bounding_box = hcat(a1, a2, a3)

# Atomic basis (fractional coordinates)
species = [:Si, :Si]
positions_frac = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
]

# Convert fractional to Cartesian coordinates  
positions_cart = [bounding_box * pos for pos in positions_frac]

# Create periodic system
ps_atoms = periodic_system(
    [Atom(species[i], positions_cart[i]) for i in 1:length(species)],
    [bounding_box[:,1], bounding_box[:,2], bounding_box[:,3]]
)

# Convert to ExtXYZ.Atoms format
atoms = ExtXYZ.Atoms(ps_atoms)

# Initialize reference model
# Option 1: Use Stillinger-Weber empirical potential
println("\nCreating Stillinger-Weber potential as reference model...")
ref_model_SW = EmpiricalPotentials.StillingerWeber()

# Option 2: Load ACE reference model used in other experiments
ace_model_path = joinpath(dirname(@__FILE__), "..", "models", "Si_ref_model.json")
println("Loading ACE reference model from: $ace_model_path")
ref_model_ACE = ACEpotentials.load_model(ace_model_path)[1]
println("ACE reference model loaded successfully")

# Choose which model to use
ref_model = ref_model_ACE  # Change to ref_model_ACE to use ACE model

# Create initial training data (just one configuration for template)
raw_data_train = [atoms]

# Dummy parameters (not used by query_US but required by signature)
model = nothing
Σ = nothing
α = nothing
Psqrt = nothing
#my_weights = Dict()

# Generate dataset using query_US
n_samples = 100000
println("\nGenerating $n_samples samples using query_US...")

dataset = []
energies = Float64[]

@showprogress for i in 1:n_samples
    selected_system = query_US(raw_data_train, model, ref_model, Σ, α, Psqrt, my_weights)
    push!(dataset, selected_system)
    
    # Extract energy (should be stored in system_data)
    energy = ustrip(potential_energy(selected_system, ref_model))
    push!(energies, energy)
end

println("\nDataset generation complete!")
println("Generated $(length(dataset)) configurations")

# Generate histogram of energies
println("\nGenerating energy histogram...")

p1 = histogram(energies, 
              xlabel="Energy (eV)", 
              ylabel="Frequency", 
              title="Energy Distribution from query_US (n=$n_samples)",
              bins=1000, 
              legend=false)

energies2 = ustrip.([potential_energy(d, ref_model_ACE) for d in dataset])
p2 = histogram(energies2, 
              xlabel="Energy (eV)", 
              ylabel="Frequency", 
              title="Energy Distribution from query_US (n=$n_samples)",
              bins=1000, 
              legend=false)

# Print statistics
println("\nEnergy Statistics:")
println("  Min: $(minimum(energies)) eV")
println("  Max: $(maximum(energies)) eV")
println("  Mean: $(mean(energies)) eV")
println("  Median: $(median(energies)) eV")
println("  Std: $(std(energies)) eV")
println("  Fraction with E < 1000 eV: $(sum(energies .< 1000) / n_samples * 100)%")
println("  Fraction with E < 100 eV: $(sum(energies .< 100) / n_samples * 100)%")
println("  Fraction with E < 10 eV: $(sum(energies .< 10) / n_samples * 100)%")

# Save the histogram
output_file = joinpath(dirname(@__FILE__), "query_US_energy_histogram.png")
savefig(p, output_file)
println("\nHistogram saved to: $output_file")

display(p)

# Optionally save the dataset to XYZ file
println("\nSaving dataset to XYZ file...")
output_xyz = joinpath(dirname(@__FILE__), "query_US_dataset.xyz")
ExtXYZ.save(output_xyz, dataset)
println("Dataset saved to: $output_xyz")

println("\n" * "="^70)
println("Dataset generation and analysis complete!")
println("="^70)
