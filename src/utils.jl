using AtomsBase
using AtomsBase: FlexibleSystem, FastSystem
using AtomsCalculators: potential_energy, forces
using ACEfit: count_observations, AbstractData
using LinearAlgebra
using Unitful: ustrip, @u_str
using ExtXYZ: Atoms
using StaticArrays: SVector
export predictive_variance

function mflexiblesystem(sys)
   c3ll = cell(sys)
   particles = [ AtomsBase.Atom(species(sys, i), position(sys, i))
                 for i = 1:length(sys) ]
   return FlexibleSystem(particles, c3ll)
end

function comp_potE_error(dataset, model)
    return sum( (potential_energy(at, model).val - at.system_data.energy)^2 for at in dataset ) / length(dataset)
end

"""
    cholesky_with_jitter(Σ; jitter_fraction=1e-6, max_jitter_fraction=1e-3, jitter_factor=10.0)

Compute Cholesky decomposition of a matrix Σ, adding jitter (multiple of identity) if needed.

This function attempts to compute the Cholesky decomposition of Σ. If the decomposition
fails (typically due to Σ not being sufficiently positive definite), it adds progressively
larger multiples of the identity matrix until the decomposition succeeds.

The initial jitter is scaled by the mean of the diagonal elements of Σ, making it
adaptive to the matrix scale.

# Parameters
- `Σ`: Symmetric matrix for which to compute Cholesky decomposition
- `jitter_fraction`: Initial jitter as fraction of mean(diag(Σ)) (default: 1e-6)
- `max_jitter_fraction`: Maximum jitter as fraction of mean(diag(Σ)) (default: 1e-3)
- `jitter_factor`: Factor by which to increase jitter on each attempt (default: 10.0)

# Returns
- `L`: Lower triangular Cholesky factor such that L*L' ≈ Σ + jitter*I
- `jitter`: Absolute amount of jitter actually used (0 if none was needed)

# Throws
- `ErrorException`: If Cholesky fails even with maximum jitter

# Example
```julia
Σ = [1.0 0.99; 0.99 1.0]  # Nearly singular
L, jitter = cholesky_with_jitter(Σ)
```
"""
function cholesky_with_jitter(Σ; jitter_fraction=1e-6, max_jitter_fraction=1e-3, jitter_factor=10.0)
    n = size(Σ, 1)
    
    # Scale jitter by mean diagonal element (typical scale of Σ)
    diag_mean = tr(Σ) / n  # equivalent to mean(diag(Σ)) but more efficient
    initial_jitter = jitter_fraction * diag_mean
    max_jitter = max_jitter_fraction * diag_mean
    
    jitter = 0.0
    
    # First try without jitter
    try
        L = cholesky(Σ)
        return L, jitter
    catch
        # Cholesky failed, will add jitter
        jitter = initial_jitter
    end
    
    # Try with increasing jitter
    while jitter <= max_jitter
        try
            Σ_jittered = Σ + jitter * I(n)
            L = cholesky(Σ_jittered)
            @warn "Added jitter=$jitter ($(jitter/diag_mean) × mean(diag(Σ))) for Cholesky decomposition"
            return L, jitter
        catch
            jitter *= jitter_factor
        end
    end
    
    error("Cholesky decomposition failed even with maximum jitter=$max_jitter ($(max_jitter_fraction) × mean(diag(Σ)))")
end

function pred_variance(Σ, xstar::Vector{T}, alpha) where {T}
    return transpose(xstar) * Σ * xstar + 1/alpha
end


function predictive_variance(x::Vector, covar::Matrix; var_e=0.0)
    return dot(x, covar * x) + var_e
end
function predictive_variance(x::Vector, covar::Matrix, Psqrt;  var_e=0.0)
    xt = Psqrt \ x
    return predictive_variance(xt, covar; var_e=var_e)
end
function predictive_variance(model, atom::AbstractSystem, covar::Matrix; Psqrt=I, var_e=0.0)
    # Check wheter this should be indeed the sum or variance
    x = sum(site_descriptors(atom, model))
    return predictive_variance(x, covar, Psqrt; var_e=var_e)
end

"""
Computes the expected reduction in variance after observing a candidate point `xtilde`
when predicting at points `xstar`, given the covariance matrix `Σ` and noise precision `alpha`.
"""
function expected_red_variance(Σ, xstar::Vector{T}, xtilde::Vector{T}, alpha) where {T}
    s = Σ * xtilde
    return (transpose(xstar) * s)^2 / (1/alpha + transpose(xtilde) * s)
end


function expected_red_variance(Σ, Xstar::Matrix{T}, xtilde::Vector{T}, alpha) where {T}
    return sum(expected_red_variance(Σ, Xstar[i,:], xtilde, alpha) for i in 1:size(Xstar,1))
end

function expected_red_variance_fast(Σ, Xstar::Matrix{T}, xtilde::Vector{T}, alpha) where {T}
    s = Σ * xtilde
    return sum((Xstar * s).^2) / (1/alpha + transpose(xtilde) * s)
end

"""
Computes the expected reduction in variance after observing a candidate points correspnding to rows of `Xtilde`
when predicting at points `xstar`, given the covariance matrix `Σ` and noise precision `alpha`.
"""
function expected_red_variance(Σ, xstar::Vector{T}, Xtilde::Matrix{T}, alpha) where {T}
    s = Xtilde * Σ 
    xt = s * xstar
    #transpose(xt) * inv(1/alpha * I + s * transpose(xtilde)) * xt
    return transpose(xt) * inv(1/alpha * I + s * transpose(Xtilde)) * xt
end

"""
Computes the sum of expected reduction in variance (over observations in xstar) after observing a candidate points correspnding to rows of `Xtilde`
when predicting at points `xstar`, given the covariance matrix `Σ` and noise precision `alpha`.
"""
function expected_red_variance(Σ, Xstar::Matrix{T}, Xtilde::Matrix{T}, alpha) where {T}
    s = Xtilde * Σ 
    
    G = inv(1/alpha * I + s * transpose(Xtilde))
    evar = 0.0
    for i in 1:size(Xstar,1)
        xt = s * Xstar[i,:]
        evar += transpose(xt) * G * xt
    end
    return evar
end


"""
Assemble feature matrix and target vector for given data and basis.
"""
function row_mapping(data::AbstractVector{<:AbstractData}, basis)
    @info "Assembling linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    return rows
end



# Function to add forces to samples
function add_forces(samples, model)
    """
    Calculate forces for all samples and create new Atoms structures with force data
    
    Parameters:
    - samples: Vector of Atoms structures
    - model: ACE potential model
    
    Returns:
    - Vector of new Atoms structures with forces included in atom_data
    """
    samples_with_forces = [] #Vector{typeof(samples[1])}()
    
    for sample in samples
        # Calculate forces for this sample
        f = forces(sample, model)
        
        # Convert forces to the right format (Vector of Vector{Float64})
        forces_data = [ustrip.(u"eV/Å", f_atom) for f_atom in f]
        
        # Create new atom_data with forces
        updated_atom_data = merge(deepcopy(sample.atom_data), (forces=forces_data,))
        
        # Create new Atoms structure with updated atom_data
        sample_with_forces = Atoms(deepcopy(updated_atom_data), sample.system_data)
        
        push!(samples_with_forces, sample_with_forces)
    end
    
    return samples_with_forces
end

"""
    add_energy(samples, model)

Calculate energy for all samples and create new Atoms structures with energy data.

This function computes potential energy for each sample in the input vector,
then creates new Atoms structures with energy stored in system_data.

# Parameters
- `samples`: Vector of Atoms structures
- `model`: ACE potential model

# Returns
- Vector of new Atoms structures with energy computed from the model

# Example
```julia
samples_with_energy = add_energy(rwmc_samples, model)
```
"""
function add_energy(samples, model)
    samples_with_energy = []
    
    for sample in samples
        # Calculate energy for this sample
        E = ustrip(u"eV", potential_energy(sample, model))
        
        # Create new system_data with energy
        updated_system_data = merge(deepcopy(sample.system_data), (energy=E,))
        
        # Create new Atoms structure with updated system_data
        sample_with_energy = Atoms(deepcopy(sample.atom_data), updated_system_data)
        
        push!(samples_with_energy, sample_with_energy)
    end
    
    return samples_with_energy
end

"""
    add_energy_forces(samples, model)

Calculate energy and forces for all samples and create new Atoms structures with both.

This function computes both potential energy and forces for each sample in the input
vector, then creates new Atoms structures with:
- Energy stored in system_data
- Forces stored in atom_data

# Parameters
- `samples`: Vector of Atoms structures
- `model`: ACE potential model

# Returns
- Vector of new Atoms structures with energy and forces computed from the model

# Example
```julia
samples_enriched = add_energy_forces(rwmc_samples, model)
```
"""
function add_energy_forces(samples, model)
    # samples_enriched = []
    
    # for sample in samples
    #     # Calculate energy and forces for this sample
    #     E = ustrip(u"eV", potential_energy(sample, model))
    #     f = forces(sample, model)
        
    #     # Convert forces to the right format (Vector of Vector{Float64})
    #     forces_data = [ustrip.(u"eV/Å", f_atom) for f_atom in f]
        
    #     # Create new atom_data with forces
    #     updated_atom_data = merge(deepcopy(sample.atom_data), (forces=forces_data,))
        
    #     # Create new system_data with energy
    #     updated_system_data = merge(deepcopy(sample.system_data), (energy=E,))
        
    #     # Create new Atoms structure with updated atom_data and system_data
    #     sample_enriched = Atoms(updated_atom_data, updated_system_data)
        
    #     push!(samples_enriched, sample_enriched)
    # end
    
    # return samples_enriched
    return [ ( f = forces(sample, model);
               E = potential_energy(sample, model); 
               add_energy_forces(sample, E, f))
           for sample in samples ]
end

"""
    add_energy_forces(sample, energy, forces)

Add pre-computed energy and forces to a single Atoms structure.

This function takes a single Atoms structure and pre-computed energy and forces values,
then creates a new Atoms structure with:
- Energy stored in system_data
- Forces stored in atom_data

# Parameters
- `sample`: Single Atoms structure
- `energy`: Potential energy value (Float64, in eV)
- `forces`: Forces as Vector of Vector{Float64} (in eV/Å), one vector per atom

# Returns
- New Atoms structure with energy and forces added

# Example
```julia
E = 10.5  # eV
f = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]  # eV/Å for each atom
sample_enriched = add_energy_forces(sample, E, f)
```
"""
function add_energy_forces(sample::Atoms, energy, forces)
    # Create new atom_data with forces
    updated_atom_data = merge(deepcopy(sample.atom_data), (forces=ustrip.(forces),))
    
    # Create new system_data with energy
    updated_system_data = merge(deepcopy(sample.system_data), (energy=ustrip(u"eV", energy),))
    
    # Create new Atoms structure with updated atom_data and system_data
    sample_enriched = Atoms(updated_atom_data, updated_system_data)
    
    return sample_enriched
end

"""
    convert_forces_to_svector(sample)

Convert forces from Vector{Vector{Float64}} to Vector{SVector{3, Float64}}.

This function takes an Atoms structure with forces stored as Vector{Vector{Float64}}
and creates a new Atoms structure with forces converted to Vector{SVector{3, Float64}}.
This is useful for compatibility with certain packages that expect static arrays.

# Parameters
- `sample`: Atoms structure with forces as Vector{Vector{Float64}}

# Returns
- New Atoms structure with forces as Vector{SVector{3, Float64}}

# Example
```julia
sample_converted = convert_forces_to_svector(raw_data["candidates"][1])
```
"""
function convert_forces_to_svector(sample::Atoms)
    # Check if forces exist in atom_data
    if !haskey(sample.atom_data, :forces)
        error("Sample does not have forces in atom_data")
    end
    
    # Convert forces from Vector{Vector{Float64}} to Vector{SVector{3, Float64}}
    forces_vec = sample.atom_data.forces
    forces_svec = [SVector{3, Float64}(f[1], f[2], f[3]) for f in forces_vec]
    
    # Create new atom_data with converted forces
    updated_atom_data = merge(sample.atom_data, (forces=forces_svec,))
    
    # Create new Atoms structure with updated atom_data
    sample_converted = Atoms(updated_atom_data, sample.system_data)
    
    return sample_converted
end



# Setter functions for AtomsBase.Atoms moved to MSamplers module

# Reminder: getter functions:

# using AtomsBase: atomic_mass
# typeof(system)
# atomic_mass( system,1 )
# species( system,1 )
# system.atom_data.mass
# typeof( system) 
# mass(system,:)
# mass(fsystem,:)
# position(system,:)
# velocity(system,:)
# velocity(fsystem,:)

#=============================================================================
PARAMETER I/O FUNCTIONS
=============================================================================#

using YAML

"""
    save_simulation_parameters(filepath::String, params::Dict)

Save simulation parameters to a YAML file.

Arguments:
- `filepath`: Path to the output YAML file
- `params`: Dictionary containing all simulation parameters
"""
function save_simulation_parameters(filepath::String, params::Dict)
    YAML.write_file(filepath, params)
    println("Saved simulation parameters to: $filepath")
end

"""
    load_simulation_parameters(filepath::String) -> Dict

Load simulation parameters from a YAML file.

Arguments:
- `filepath`: Path to the YAML file containing parameters

Returns:
- Dictionary containing all simulation parameters
"""
function load_simulation_parameters(filepath::String)
    if !isfile(filepath)
        error("Parameter file not found: $filepath")
    end
    params = YAML.load_file(filepath)
    println("Loaded simulation parameters from: $filepath")
    return params
end