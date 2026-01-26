using Pkg
Pkg.activate(".")

using ExtXYZ
using Random
using Printf
using ACESIDopt: convert_forces_to_svector
using AtomsBase
using Unitful


# Parse arguments
# input_file, n_samples, seed = parse_arguments()
input_file = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/experiments/results/ptd_ACE_silicon_dia-primitive-2-large-c/data/replica_1_samples.xyz"
n_samples = 100
seed = 42
# Check if input file exists
if !isfile(input_file)
    println("Error: Input file not found: $input_file")
    exit(1)
end

# Set random seed if provided
if seed !== nothing
    Random.seed!(seed)
    println("Random seed set to: $seed")
end

# Load data
println("\nLoading data from: $input_file")
data_all = ExtXYZ.load(input_file)
n_total = length(data_all)

atoms = data_all[1]
function randomize_positions!(atoms)
    for i in 1:length(atoms)
        atoms.atom_data.position[i] = sum(r * rand() for r in cell_vectors(atoms))
    end
end
# typeof(atoms) <: AbstractSystem
# bounding_box

# atoms = data_all[1]
# typeof(atoms)
# atoms.system_data.cell_vectors
# atoms.atom_data
# rand()


# ustrip.(cell_vectors(atoms))



# # --- primitive diamond lattice vectors (rhombohedral)
# a1 = a0/2 .* [0.0, 1.0, 1.0]
# a2 = a0/2 .* [1.0, 0.0, 1.0]
# a3 = a0/2 .* [1.0, 2.0, 0.0]

# bounding_box = hcat(a1, a2, a3)

# positions_frac = [
#     [0.0, 0.0, 0.0],
#     [1.0, 1.0, 0],
# ]

# bounding_box_vec = [ bounding_box[:,1], bounding_box[:,2], bounding_box[:,3]]
# bounding_box_vec 
# # --- build periodic system (2 atoms)
# # Convert fractional to Cartesian coordinates  
# positions_cart = [bounding_box * pos for pos in positions_frac]

#=

################### Test biased ACE model ####################

=#

#%%
using ACEpotentials, AtomsBase
using ACEpotentials: make_atoms_data
using ACEfit
using LinearAlgebra: I, Diagonal
raw_data_train = data_all[1:100:end]

model = ace1_model(elements = [:Si,],
                   Eref = [:Si => -158.54496821],
                   order = 3,
                   totaldegree = 8);


mm_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))
data_train = make_atoms_data(raw_data_train, model; 
                                energy_key = "energy", 
                                force_key = "forces", 
                                virial_key = nothing, 
                                weights = mm_weights())
A_train, Y_train, W_train = ACEfit.assemble(data_train, model)
Psqrt =  I #_make_prior(model, ACE_PRIOR_ORDER, nothing)
Awp_train = Diagonal(W_train) * (A_train / Psqrt) 
Yw_train = W_train .* Y_train


using LinearAlgebra
using ACEfit: bayesian_linear_regression
solver = ACEfit.BLR(committee_size = 10000, factorization = :svd)
result1 = bayesian_linear_regression(Awp_train, Yw_train; solver.kwargs..., ret_covar = true)
result1["covar"]
coeffs = Psqrt \ result1["C"]   
# dispatch setting of parameters 
ACEpotentials.Models.set_linear_parameters!(model, coeffs)
if haskey(result1, "committee")
    co_coeffs = result1["committee"]
    co_ps_vec = [ Psqrt \ co_coeffs[:,i] for i in 1:size(co_coeffs,2) ]
    set_committee!(model, co_ps_vec)
end


function predictive_variance(x::Vector, covar::Matrix; var_e=0.0)
    return dot(x, covar * x) + var_e
end
function predictive_variance(x::Vector, covar::Matrix, Psqrt;  var_e=0.0)
    xt = Psqrt \ x
    return predictive_variance(xt, covar; var_e=var_e)
end
function predictive_variance(model, atom::AtomsBase.AbstractSystem, covar::Matrix; Psqrt=I, var_e=0.0)
    # Check wheter this should be indeed the sum or variance
    x = sum(site_descriptors(atom, model))
    return predictive_variance(x, covar, Psqrt; var_e=var_e)
end

using AtomsCalculators: potential_energy
potential_energy(atoms, model)

import AtomsCalculators

struct biasedACEModel{T}
    model
    Σ::Matrix{T}
    Psqrt
    Temp
end

function AtomsCalculators.potential_energy(atoms::AtomsBase.AbstractSystem, bmodel::biasedACEModel) 
    kB = 8.617333262e-5  # eV/K
    uc =  predictive_variance(bmodel.model, atoms, bmodel.Σ; Psqrt=bmodel.Psqrt)
    return potential_energy(atoms, bmodel.model) + uc/(kB * bmodel.Temp) * u"eV"
end


bmodel = biasedACEModel(model, result1["covar"], Psqrt, 300.0)
AtomsCalculators.potential_energy(atoms, bmodel) 

