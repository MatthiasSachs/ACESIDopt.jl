using Pkg
# Uncomment the next line if installing Julia for the first time
# Pkg.Registry.add("General")
# Pkg.activate("..")
# Pkg.status()

#%%
using LaTeXStrings, MultivariateStats, Plots, PrettyTables, Printf,
      Statistics, Suppressor, ExtXYZ, Unitful

using ACEpotentials, AtomsBase

#%%
mace_dataset_train = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_train_100frames.xyz")
mace_dataset_test = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_test_100frames.xyz")

mace_dataset_test[1].system_data.energy


# AtomsBase.position(system, 1)
# fieldnames(typeof(system.atom_data))
# system.atom_data.forces
# system.system_data
# typeof(system)
#%%
model = ace1_model(elements = [:C, :H, :O, :N],
                   rcut = 5.5,
                   order = 2,        # body-order - 1
                   totaldegree = 4 );
#%%
descriptors = []
for system in mace_dataset_train
    struct_descriptor = sum(site_descriptors(system, model)) / length(system)
    push!(descriptors, struct_descriptor)
end
println("Computed descriptors for $(length(descriptors)) systems")

#%%
# system[:energy]
# system[1][:forces]
# fieldnames(typeof(system))
#[:forces]
#%%
using ACEfit, MLJ
using PythonCall
using MLJScikitLearnInterface
solver = ACEfit.QR(lambda=1e-1)
# ARDRegressor = @load ARDRegressor pkg=MLJScikitLearnInterface


# Create the solver itself and give it parameters
# solver = ARDRegressor(
#     max_iter = 300,
#     tol = 1e-3,
#     threshold_lambda = 10000
# )
# solver = ACEfit.SKLEARN_BRR( max_iter = 300)

#solver = ACEfit.SKLEARN_BRR( max_iter = 300)

data_keys = (energy_key = "energy", )

# default_weights() = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0))

# weights = ACEpotentials.default_weights()
# data = ACEpotentials.make_atoms_data(mace_dataset_train, model; 
#                           energy_key =  "energy", 
#                           force_key  = nothing,
#                           virial_key = nothing,
#                           weights = weights)

acefit!(mace_dataset_train, model;
        solver=solver, data_keys...);

@info("Training Errors")
comp_potE_error(mace_dataset_train, model)

#g = compute_errors(mace_dataset_train, model; data_keys...);
#sum( (potential_energy(at, model).val - at.system_data.energy)^2 for at in mace_dataset_train ) / length(mace_dataset_train)

@info("Test Error")
comp_potE_error(mace_dataset_test, model)

#%%

using ACEpotentials
using ACEpotentials.Models
using AtomsCalculators: potential_energy, energy_forces_virial, forces
using AtomsCalculators
fieldnames(typeof(model))
potential_energy(mace_dataset_train[1], model)
# forces(mace_dataset_train[2], model)

function comp_potE_error(dataset, model)
    return sum( (potential_energy(at, model).val - at.system_data.energy)^2 for at in dataset ) / length(dataset)
end

@info("Training Errors")
comp_potE_error(mace_dataset_train, model)

#g = compute_errors(mace_dataset_train, model; data_keys...);
#sum( (potential_energy(at, model).val - at.system_data.energy)^2 for at in mace_dataset_train ) / length(mace_dataset_train)

@info("Test Error")
comp_potE_error(mace_dataset_test, model)



#%%
# Vary totaldegree from 4 to 10 and collect train/test errors
totaldegree_values = 3:6
train_rmse_E = Float64[]
test_rmse_E = Float64[]

# solver = ACEfit.SKLEARN_ARD( max_iter = 300, tol = 1e-3, threshold_lambda = 10000)

# solver = ARDRegressor(
#     max_iter = 300,
#     tol = 1e-3,
#     threshold_lambda = 10000
# ) 
# solver = ACEfit.QR(lambda=1e-2)
solver = ACEfit.SKLEARN_BRR( max_iter = 300)

#ACEfit.QR(lambda=1e-1)
data_keys = (energy_key = "energy", force_key = "forces")

for td in totaldegree_values
    println("\n" * "="^60)
    println("Fitting model with totaldegree = $td")
    println("="^60)
    
    # Create model with current totaldegree
    model = ace1_model(elements = [:C, :H, :O, :N],
                       rcut = 5.5,
                       order = 3,        # body-order - 1
                       totaldegree = td)
    
    # Fit the model
    acefit!(mace_dataset_train, model; solver=solver, data_keys...)
    
    # Compute and store training errors
    @info("Training Errors for totaldegree = $td")
    push!(train_rmse_E, comp_potE_error(mace_dataset_train, model))
    
    # Compute and store test errors
    @info("Test Errors for totaldegree = $td")
    push!(test_rmse_E, comp_potE_error(mace_dataset_test, model))
    
    println("Train RMSE: $(train_rmse_E[end]), Test RMSE: $(test_rmse_E[end])")
end


# Plot the results
plot(totaldegree_values, train_rmse_E, 
     label="Train RMSE", 
     marker=:circle, 
     linewidth=2,
     xlabel="Total Degree",
     ylabel="Energy RMSE",
     title="ACE Model Performance vs Total Degree",
     legend=:topright)
plot!(totaldegree_values, test_rmse_E, 
      label="Test RMSE", 
      marker=:square, 
      linewidth=2)

#%%

