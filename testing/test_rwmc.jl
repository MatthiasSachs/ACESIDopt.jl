using ACESIDopt
using ACESIDopt.MSamplers
using ACEpotentials, AtomsBase, ExtXYZ

# Load a test system
raw_data = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_train_100frames.xyz")

# Create a simple ACE model
model = ace1_model(elements = [:C, :H, :O, :N],
                   rcut = 5.5,
                   order = 2,
                   totaldegree = 5)

# Test RWMC sampling with a small number of samples
println("Testing RWMC sampling...")
rwmc_initial = deepcopy(raw_data[1])
T_test = 300.0
n_samples = 10
step_size = 0.01
burnin = 5
thin = 1

samples, acceptance, traj = run_rwmc_sampling(
    rwmc_initial, model, n_samples, T_test;
    step_size=step_size, burnin=burnin, thin=thin
)

println("\nTest completed successfully!")
println("Number of samples collected: ", length(samples))
println("Number of energy values: ", length(traj.energy))
println("Acceptance rate: ", round(acceptance, digits=3))
println("First sample energy: ", samples[1].system_data.energy)
println("Energy from trajectory: ", traj.energy[1])
