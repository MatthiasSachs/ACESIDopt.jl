using ExtXYZ, AtomsBase

# Load a test system
raw_data = ExtXYZ.load("/Users/msachs2/Documents/GitHub-2/testASECalculators/results/glycine_remd_parallel2/replica_0_train_100frames.xyz")

test_system = raw_data[1]
println("Type: ", typeof(test_system))
println("Fields: ", fieldnames(typeof(test_system)))
println("\nAtom data fields: ", fieldnames(typeof(test_system.atom_data)))
println("\nSystem data fields: ", fieldnames(typeof(test_system.system_data)))
