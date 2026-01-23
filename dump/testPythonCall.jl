using PythonCall

# Test that PythonCall is working
println("PythonCall loaded successfully")

# Create a simple Python lambda function directly
print_one_number = pyeval("lambda x: x * 2", Main)

# Call the Python function from Julia
my_number_to_print = 10
result = print_one_number(my_number_to_print)
println("Result in Julia: ", pyconvert(Int, result))

# Test with Python math module
math = pyimport("math")
println("Pi from Python: ", pyconvert(Float64, math.pi))
println("Sin(π/4) from Python: ", pyconvert(Float64, math.sin(math.pi / 4)))

result
#%%
using PythonCall

i = 3
@pyexec (i=i, j=4) => """
a=i+j
b=i/j
""" => (a::Int64,b::Float64)
typeof(a)
#%%
@pyexec """
def python_sum(i, j):
    return i + j
""" => python_sum
#%%
@pyexec """
import numpy as np

def calculate_average(data):
    # This is a multi-line Python function
    average = np.mean(data)
    return average
""" => calculate_average # Assigns the Python function to a Julia variable
calculate_average([1.0, 2.0, 3.0, 4.0, 5.0 ])

# # Prepare some Julia data
# julia_data = [1.0, 2.0, 3.0, 4.0, 5.0]

# # Call the Python function from Julia
# result = calculate_average_py(julia_data)

# println("The average is: ", result)

#%%
# @pyexec """
# from ase.calculators.emt import EMT
# def calculate_emt_properties(atoms):
#     atoms.calc = EMT()
#     forces = atoms.get_forces()
#     energy = atoms.get_potential_energy()
#     return forces, energy
# """ => calculate_emt_properties
#%%
# @pyexec """
# global EMT
# from ase.calculators.emt import EMT
# def calculate_emt_properties(atoms):
#     atoms.calc = EMT()
#     forces = atoms.get_forces()
#     energy = atoms.get_potential_energy()
#     return energy, forces
# """ => calculate_emt_properties

# #%%
# @pyexec """
# global ORCA, OrcaProfile
# from ase.calculators.orca import ORCA, OrcaProfile
# def calculate_emt_properties3(system):
#     system.calc = ORCA(
#         profile= OrcaProfile(command='/Users/msachs2/Library/orca_6_1_1/orca'),
#         charge=0,                          
#         mult=1,                            
#         orcasimpleinput='HF def2-SVP engrad'
#     )
#     energy = system.get_potential_energy()   
#     # Calculate forces
#     print("Calculating forces...")
#     forces = system.get_forces()
#     return energy, forces
# """ => calculate_emt_properties3
# #%%

# using StaticArrays

# function queryASEModel(system; calculator=:EMT, kwargs...)
#     if calculator==:EMT
#         print("Using EMT calculator")
#         E,F = calculate_emt_properties(convert_ase(system))
#     elseif calculator==:ORCA
#         print("Using ORCA calculator")
#         E,F = calculate_emt_properties3(convert_ase(system))
#     else
#         error("Calculator $(calculator) not implemented yet.")
#     end
#     F = pyconvert(Array{Float64,2}, F)
#     E = pyconvert(Float64, E)
#     return (energy=E, forces=[SVector{3,Float64}(F[i,:]) for i=1:size(F,1)])
# end

#%%

# function queryASEModel(system, calculator=:EMT; kwargs...)
#     E,F = calculate_emt_properties(convert_ase(system))
#     F = pyconvert(Array{Float64,2}, F)
#     E = pyconvert(Float64, E)
#     return (energy=E, forces=[SVector{3,Float64}(F[i,:]) for i=1:size(F,1)])
# end



#%%
using ACESIDopt
using ACESIDopt: mflexiblesystem, queryASEModel
using AtomsBase
using ExtXYZ

xyz_file = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/alanin/alanine-2.xyz"
atoms = ExtXYZ.load(xyz_file)


system = mflexiblesystem(atoms)
a1=queryASEModel(system; calculator=:EMT)
a2=queryASEModel(system; calculator=:ORCA)
(a1.forces-a2.forces)./a1.forces


generate_potential_energy2(convert_ase(system))
#%%
F=pyconvert(Array{Float64,2}, F)

a = pyconvert(AbstractSystem, calculate_emt_properties(convert_ase(system)))

#%%
@pyexec (atoms=atoms,) => """
def calculate_emt_properties2(atoms):
    return 1
""" => test1
#%%
using AtomsBase
using ExtXYZ
using PythonCall

# Read the alanine-1.xyz file
xyz_file = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/alanin/alanine-2.xyz"
atoms = ExtXYZ.load(xyz_file)

using AtomsBuilder, AtomsCalculators
using AtomsBase: FlexibleSystem, FastSystem
#using AtomsCalculators: potential_energy
function _flexiblesystem(sys)
   c3ll = cell(sys)
   particles = [ AtomsBase.Atom(species(sys, i), position(sys, i))
                 for i = 1:length(sys) ]
   return FlexibleSystem(particles, c3ll)
end;
#%%
system = _flexiblesystem(atoms)

using ASEconvert
ase_emt = pyimport("ase.calculators.emt")
calculator = ASEcalculator(ase_emt.EMT())
using AtomsCalculators
AtomsCalculators.energy_forces(system, calculator)
