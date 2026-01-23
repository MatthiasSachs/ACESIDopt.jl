using AtomsBase
using ExtXYZ
using PythonCall

# Read the alanine-1.xyz file
xyz_file = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/alanin/alanine-2.xyz"
atoms = ExtXYZ.load(xyz_file)

# Display information about the loaded structure
println("Loaded structure from: ", xyz_file)
println("Number of atoms: ", length(atoms))
println("Atomic symbols: ", [atomic_symbol(atom) for atom in atoms])

#%%
using AtomsBuilder, AtomsCalculators, AtomsBase
using AtomsBase: FlexibleSystem, FastSystem
using AtomsCalculators: potential_energy
function _flexiblesystem(sys)
   c3ll = cell(sys)
   particles = [ AtomsBase.Atom(species(sys, i), position(sys, i))
                 for i = 1:length(sys) ]
   return FlexibleSystem(particles, c3ll)
end;
#%%
_flexiblesystem(atoms)

#%%
# using Pkg
# Pkg.add("ASEconvert")
# Pkg.add("AtomsBase")
# Pkg.add("PythonCall") 

#%%
using ASEconvert

# 1. Make an ASE atoms object in Python (via Julia)
# The 'pyimport("ase.build").bulk' returns a Python object
atoms_ase = pyimport("ase.build").bulk("Si") * pytuple((4, 1, 1)) # Creates a silicon supercell

# 2. Convert the Python ASE object to a Julia AtomsBase-compatible structure
atoms_ab = pyconvert(AbstractSystem, atoms_ase)

# 3. (Optional) Convert the Julia structure back to an ASE object
newatoms_ase = convert_ase(atoms_ab)

convert_ase(atoms)
atom2 = pyconvert(AbstractSystem, convert_ase(atoms))
atoms


#%%
math = pyimport("math")
math.pi
a= math.sin(math.pi / 4)
a*2
pyconvert(Float64, a)

#%%
# Define a Python function using a multiline string
pyscript = py"""
def print_one_number(my_number):
    print(my_number)
    return my_number * 2
"""

# Call the Python function from Julia
my_number_to_print = 10
# Access the function and call it, result is a Julia object
result = pyscript.print_one_number(my_number_to_print)
println("Result in Julia: ", result)

#%%
# Assuming you have a file named 'simple.py' in your working directory
# with some Python code (e.g., defining a variable 'data = [1, 2, 3]')

py"""
def run_script(filename):
    with open(filename, 'r') as f:
        exec(f.read(), globals())
    # You might need to adjust 'globals()' based on your script's needs
"""


# Execute the script
py"run_script"("simple.py")

# Now you can access variables or functions defined in the script
# For example, if 'simple.py' defined a variable 'data':
data_from_python = py"data"
println("Data from Python script: ", data_from_python)

