using ACESIDopt
using ACESIDopt: mflexiblesystem, queryASEModel
using AtomsBase
using ExtXYZ

xyz_file = "/Users/msachs2/Documents/GitHub-2/ACESIDopt.jl/data/alanin/alanine-1.xyz"
atoms = ExtXYZ.load(xyz_file)


system = mflexiblesystem(atoms)
a1=queryASEModel(system; calculator=:EMT)
# a2=queryASEModel(system; calculator=:ORCA)
a3=queryASEModel(system; calculator=:MACE)
(a1.forces-a3.forces)./a3.forces


using ASEconvert
using AtomsCalculators
using PythonCall

ase_emt = pyimport("ase.calculators.emt")
calculator = ASEcalculator(ase_emt.EMT())
using AtomsCalculators
AtomsCalculators.energy_forces(system, calculator)

