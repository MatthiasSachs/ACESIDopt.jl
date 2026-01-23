
using PythonCall
using StaticArrays
using ASEconvert: convert_ase
using Unitful: ustrip, @u_str

# Global variables to hold Python functions
_calculate_emt_properties = nothing
_calculate_orca_properties = nothing
_calculate_mace_properties = nothing
_python_funcs_initialized = false

function _init_python_funcs()
    global _calculate_emt_properties, _calculate_orca_properties, _calculate_mace_properties, _python_funcs_initialized
    
    if !_python_funcs_initialized
        # Initialize EMT calculator function
        pyexec("""
from ase.calculators.emt import EMT
def calculate_emt_properties(atoms):
    atoms.calc = EMT()
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    return energy, forces
""", Main)
        _calculate_emt_properties = pyeval("calculate_emt_properties", Main)
        
        # Initialize ORCA calculator function  
        pyexec("""
from ase.calculators.orca import ORCA, OrcaProfile
def calculate_orca_properties(system):
    system.calc = ORCA(
        profile= OrcaProfile(command='/Users/msachs2/Library/orca_6_1_1/orca'),
        charge=0,
        mult=1,
        orcasimpleinput='HF def2-SVP engrad'
    )
    energy = system.get_potential_energy()
    forces = system.get_forces()
    return energy, forces
""", Main)
        _calculate_orca_properties = pyeval("calculate_orca_properties", Main)
        
        # Initialize MACE calculator function
        pyexec("""
import mace.calculators
def calculate_mace_properties(atoms):
    atoms.calc = mace.calculators.mace_mp(
        model="small",
        dispersion=False,
        default_dtype="float32",
        device='cpu'
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces
""", Main)
        _calculate_mace_properties = pyeval("calculate_mace_properties", Main)
        
        _python_funcs_initialized = true
    end
end

function queryASEModel(system; calculator=:EMT, kwargs...)
    _init_python_funcs()
    
    if calculator==:EMT
        print("Using EMT calculator")
        E,F = _calculate_emt_properties(convert_ase(system))
    elseif calculator==:ORCA
        print("Using ORCA calculator")
        E,F = _calculate_orca_properties(convert_ase(system))
    elseif calculator==:MACE
        print("Using MACE calculator")
        E,F = _calculate_mace_properties(convert_ase(system))
    else
        error("Calculator $(calculator) not implemented yet.")
    end
    F = pyconvert(Array{Float64,2}, F)
    E = pyconvert(Float64, E)
    return (energy=E * u"eV", forces=[SVector{3,Float64}(F[i,:])*u"eV/Å" for i=1:size(F,1)])
end

