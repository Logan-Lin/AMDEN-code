
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpslib import LAMMPSlib

lmpheader = [
    'units metal',
    'atom_modify map array sort 0 0'
]
lmpcmds = lambda fixed: [
    f"variable potential getenv 'TERSOFF'",
    "pair_style tersoff",
    "pair_coeff * * ${potential}/2007_SiO.tersoff O Si"
]

class TersoffCalc (LAMMPSlib):
    def __init__(self, fixed=False):
        super().__init__(
            lammps_header=lmpheader, 
            lmpcmds=lmpcmds(fixed), 
            log_file=None, 
            atom_types=inv_element_map,  
            atom_type_masses=masses)

element_map = {
     1: 'O',
     2: 'Si',
}

inv_element_map = {v: k for k, v in element_map.items()}
masses = {s: atomic_masses[atomic_numbers[s[:2].strip()]] for s in inv_element_map.keys()}

