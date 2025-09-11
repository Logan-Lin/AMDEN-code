
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpslib import LAMMPSlib

lmpheader = [
    'units metal',
    'atom_style charge',
    'atom_modify map array sort 0 0'
]
lmpcmds = [
    f"variable potential getenv 'BMP'",
    # "units           metal",
    # "boundary        p p p",
    # "atom_style      charge",
    "# read_data       ${inputfile}",
    "include ${potential}/in.BMP",
]

class BMPcalc (LAMMPSlib):
    def __init__(self):
        super().__init__(
            lammps_header=lmpheader, 
            lmpcmds=lmpcmds, 
            log_file=None, 
            atom_types=inv_element_map, #{x[:]: inv_element_map[x] for x in inv_element_map}, 
            atom_type_masses=masses)



element_map = {
     1: 'Si',
     2: 'O',
     3: 'Li',
     4: 'Na',
     5: 'K',
     6: 'Fe2+',
     7: 'Fe3+',
     8: 'Al',
     9: 'P',
    10: 'Ca',
    11: 'Be',
    12: 'Sr',
    13: 'Ba',
    14: 'Sc',
    15: 'Ti',
    16: 'Zr',
    17: 'Cr',
    18: 'Mn II',
    19: 'Mn III',
    20: 'Co',
    21: 'Ni',
    22: 'Cu I',
    23: 'Cu II',
    24: 'Ag',
    25: 'Zn',
    26: 'Ge',
    27: 'Sn',
    28: 'Nd',
    29: 'Gd',
    30: 'Er',
    31: 'Ga',
    32: 'Ce III',
    33: 'Ce IV',
    34: 'V IV',
    35: 'V V',
    36: 'Mg',
    37: 'Eu',
    38: 'B',
}

inv_element_map = {v: k for k, v in element_map.items()}


masses = {s: atomic_masses[atomic_numbers[s[:2].strip()]] for s in inv_element_map.keys()}


formal_charges = {
    1 : 4,
    2 : -2,
    3 : 1,
    4 : 1,
    5 : 1,
    6 : 2,
    7 : 3,
    8 : 3,
    9 : 5,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 3,
    15: 4,
    16: 4,
    17: 3,
    18: 2,
    19: 3,
    20: 2,
    21: 2,
    22: 1,
    23: 2,
    24: 1,
    25: 2,
    26: 4,
    27: 4,
    28: 3,
    29: 3,
    30: 3,
    31: 3,
    32: 3,
    33: 4,
    34: 4,
    35: 5,
    36: 2,
    37: 3,
    38: 3,
}

charges = {
    1 :  2.4,
    2 : -1.2,
    3 :  0.6,
    4 :  0.6,
    5 :  0.6,
    6 :  1.2,
    7 :  1.8,
    8 :  1.8,
    9 :  3.0,
    10:  1.2,
    11:  1.2,
    12:  1.2,
    13:  1.2,
    14:  1.8,
    15:  2.4,
    16:  2.4,
    17:  1.8,
    18:  1.2,
    19:  1.8,
    20:  1.2,
    21:  1.2,
    22:  0.6,
    23:  1.2,
    24:  0.6,
    25:  1.2,
    26:  2.4,
    27:  2.4,
    28:  1.8,
    29:  1.8,
    30:  1.8,
    31:  1.8,
    32:  1.8,
    33:  2.4,
    34:  2.4,
    35:  3.0,
    36:  1.2,
    37:  1.8,
    38:  1.8,
}
