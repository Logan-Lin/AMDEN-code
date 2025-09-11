from ase.io import read, write
from bmp import *

def read_lammpstrj(inf):
    ats = read(inf, index=':')
    for s in ats:
        s.set_chemical_symbols([element_map[x] for x in s.get_atomic_numbers()])
    return ats

def read_lammpsdata(inf):
    ats = read(inf, index=':', format='lammps-data', Z_of_type={i: i for i in range(100)})
    for s in ats:
        s.set_chemical_symbols([element_map[x] for x in s.get_atomic_numbers()])
    return ats

def write_lammpsdata(outf, ats):
    write(outf, ats, format='lammps-data', specorder=[element_map[x] for x in range(1, 39)], atom_style='charge')

def convert_to_extxyz(inf, outf):
    ats = read_lammpstrj(inf)
    for s in ats:
        s.set_chemical_symbols([element_map[x] for x in s.get_atomic_numbers()])
    write(outf, ats)
