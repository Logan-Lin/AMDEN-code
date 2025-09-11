import numpy as np
import matplotlib.pyplot as plt
from lmpio import read_lammpstrj 
from ase.io import write
from sys import argv
from ase.units import J, mol, kg
from scipy.optimize import curve_fit

do_plot = False


def main():
    if len(argv) < 2:
        print('Usage: python main_check_melted.py id')

    id = int(argv[1])

    check_melt(id)


def check_melt(id):
    ats = read_lammpstrj(f'data/{id:05d}/melt_uw.lammpstrj')
     
    rmsds = []
    rmsds_resc = [] 

    init = ats[len(ats)//10]
    last = ats[-1]
    init.set_cell(last.get_cell(), scale_atoms=True)

    elrmsds = {}

    # print RMSD for each element separately
    els = init.get_chemical_symbols()
    for el in set(els):
        idxs = [i for i, e in enumerate(els) if e == el]
        elrmsd = np.sqrt(3 * np.mean((init.get_positions()[idxs] - last.get_positions()[idxs])**2))
        elrmsds[el] = elrmsd

    with open(f'data/{id:05d}/rmsd.txt', 'w') as f:
        for el in elrmsds:
            f.write(f'{el} {elrmsds[el]}\n')

    for el in elrmsds:
        if elrmsds[el] < 20.0:
            print(f'ERROR: {el} not melted (rmsd={elrmsds[el]})')
            exit(1)

    return



if __name__ == '__main__':
    main()
