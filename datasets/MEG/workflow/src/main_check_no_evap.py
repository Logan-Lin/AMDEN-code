import numpy as np
import matplotlib.pyplot as plt
from lmpio import read_lammpstrj 
from ase.io import write
from sys import argv
from ase.units import J, mol, kg
from scipy.optimize import curve_fit


V_FACTOR=1.6

def main():
    if len(argv) < 3:
        print('Usage: python main_check_no_evap.py id phase')

    id = int(argv[1])
    phase = argv[2]

    ats = read_lammpstrj(f'data/{id:05d}/evap_init.lammpstrj')[-1]
    v0 = ats.get_volume()

    check_no_evap(id, phase, v0)
    check_no_evap(id, phase, v0)


def check_no_evap(id, phase, v0):
    ts = []
    vs = []
    es = []
    with open(f'data/{id:05d}/{phase}.txt', 'r') as f:
        f.readline()
        f.readline()
        for l in f:
            s = l.split()
            ts.append(float(s[1]))
            vs.append(float(s[2]))
            es.append(float(s[3]))
    ts = np.array(ts)
    vs = np.array(vs)
    es = np.array(es)

    if np.any(vs > v0 * V_FACTOR):
        print(f'ERROR: Evaporation during {phase}ing')
        exit(1)


if __name__ == '__main__':
    main()
