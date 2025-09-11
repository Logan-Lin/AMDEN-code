from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
from bmp import *
from sqnm.vcsqnm_for_ase import aseOptimizer
from lmpio import *
import numpy as np
from ase.units import GPa
from sys import argv
from os import path



def main():
    if len(argv) <= 1:
        print('Usage: python elastic.py <id>')
        return

    id = int(argv[1])
    s = read_lammpsdata(f'data/{id:05d}/sample.data')[-1]

    calc_c(s, id)


def calc_c(ats, id, delta=2.e-2, force_tol=0.05):

    ats.calc = BMPcalc(fixed=False)
    opt = aseOptimizer(ats, 
                vc_relax=True, 
                force_tol=force_tol,
                initial_step_size=-0.001,
                nhist_max=10,
                maximalSteps=1500,
            )
    opt.optimize()

    write(f'data/{id:05d}/opt.extxyz', ats)

    stress_0 = ats.get_stress(voigt=False)

    c = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            stress_lr = [stress_0]
            for lr in [+1]:
                s = ats.copy()
                s.calc = ats.calc
                strain = np.eye(3)
                strain[i, j] += delta * lr
                s.set_cell(s.get_cell(complete=True) @ strain, scale_atoms=True)
                opt = aseOptimizer(s, 
                            vc_relax=False, 
                            force_tol=force_tol,
                            initial_step_size=-0.001,
                            nhist_max=10,
                            maximalSteps=1500,
                        )
                opt.optimize()
                stress = s.get_stress(voigt=False)
                stress_lr.append(stress)

            stress_l, stress_r = stress_lr
            c[:, :, i, j] = (stress_r - stress_l) / (1 * delta)

    cc_idx = [
        (0, 0), # xx
        (1, 1), # yy
        (2, 2), # zz
        (1, 2), # yz
        (2, 0), # zx
        (0, 1)  # xy
    ]
    cc = np.zeros((6, 6))   
    for i in range(6):
        for j in range(6):
            # c[i,j,k,l] should be equal to c[i,j,l,k]
            # c[i,j,k,l] = c[j,i,k,l] because stress is symmetric
            ca = c[cc_idx[i][0], 
                   cc_idx[i][1], 
                   cc_idx[j][0], 
                   cc_idx[j][1]]
            cb = c[cc_idx[i][0], 
                   cc_idx[i][1], 
                   cc_idx[j][1], 
                   cc_idx[j][0]]
            # print(i, j, ca, cb, ca-cb)
            cc[i, j] = (ca + cb) / 2

    # print()
    # for i in range(6):
    #     print(' '.join(f'{x:11.4e}'  for x in cc[i, :]))

    # print()
    for i in range(6):
        print(' '.join(f'{x / GPa:11.4e}'  for x in cc[i, :]))

    c11s = [cc[0,0], cc[1,1], cc[2,2]]
    c12s = [cc[0,1], cc[1,2], cc[2,0], cc[1,0], cc[2,1], cc[0,2]]
    c44s = [cc[3,3], cc[4,4], cc[5,5]]
    print('C11', np.mean(c11s) / GPa, np.std(c11s) / GPa)
    print('C12', np.mean(c12s) / GPa, np.std(c12s) / GPa)
    print('C44', np.mean(c44s) / GPa, np.std(c44s) / GPa)

    print()
    print('Cij [GPa]')
    for i in range(6):
        print(' '.join(f'{x / GPa:11.4e}'  for x in cc[i, :]))


    # Yooungs:
    # (C11 - C12)(C11 + 2C12)/(C11 + C12)


if __name__ == '__main__':
    main() 
