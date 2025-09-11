from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
from bmp import *
from sqnm.vcsqnm_for_ase import aseOptimizer
import numpy as np
import matplotlib.pyplot as plt
import json
from ase.units import GPa
import multiprocessing
from scipy.stats import gaussian_kde
from random import sample



def main():
    samples = read('../data/MEG.extxyz', index=':10')

    for i, sample in enumerate(samples):
        C = calc_c(sample)
        E = calc_E(*C)
        G = calc_G(*C)
        K = calc_K(*C)
        print(f'{i}: E={E:.2f} G={G:.2f} K={K:.2f}')





def calc_energy(ats):
    tats = ats.copy()
    ats.calc = BMPcalc()
    e0 = tats.get_potential_energy()
    opt = aseOptimizer(tats, 
        vc_relax=True, 
        force_tol=1e-2,
        initial_step_size=-0.001,
        nhist_max=10,
        maximalSteps=1500,
    )
    opt.optimize()
    nat = len(tats)
    return e0 / nat, tats.get_potential_energy() / nat


def calc_c(ats, delta=2.e-2, force_tol=0.05): #0.05):
    tats = ats.copy()
    tats.calc = BMPcalc()
    opt = aseOptimizer(tats, 
                vc_relax=True, 
                force_tol=force_tol,
                initial_step_size=-0.001,
                nhist_max=10,
                maximalSteps=1500,
            )
    opt.optimize()

    stress_0 = tats.get_stress(voigt=False)

    c = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            stress_lr = [stress_0] 
            for lr in [+1.]:
                s = tats.copy()
                s.calc = tats.calc
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
            cc[i, j] = (ca + cb) / 2


    c11s = [cc[0,0], cc[1,1], cc[2,2]]
    c12s = [cc[0,1], cc[1,2], cc[2,0], cc[1,0], cc[2,1], cc[0,2]]
    c44s = [cc[3,3], cc[4,4], cc[5,5]]

    return np.mean(c11s) / GPa, np.mean(c12s) / GPa, np.mean(c44s) / GPa

calc_E = lambda c11, c12, c44: (c11-c12) * (c11 + 2 * c12) / (c11 + c12)
calc_G = lambda c11, c12, c44: c44
calc_K = lambda c11, c12, c44: (c11 + 2 * c12) / 3


if __name__ == '__main__':
    main()
