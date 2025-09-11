import numpy as np
import matplotlib.pyplot as plt
from lmpio import read_lammpstrj, read_lammpsdata
from ase.io import write
from sys import argv
from ase.units import J, mol, kg
from sys import argv
from scipy.optimize import curve_fit
from sqnm.vcsqnm_for_ase import aseOptimizer
from bmp import BMPcalc
import os

no_quench = True
do_plot = False
t_room = 300 # temperature at which alpha is calculated

def main():
    if len(argv) < 2:
        print('Usage: python main_thermal_expansion.py id')

    id = int(argv[1])
        # eq_trj = read_lammpstrj(f'data/quench/{i:05d}/equlibrate.lammpstrj')
        # write(f'data/quench/{i:05d}/equlibrate.extxyz', eq_trj)

    ts = []
    vs = []
    es = [] # enthalpy | includes kinetic energy
    with open(f'data/{id:05d}/heat.txt', 'r') as f:
        f.readline()
        f.readline()
        for l in f:
            s = l.split()
            ts.append(float(s[1]))
            vs.append(float(s[2]))
            es.append(float(s[4]))
    # ats = read_lammpstrj(f'data/{id:05d}/sample.lammpstrj')[-1]
    ats = read_lammpsdata(f'data/{id:05d}/sample.data')[-1]
    mass = sum(ats.get_masses())
    ts = np.array(ts)
    vs = np.array(vs)
    es = np.array(es)


    alpha, cp = analyze_heating(ts, vs, es, do_plot)

    if not no_quench:
        ts = []
        vs = []
        es = []
        with open(f'data/{id:05d}/quench.txt', 'r') as f:
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

        tg, alpha_tg_1, alpha_tg_2 = find_Tg(ts, vs)


        traj = read_lammpstrj(f'data/{id:05d}/quench_uw.lammpstrj')
        for t in traj:
            t.wrap()

        with open(f'data/{id:05d}/T_HIGH.txt', 'r') as f:
            t_high = float(f.readline())
            
        tg_e, ts, es, e0s = find_Tg_enthalpy(id, t_high, traj)

        with open(f'data/{id:05d}/enthalpy.txt', 'w') as f:
            for t, e0, e in zip(ts, e0s, es):
                f.write(f'{t} {e0} {e}\n')

    # traj = read_lammpstrj(f'data/tg/{id:05d}/heat.lammpstrj')
    # tg_e, ts, es, e0s = find_Tg_enthalpy(50, traj, cooling_rate=-14.75e12)

    # print('id  fomula   alpha     Cp [J/kg/K]    Cp [J/Mol/K]')
    # print(id, ats.symbols, alpha, alpha2, cp / J / mass * kg, cp / len(ats) * mol / J)
    print('alpha        ', alpha)
    # print('alpha_quench ', alpha_quench)
    print('Cp_J/kg/K    ', cp / J / mass * kg)
    print('Cp_J/mol/K   ', cp / len(ats) * mol / J)
    if not no_quench:
        print('Tg_enth_K    ', tg_e)
        print('Tg_K         ', tg)
        print('alpha_tg_1   ', alpha_tg_1)
        print('alpha_tg_2   ', alpha_tg_2)

def analyze_heating(ts, vs, es, do_plot=False):
    # fit linear curve to v(t)
    pv = np.polyfit(ts, vs, 1)

    lts = np.linspace(np.min(ts), np.max(ts), ts.size)
    # print(i, pv)
    if do_plot:
        plt.scatter(ts, vs, s=1)
        plt.plot(lts, np.polyval(pv, lts), color='red')
        plt.show()
    alpha = pv[0] / np.polyval(pv, t_room)

    pe = np.polyfit(ts, es, 1)
    # print(i, pe)
    if do_plot:
        plt.scatter(ts, es, s=1)
        plt.plot(lts, np.polyval(pe, lts), color='red')
        plt.show()
    cp = pe[0] 
    return alpha, cp

def find_Tg(ts, vs):
    # ts = np.linspace(t_max, t_min, len(vs))
    # vs = vs[ts<2700]
    # ts = ts[ts<2700]
    t_min = np.min(ts)
    t_max = np.max(ts)
    # vs = (vs - np.max(vs)) / vs
    
    v_t = lambda t, tg, v1, vg, v2: np.where(t < tg, v1 + (t-t_min) / (tg-t_min) * (vg-v1), vg + (t-tg) / (t_max-tg) * (v2-vg))
    tgs = np.linspace(t_min+50, t_max-200, 1000)
    cs = []
    c_min = 1e20
    popt_opt = None
    tg_opt = None
    p0 = [vs[0], vs[ts.size//2], vs[-1]]
    for tg in tgs:
        f = lambda t, v1, vg, v2: v_t(t, tg, v1, vg, v2)
        # TODO: Implement bounds and rewrite f to ensure that kink is convex
        popt, _ = curve_fit(f, ts, vs, p0=p0)
        p0 = popt
        c = np.mean((vs - f(ts, *popt))**2)
        if c < c_min:
            c_min = c
            popt_opt = popt
            tg_opt = tg
        cs.append(c)
    if do_plot:
        plt.plot(tgs, cs)
        plt.show()

    tg = tg_opt
    f = lambda t: v_t(t, tg, *popt_opt)
    if do_plot:
        ts_sorted = np.sort(ts)
        plt.scatter(ts, vs, s=1)
        plt.axvline(tg, color='red')
        plt.plot(ts_sorted, f(ts_sorted), color='red')
        plt.show()

    v1, vg, v2 = popt_opt
    alpha1 = (vg - v1) / (tg - t_min) / (v1 + vg) * 2
    alpha2 = (v2 - vg) / (t_max - tg) / (vg + v2) * 2
    return tg, alpha1, alpha2

def find_Tg_enthalpy(id, t_high, traj, cooling_rate=5e12):
    # read data from file if it exists
    if os.path.exists(f'data/{id:05d}/enthalpy.txt'):
        with open(f'data/{id:05d}/enthalpy.txt', 'r') as f:
            ts = []
            e0s = []
            es = []
            for l in f:
                s = l.split()
                ts.append(float(s[0]))
                e0s.append(float(s[1]))
                es.append(float(s[2]))

    else:
        # import logging
        # logging.basicConfig(level=logging.INFO)
        n = len(traj)
        n_step = n // 40
        e0s = []
        es = []
        ts = []
        ts_all = t_high - (np.arange(n) + 1) * 1000 * 2.e-15 * cooling_rate

        irange = range(0, n, n_step) 
        if cooling_rate > 0:
            irange = range(15 * n_step, n, n_step)

        for i in irange:
            ats = traj[i]
            ats.calc = BMPcalc()
            e0s.append(ats.get_potential_energy())
            opt = aseOptimizer(ats, 
                        vc_relax=True, 
                        force_tol=0.05,
                        nhist_max=10,
                        initial_step_size=-0.001,
                        maximalSteps=1500,
                               )
            opt.optimize()
            # dump every 1000th timestep, dt=2e-15 s, cooling rate=5e12 K/s
            # t = t_high - (i+1) * 1000 * 2.e-15 * cooling_rate
            ts.append(ts_all[i])
            es.append(ats.get_potential_energy())
            # print(i, ts[-1], e0s[-1], es[-1], np.max(np.abs(ats.get_forces())))

    if do_plot:
        plt.plot(ts, e0s)
        plt.plot(ts, es)
        plt.show()

    ts = np.array(ts)
    es = np.array(es)
    tg, _, _ = find_Tg(ts, es)
    return tg, ts, es, e0s


if __name__ == '__main__':
    main()
