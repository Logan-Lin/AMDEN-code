import numpy as np
import matplotlib.pyplot as plt
from lmpio import read_lammpstrj 
from ase.io import write, read
from sys import argv
from ase.units import J, mol, kg
from sys import argv
from scipy.optimize import curve_fit
from ase.units import kB, bar 

do_plot = False

def main():
    if len(argv) < 2:
        id = None
        f_evap = 'evap.txt'
        f_sample = 'evap.lammpstrj'
        do_plot = True
    else:
        id = int(argv[1])
        f_evap = f'data/{id:05d}/evap.txt'
        f_sample = f'data/{id:05d}/init.extxyz'
        do_plot = False

    ts = []
    vs = []
    ps = []

    with open(f_evap, 'r') as f:
        f.readline()
        f.readline()
        for l in f:
            s = l.split()
            ts.append(float(s[1]))
            vs.append(float(s[2]))
            # es.append(float(s[3]))
            ps.append(float(s[3]))
    ats = read(f_sample, index=-1)
    nat = len(ats)
    v = vs[0]
    ts = np.array(ts)
    vs = np.array(vs)

    if do_plot:
        plt.scatter(ts, ps, s=1)
        plt.show()


    t_min = np.min(ts)
    t_max = np.max(ts)
    
    # VdW quation of state: dp/dT = n * kB / (v-b)
    # -> x = n_effective / n * v / (v-b) -> should be between 0 and 1
    p_t = lambda t, te, x: np.where(t < te, 0., x * nat * kB / v * (t-te) / bar)
    tes = np.linspace(t_min+50, t_max-200, 1000)
    cs = []
    for te in tes:
        f = lambda t, x: p_t(t, te, x)
        p0 = [1.]
        bounds = ([0.], [2.])
        popt, _ = curve_fit(f, ts, ps, p0=p0, bounds=bounds)
        c = np.mean((ps - f(ts, *popt))**2)
        cs.append(c)
    if do_plot:
        plt.plot(tes, cs)
        plt.show()

    te = tes[np.argmin(cs)]
    f = lambda t, x: p_t(t, te, x)
    p0 = [1.]
    bounds = ([0.], [2.])
    popt, _ = curve_fit(f, ts, ps, p0=[1.], bounds=bounds)
    x = popt[0]
    f = lambda t: p_t(t, te, *popt)
    if do_plot:
        ts_sorted = np.sort(ts)
        plt.scatter(ts, ps, s=1)
        plt.axvline(te, color='red')
        plt.plot(ts_sorted, f(ts_sorted), color='red')
        plt.xlabel('T')
        plt.ylabel('P')
        plt.show()

    print(te * 0.75)

    if id is None:
        return

    with open(f'data/{id:05d}/evaporation.txt', 'w') as f:
        f.write(f'Te_K {te}\n')
        f.write(f'x {x}\n')

if __name__ == '__main__':
    main()
