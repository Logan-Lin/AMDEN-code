import enum
from lib.rdf import *
import json
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.calculators.kim.kim import KIM
from sqnm.vcsqnm_for_ase import aseOptimizer
from os.path import exists
from multiprocessing import Pool, cpu_count
from scipy.stats import gaussian_kde
import json

        
    
N_DATA = None
SAVE_FIGS = False

blue = '#381d8a'
red = '#cf4229'
green = '#28a669'

def main():
    idx = ':' if N_DATA is None else f':{N_DATA}'
    data = {
        'melt': read(f'../data/Si-melt.extxyz', index=idx),
        'quench': read(f'../data/Si-quench.extxyz', index=idx),
        'anneal': read(f'../data/Si-anneal.extxyz', index=idx),
    }
    print('DONE reading data')


    labels = {
        lbl: lbl for lbl in data
    }

    colors = {
        'anneal': blue,
        'quench': red,
        'melt': green,
    }

    def mk_plots(data, is_opt):
        energies(data, labels, colors, is_opt)
        adfs(data, 2.7, labels, colors, is_opt)
        rdfs(data, 7., labels, colors, is_opt)

    print('Before Optimization')
    mk_plots(data, False)
    print('Optimizing all structures')
    data_opt = {
        lbl: optim_all(data[lbl]) for lbl in data
    }
    print()
    print('After Optimization')
    mk_plots(data_opt, True)


def filter(data, e_max):
    for d in data:
        structures = data[d]
        n = len(structures)
        attach_calc(structures)
        data[d] = [s for s in data[d] if s.get_potential_energy() < e_max]
        n_f = len(data[d])
        print(f'Filtered {d}: {n} -> {n_f}')
        detach_calc(structures)


def optim_all(structs, n_cpu=None):
    if n_cpu is None:
        n_cpu = cpu_count()
    pool = Pool(n_cpu)
    opt_structs = pool.map(opt, structs)
    pool.close()
    print()
    return opt_structs

def opt(s):
    # install with kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006
    attach_calc([s])
    opt = aseOptimizer(s, 
                vc_relax=True, 
                force_tol=5.e-4,
                nhist_max=20,
                   )
    opt.optimize()
    s.calc = None
    print('*', end='')
    return s

# TODO: Use new, bin free, implementation
def rdfs(data, rc, labels, colors, is_opt=False):
    rs = np.linspace(0, rc, 1000)
    nbins = 300
    bins = np.linspace(0, rc, nbins+1)
    rdfs = {}
    gs = {}

    with Pool() as pool:
        rdfs = pool.starmap(rdf_cnt, zip(data.values(), repeat(rc+0.5), repeat('Si'), repeat(bins)))
        rdfs = {k: v for k, v in zip(data.keys(), rdfs)}

    ys = {}
    for n in data:
        rdff, nat = rdfs[n]
        ncts = np.sum(rdff)
        rdff = rdff / ncts / (bins[1:] - bins[:-1])
        rs = (bins[1:] + bins[:-1]) / 2
        phi = np.sum([len(s) for s in data[n]]) / np.sum([s.cell.volume for s in data[n]])
        norm = np.pi * 4 * rs**2 * nat / ncts * phi + 1.e-10 
        rdff /= norm

        linestyle = '-' if 'ref' in n else '--'
        plt.plot(rs, rdff, label=labels[n], color=colors[n], linestyle=linestyle, zorder=1)
        gs[n] = rdff
        ys[n] = rdff

    if is_opt:
        plt.ylim([-0.4, 7.])
    else:
        plt.ylim([-0.4, 5.])
    plt.xlim([1.5, rc])
    plt.xlabel('Radius [Å]')
    plt.ylabel('$g(r)$')
    plt.legend()
    plt.tight_layout()
    do_fig(is_opt, 'rdf')

    qs = np.linspace(1., 20., 1000)
    dr = rs[1] - rs[0]
    s = np.zeros_like(qs)

    # https://gist.github.com/by256/b747e0bb9693c913249e83d30ace9dc2
    # https://en.wikipedia.org/wiki/Radial_distribution_function
    ys = {}
    for n in data:
        rdff = gs[n]
        phi = np.sum([len(s) for s in data[n]]) / np.sum([s.cell.volume for s in data[n]])
        h = rdff - 1.
        qr = np.outer(qs, rs)
        s = 1. + phi * 4. * np.pi / qs * dr * np.sum(rs[None, :] * h[None, :] * np.sin(qr), axis=1)
        print(np.mean(rs**2 * h))
        # s = 1 + phi * s 
        linestyle = '-' if 'ref' in n else '--'
        plt.plot(qs, s, label=labels[n], color=colors[n], linestyle=linestyle, zorder=1)
        ys[n] = s
    plt.xlabel('Q [Å$^{-1}$]')
    plt.ylabel('$S(Q)$')
    plt.ylim([-0.8, 2.0])
    plt.legend()
    plt.tight_layout()
    do_fig(is_opt, 'S')



def adfs(data, rc, labels, colors, is_opt=False):
    nbins = 300
    bins = np.linspace(0, 180, nbins+1)
    bin_pos = (bins[1:] + bins[:-1]) / 2
    ys = {}
    for n in data:
        structures = data[n]
        angs = angle_df(structures, rc) / np.pi * 180.0 
        dens, _ = np.histogram(angs, bins=bins, density=True)
        ls = '-' if 'ref' in n else '--'
        plt.plot(bin_pos, dens, label=labels[n], color=colors[n], linestyle=ls)
        ys[n] = dens
    plt.xlim([0, 180])
    plt.ylim([-0.0015, 0.045])
    plt.xlabel('Bond Angle [°]')
    plt.legend()
    plt.tight_layout()
    do_fig(is_opt, 'adf')
    


def attach_calc(strs):
    calculator = KIM("SW_StillingerWeber_1985_Si__MO_405512056662_006",
                     options={'ase_neigh': False,
                              'release_GIL': True
                              })
    for s in strs:
        s.calc = calculator

def detach_calc(strs):
    for s in strs:
        s.calc = None


def energies(data, labels, colors, is_opt=False):
    energies = {}
    for n in data:
        structures = data[n]
        attach_calc(structures)
        es = [s.get_potential_energy() for s in structures]
        detach_calc(structures)
        energies[n] = es

    e_min = np.min([np.min(energies[n]) for n in energies])
    e_max = np.max([np.max(energies[n]) for n in energies])
    e_range = e_max - e_min

    es = np.linspace(e_min - 0.2 * e_range, e_max + 0.2 * e_range, 1000)

    ys = {}
    for n in data:
        kde = gaussian_kde(energies[n])
        linestyle = '-' if 'ref' in n else '--'
        plt.plot(es, kde(es), label=labels[n], color=colors[n], linestyle=linestyle, zorder=1)
        ys[n] = kde(es)
        plt.fill_between(es, 0, kde(es), alpha=0.2, color=colors[n], zorder=0)

    plt.plot(es, np.zeros_like(es), color='dimgray', linestyle='-', zorder=2)

    if is_opt:
        plt.ylim([-0.02, 0.6])
    else:
        plt.ylim([-0.02, 0.4])
    plt.xlabel('Energy [eV]')
    plt.ylabel('Sample Distribution')
    plt.legend()
    plt.tight_layout()
    do_fig(is_opt, 'energy')


def do_fig(is_opt, name):
    if SAVE_FIGS:
        optstr = 'opt-' if is_opt else ''
        figname = f'plots/{figprefix}{optstr}{name}.pdf'
        print(f'Saving figure: {figname}')
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()



if __name__ == '__main__':
    plt.rcParams.update({'font.size': 20}) 
    main()




