from ase.io import read, write
import numpy as np
from ase.geometry.analysis import Analysis
import matplotlib.pyplot as plt
from ase.neighborlist import NeighborList
from sklearn.neighbors import KernelDensity
from ase.symbols import symbols2numbers
from numba import jit
from multiprocessing import Pool
from itertools import repeat


def rdf(ats, rmax=5.0, nbins=100, elements=None):
    if elements is not None:
        if type(elements) is str:
            elements = symbols2numbers([elements])[0]
            el_ats = [at for at in ats if elements in at.get_atomic_numbers()]
        elif type(elements[0]) is str:
            elements = symbols2numbers(elements)
            el_ats = [at for at in ats if all([el in at.get_atomic_numbers() for el in elements])]
    else:
        el_ats = ats
    print(len(el_ats))
    ana = Analysis(el_ats)

    rdfs = ana.get_rdf(rmax=rmax, nbins=nbins, return_dists=True, elements=elements)
    df = np.mean([x[0] for x in rdfs], axis=0)
    r = rdfs[0][1]
    return r, df
    # plt.plot(dist, rdf, label=label)

def partial_rdf(ats, el, rmax=5.0, nbins=100):
    rs = np.linspace(0, rmax, nbins)
    df = np.zeros(nbins)
    n_ats = 0
    for at in ats:
        nl = NeighborList([rmax / 2.0] * len(at), skin=0.0, self_interaction=False, bothways=True)
        nl.update(at)
        els = at.get_chemical_symbols()
        for i in range(len(at)):
            indices, offsets = nl.get_neighbors(i)
            if els[i] == el:
                n_ats += 1
                for i_neigh, offs in zip(indices, offsets):
                    r = np.linalg.norm(at.positions[i] - at.positions[i_neigh] - offs @ at.get_cell())
                    assert r < rmax
                    i_bin = int(r / rmax * nbins)
                    df[i_bin] += 1
    return rs, df / n_ats

def rdf_cnt(ats, rmax, el=None, bins=None):
    rs = []
    if bins is not None:
        bin_cts, _ = np.histogram(rs, bins=bins, density=False) 

    nat = 0
    for at in ats:
        nl = NeighborList([rmax / 2.0] * len(at), skin=0.0, self_interaction=False, bothways=True)
        nl.update(at)
        els = at.get_chemical_symbols()
        for i in range(len(at)):
            if els[i] == el or el is None:
                nat += 1
                indices, offsets = nl.get_neighbors(i)
                for i_neigh, offs in zip(indices, offsets):
                    r = np.linalg.norm(at.positions[i] - at.positions[i_neigh] - offs @ at.get_cell())
                    assert r < rmax
                    rs.append(r)
        if bins is not None:
            tmp_cts, _ = np.histogram(rs, bins=bins, density=False)
            bin_cts += tmp_cts
            rs = []

    if bins is None:
        return rs, nat
    else:
        return bin_cts, nat




def coordination(ats, rmax, elements=None):
    if elements is not None:
        if type(elements[0]) is str:
            elements = symbols2numbers(elements)
    counts = []

    for at in ats:
        nl = NeighborList([rmax / 2.0] * len(at), skin=0.0, self_interaction=False, bothways=True)
        nl.update(at)
        nums = at.get_atomic_numbers()
        for i in range(len(at)):
            indices, offsets = nl.get_neighbors(i)
            if elements is not None:
                if nums[i] == elements[0]:
                    if len(elements) == 1:
                        counts.append(len(indices))
                    else:
                        counts.append(len([idx for idx in indices if nums[idx]==elements[1]]))
            else:
                counts.append(len(indices))

    return counts


def angle_df(ats, rmax, elements=None):
    if elements is not None:
        if type(elements[0]) is str:
            elements = symbols2numbers(elements)
    with Pool() as pool:
        angles = pool.starmap(ats_to_angs, zip(ats, repeat(rmax), repeat(elements)))
    angles = [ang for at_angs in angles for ang in at_angs]
    return np.array(angles)

def ats_to_angs(at, rmax, elements):
    angles = []
    nums = at.get_atomic_numbers()
    nl = NeighborList([rmax / 2.0] * len(at), skin=0.0, self_interaction=False, bothways=True)
    nl.update(at)
    pos = at.get_positions()
    lat = np.array(at.get_cell())        
    for i_at in range(len(at)):
        indices, offsets = nl.get_neighbors(i_at)
        angles.extend(get_angles_opt(i_at, pos, lat, nums, indices, offsets, elements))
    return angles

# small perfomance gain, most compute is spent in neighborlist
@jit
def get_angles_opt(i_at, pos, lat, nums, indices, offsets, elements):
    pos_at = pos[i_at, :]
    angles = []
    for i_nei in range(len(indices)):
        pos_i = pos[indices[i_nei], :] + (1. * offsets[i_nei, :]) @ lat
        for j_nei in range(i_nei):
            if elements is None or (
                elements[0] == nums[i_at] and
                elements[1] == nums[indices[i_nei]] and
                elements[2] == nums[indices[j_nei]]):

                pos_j = pos[indices[j_nei], :] + (1. * offsets[j_nei, :]) @ lat
                v_i = pos_i - pos_at + 1.e-8
                v_j = pos_j - pos_at + 1.e-8
                angles.append(np.arccos(v_i.dot(v_j) / np.linalg.norm(v_i) / np.linalg.norm(v_j)))
    return angles


def el_ratio(ats, el):
    if type(el) is str:
        el = symbols2numbers([el])[0]
    n_tot = 0
    n_el = 0
    for at in ats:
        nums = at.get_atomic_numbers()
        n_tot += len(nums)
        n_el += len([n for n in nums if n==el])
    return n_el / n_tot


def read_ats(fname):
    if type(fname) is list:
        ats = []
        for f in fname:
            a = read(f, index=':')
            ats.extend(a)
    else:
        ats = read(fname, index=':')

    return ats

