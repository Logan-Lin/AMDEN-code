from ase.io import read, write
# from vitrum.utility import get_random_packed
from bmp import element_map, inv_element_map, charges, formal_charges, BMPcalc
import numpy as np
from ase.data import covalent_radii
from ase.symbols import symbols2numbers
from ase import Atoms
from ase.neighborlist import NeighborList
from math import comb
from itertools import combinations
import json
import matplotlib.pyplot as plt
from lmpio import *
from os import makedirs
from sys import argv
from sqnm.vcsqnm_for_ase import aseOptimizer





def generate_compositions():
    glass_formers = ['Si', 'P', 'Al']
    modifiers = ['Li', 'Be', 'K', 'Ca', 'Ti', 'Ba', 'Zn']
 
    target_nat = 800
    max_modifiers = 4
    max_modifier_ratio = 0.4
    n_modifier_steps = 4 #5
    n_glass_former_steps = 6 #6

    structures = []

    systems = []
    sys_id = 0
    ratios = []
    target_ratios = []
    nats = []
    
    for n_gf in range(1, len(glass_formers)+1): #
        for gf_comb in combinations(glass_formers, n_gf):
            for gf_fracs in get_all_fractions(n_gf, n_glass_former_steps, sum_to_one=True):
                for n_modifiers in range(max_modifiers + 1):
                    for mod_comb in combinations(modifiers, n_modifiers):
                        for mod_fracs in get_all_fractions(n_modifiers, n_modifier_steps) * max_modifier_ratio:
                            target_modifier_ratios = {el: r for el, r in zip(mod_comb, mod_fracs)}
                            target_glass_former_ratios = {el: r * (1-sum(mod_fracs)) for el, r in zip(gf_comb, gf_fracs)}
                            systems.append({
                                'id': sys_id,
                                'target_num_atoms': target_nat,
                                'target_modifier_ratios': target_modifier_ratios,
                                'target_glass_former_ratios': target_glass_former_ratios
                            })
                            print(json.dumps(systems[-1], indent=2))
                            sys_id += 1
                                   
    with open('data/systems.json', 'w') as f:
        json.dump(systems, f, indent=2)
            
    return

def generate_structure(id):
    systems = json.load(open('data/systems.json'))
    s = systems[id]
    gfs = [(el, r) for el, r in s['target_glass_former_ratios'].items()]
    mods = [(el, r) for el, r in s['target_modifier_ratios'].items()]
    target_nat = s['target_num_atoms']
    comp = optimize_atom_composition(gfs + mods, target_nat, tryrange=5, l_atomsum=1.e-5, l_elratio=1.)
    ats = pack_cell(comp)
    s['symbols'] = str(ats.symbols)
    s['num_atoms'] = len(ats)
    s['composition'] = comp
    n = sum(comp[el] for el in comp if el != 'O')
    s['modifier_ratios'] = {el: comp[el] / n for el, x in mods}
    s['glass_former_ratios'] = {el: comp[el] / n for el, x in gfs}

    try:
        makedirs(f'data/{id:05d}')
    except:
        print('WARNING: Dir already exists')

    write(f'data/{id:05d}/pre-optim.extxyz', ats)
    ats.calc = BMPcalc()

    for i in range(10):
        # SD preopt
        ss = 0.001
        f = ats.get_forces()
        while np.linalg.norm(f) > 6.0:
            last_f = f
            f = ats.get_forces()
            if np.sum(f * last_f) < 0.:
                ss = ss * 0.5
            else:
                ss = ss * 1.1

            step = f * ss 
            ss = max(ss, 0.00001)
            # limit max step to 1.0
            step = step / max(1.0, np.linalg.norm(step))
            ats.set_positions(ats.get_positions() + ss * f)
            ats.wrap()
            # print('SD', ats.get_potential_energy(), ss, np.linalg.norm(step), np.linalg.norm(f))

        # optimize cell volume
        pressure = -np.trace(ats.get_stress(False)) / 3
        ssl = 0.01
        while abs(pressure) > 0.005:
            # print(pressure, ats.get_potential_energy(), ats.get_volume())
            last_p = pressure
            ats.set_cell(ats.get_cell(True) * (1. + ssl * np.sign(pressure)), scale_atoms=True)
            ats.wrap()
            pressure = -np.trace(ats.get_stress(False))
            if pressure * last_p < 0:
                ssl = ssl * 0.5
            else:
                ssl = ssl * 1.1
            # print(pressure)
        # write(f'data/{id:05d}/optim-{i:05d}.extxyz', ats)


    # set log level to info
    # import logging
    # logging.basicConfig(level=logging.INFO)

    opt = aseOptimizer(ats, 
                vc_relax=True, 
                force_tol=0.1,
                nhist_max=10,
                # initial_step_size=0.0001, # TODO: getting crashes because atoms get too close. Maybe pre-relax or use max stepsize?
                initial_step_size=ss * 0.1,
                maximalSteps=2000,
            )
    # for _ in range(100):
    #     prev_pos, prev_lat = ats.get_positions(), ats.get_cell(True)
    #     opt.step(ats)
    #     print(np.linalg.norm(prev_pos - ats.get_positions()))
    #     print(np.linalg.norm(prev_lat - ats.get_cell()))
    #     print('QN', ats.get_potential_energy())

    opt.optimize()
    ats.wrap()
    # write(f'data/init/{id:05d}/mid-optim.extxyz', ats)
    # write_lammpsdata(f'data/init/{id:05d}/mid-optim.data', ats)
    # opt = aseOptimizer(ats, 
    #             vc_relax=True, 
    #             force_tol=0.01,
    #             nhist_max=20,
    #             initial_step_size=0.01,
    #         )
    # opt.optimize()
    # ats.wrap()

    write(f'data/{id:05d}/init.extxyz', ats)
    write_lammpsdata(f'data/{id:05d}/init.data', ats)
    json.dump(s, open(f'data/{id:05d}/system.json', 'w'), indent=2)


# compute fractions of elements to add, are all non zero
def get_all_fractions(n, n_steps, sum_to_one=False):
    if n==0:
        return np.array([[]])
    fracs = []
    idx = [1] * n
    while sum(idx) <= n_steps:
        fracs.append(idx.copy())
        for i in range(n-1, -1, -1):
            if (idx[i] >= n_steps or sum(idx) >= n_steps) and i>0:
                idx[i] = 1
            else:
                idx[i] += 1
                break
    if sum_to_one:
        fracs = [x for x in fracs if sum(x) == n_steps]
    return np.array(fracs) / n_steps




def optimize_atom_composition(elements, n_target, l_atomsum=1., l_elratio=1.e3, tryrange=6):
    # Extract element names, target ratios, and charges from input
    element_names = [x[0] for x in elements]
    element_ids = [inv_element_map[el] for el in element_names]
    target_ratios = [x[1] for x in elements]
    charges = [formal_charges[id] for id in element_ids]
    o_charge = formal_charges[inv_element_map['O']]
    

    # make a reasonble initial guess
    # how much o for '1' of the other elements
    o_ratio = sum(-1. * r * q / o_charge for r, q in zip(target_ratios, charges))

    n_o = 0
    el_count = []
    for i, el in enumerate(element_names):
        c = np.lcm(charges[i], o_charge)
        n = int(np.round(n_target * target_ratios[i] / (1. + o_ratio) / c * charges[i]))
        ac = n * c // charges[i]
        el_count.append(ac)
        n_o = n_o - n * c // o_charge
    o_count = n_o

    def loss(el_count, o_count):
        not_o_count = sum(el_count)
        nat_loss = l_atomsum * (not_o_count + o_count - n_target)**2
        rat_loss = l_elratio * sum((n/not_o_count / r - 1.0)**2 for n, r in zip(el_count, target_ratios))
        return nat_loss + rat_loss

    def total_charge(el_count, o_count):
        return sum(n * c for n, c in zip(el_count, charges)) + o_count * o_charge

    # start with initial guess
    l_best = loss(el_count, o_count)
    el_count_init = el_count.copy()
    el_count_best = el_count.copy()
    o_count_best = o_count


    # let all el_counts go from el_count_init - tryrange to el_count_init + tryrange
    el_count = [max(0, x - tryrange) for x in el_count_init]
    while any([x < (xi + tryrange) for x, xi in zip(el_count, el_count_init)]):
        # calculate number of O to compensate charge
        o_count = int(-1 * np.round(total_charge(el_count, 0) / o_charge))
        if total_charge(el_count, o_count) == 0:  # valid solution
            l = loss(el_count, o_count)
            if l < l_best:
                el_count_best = el_count.copy()
                o_count_best = o_count
                l_best = l
        for i in range(len(el_count)):  # count
            if el_count[i] < el_count_init[i] + tryrange:
                el_count[i] += 1
                break
            elif el_count[i] == el_count_init[i] + tryrange:
                el_count[i] = max(0, el_count_init[i] - tryrange)


    return {el: int(c) for el, c in zip(element_names, el_count_best)} | {'O': int(o_count_best)}


def pack_cell(elements, v_factor=4.0):
    nat = sum(elements.values())
    pos = np.random.rand(nat, 3)
    els = np.array([symbols2numbers(el) for el in elements for _ in range(elements[el])]).flatten()
    radii = covalent_radii[els]
    v = np.sum(4. / 3. * np.pi * radii**3) * v_factor
    lat = np.eye(3) * v**(1./3.)
    pos = pos @ lat

    ats = Atoms(els, cell=lat, pbc=True, positions=pos)
    skin = 0.0
    nl = NeighborList(radii, self_interaction=False, bothways=True, skin=skin)

    for it in range(1000):
        nl.update(ats)
        pos = ats.get_positions()
        dx = np.zeros(pos.shape)
        dsum = 0
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            rs = pos[indices, :] + offsets @ lat - pos[i, :]
            ds = np.linalg.norm(rs, axis=1) - (radii[indices] + radii[i])
            if np.any(ds > 0):
                print('Assertion failed: ds <= 0')
                quit()
            dsum += np.sum(ds) #np.minimum(np.sum(ds), 0.)
            ds -= skin
            dx[i,:] = np.sum(rs / np.linalg.norm(rs, axis=1)[:, None] * ds[:, None], axis=0)
            
        # print(it, dsum, np.linalg.norm(dx))
        ats.set_positions(pos + dx)
        if dsum >= -1.e-5:
            break
        # print(i, ds)
    else: 
        print('Not converged')
        quit()

    ats.wrap()

    # print('----')
    return ats

def retry_gen(id, n=0):
    try:
        generate_structure(id)
    except:
        print(f'WARN: failed optimizing ... retrying {n}')
        if n < 10:
            retry_gen(id, n+1)
        else:
            print(f'ERROR: failed optimizing')
            exit(1)
    
if __name__ == '__main__':
    if len(argv) > 1:
        retry_gen(int(argv[1]))
    else:
        generate_compositions()
        
