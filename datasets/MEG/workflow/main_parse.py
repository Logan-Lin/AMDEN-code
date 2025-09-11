import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from os import path
import json
from lmpio import read_lammpstrj
from ase.io import write, read
from ase.data import atomic_masses, atomic_numbers
from ase.units import kg, m


def parse(dir, n=9240):
    systems = json.load(open(path.join(dir, 'systems.json'), 'r'))
    samples = {}

    for ii in range(n):
        # check if directory exists
        if not path.exists(path.join(dir, f'{ii:05d}')):
            continue

        if not path.exists(path.join(dir, f'{ii:05d}/sample.data')):
            print(f'No sample file for {ii}')
            continue
        else:
            sample = read(path.join(dir, f'{ii:05d}/sample.data'), format='lammps-data')
            sample.wrap()
            props = {}


        if not path.exists(path.join(dir, f'{ii:05d}/thermal.txt')):
            print(f'No thermal file for {ii}')
            continue
        else:
            lines = open(path.join(dir, f'{ii:05d}/evaporation.txt'), 'r').readlines()
            te = float(lines[0].split()[1])
            xe = float(lines[1].split()[1])

            lines = open(path.join(dir, f'{ii:05d}/thermal.txt'), 'r').readlines()
            alpha = float(lines[0].split()[1])
            cp_jkgk = float(lines[1].split()[1])
            cp_jmolk = float(lines[2].split()[1])
            tg = float(lines[3].split()[1])
            tm = float(lines[4].split()[1])
            props |= {
                'alpha': alpha,
                'Cp [J/kg/K]': cp_jkgk,
                'Cp [J/Mol/K]': cp_jmolk,
                'Tg [K]': tg,
                'Tm [K]': tm,
                'Tb [K]': te,
                'xB': xe
            }

        if not path.exists(path.join(dir, f'{ii:05d}/elastic.txt')):
            print(f'No elastic file for {ii}')
            continue
        else:
            lines = open(path.join(dir, f'{ii:05d}/elastic.txt'), 'r').readlines()
            if len(lines) < 9:
                print(f'Empty elastic file for {ii}')
                continue
            else:
                cij = [[float(x) for line in lines[:6] for x in line.split()]]
                c11, dc11 = (float(x) for x in lines[6].split()[1:3])
                c12, dc12 = (float(x) for x in lines[7].split()[1:3])
                c44, dc44 = (float(x) for x in lines[8].split()[1:3])
                E = (c11-c12) * (c11 + 2 * c12) / (c11 + c12)  # Young's modulus
                K = (c11 + 2 * c12) / 3  # Bulk modulus
                props |= {
                    'Cij [GPa]': cij,
                    'C11 [GPa]': c11,
                    'C12 [GPa]': c12,
                    'C44 [GPa]': c44,
                    'C11 error': dc11,
                    'C12 error': dc12,
                    'C44 error': dc44,
                    'E [GPa]': E,
                    'G [GPa]': c44,  # Shear modulus
                    'K [GPa]': K,
                    'nu': E / (2 * c44) - 1  # Poissons ratio
                }

        samples[ii] = sample
        systems[ii]['properties'] = props

    return samples, systems






def plot_property(data_a, data_b, formulas, title, ylabel, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    x = np.arange(len(formulas))
    
    plt.plot(data_a, marker='o', label='Run A')
    plt.plot(data_b, marker='o', linestyle='--', label='Run B')
    plt.xticks(x, formulas, rotation=-90, ha='left')
    plt.xlabel('Chemical Formula')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_correlation(data_a, data_b, title, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    r, _ = stats.pearsonr(data_a, data_b)
    
    plt.scatter(data_a, data_b)
    plt.xlabel('Run A')
    plt.ylabel('Run B')
    plt.title(f'{title}\nCorrelation (r = {r:.3f})')
    
    min_val = min(min(data_a), min(data_b))
    max_val = max(max(data_a), max(data_b))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_elastic_constants(systems_a, systems_b, idx, formulas):
    constants = ['C11', 'C12', 'C44']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(formulas))
    
    for const in constants:
        vals_a = [systems_a[j]['properties'][f'{const} [GPa]'] for j in idx]
        vals_b = [systems_b[j]['properties'][f'{const} [GPa]'] for j in idx]
        
        ax.plot(x, vals_a, marker='o', label=f'{const} Run A')
        ax.plot(x, vals_b, marker='o', linestyle='--', label=f'{const} Run B')
    
    ax.set_xticks(x)
    ax.set_xticklabels(formulas, rotation=-90, ha='left')
    ax.set_xlabel('Chemical Formula')
    ax.set_ylabel('Elastic Constant (GPa)')
    ax.set_title('Elastic Constants Comparison')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    samples, systems = parse('./data')
    properties = []

    elements = set()
    for i in samples:
        elements |= set(samples[i].get_chemical_symbols())
    print(elements)

    for i in samples:
        el_ratio = {}
        for el in elements:
            el_ratio[el] = samples[i].get_chemical_symbols().count(el) / len(samples[i])
        props = systems[i]['properties']
        props['id'] = i
        mass = sum([atomic_masses[atomic_numbers[s]] for s in samples[i].get_chemical_symbols()])
        props['density [g/cm^3]'] = mass / samples[i].get_volume() / kg * m**3 * 1000 / 100**3

        for el in elements:
            props[f'{el} ratio'] = el_ratio[el]

        properties.append(props)
    write('samples.extxyz', samples.values())
    json.dump(properties, open('properties.json', 'w'), indent=4)
    print("N samples: ", len(properties))


