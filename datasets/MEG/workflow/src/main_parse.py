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
            # s = read_lammpstrj(path.join(dir, f'mq/{ii:05d}/sample.lammpstrj'))[-1]
            # read sample.data
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

        # print(systems[ii]['properties'])
        samples[ii] = sample
        systems[ii]['properties'] = props

    return samples, systems




def plot_property(data_a, data_b, formulas, title, ylabel, figsize=(10, 6)):
    return
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

def plot_correlation(data_a, data_b, title, figsize=(6, 6)):
    rmsd = np.sqrt(np.mean((np.array(data_a) - np.array(data_b))**2))
    print(f'{title} RMSD: {rmsd:.3f}')
    plt.figure(figsize=figsize)
    r, _ = stats.pearsonr(data_a, data_b)
    
    plt.scatter(data_a, data_b, label=f'Different Structure ($\\rho$ = {r:.3f})')
    plt.xlabel(title)
    # plt.title(f'{title}\nCorrelation (r = {r:.3f})')
    
    min_val = min(min(data_a), min(data_b))
    max_val = max(max(data_a), max(data_b))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_correlation_combined(data_a, data_b, data_c, lbl):
    plt.figure(figsize=(6, 6))
    plt.scatter(data_a, data_b, label=f'Different Structure ($\\rho$ = {stats.pearsonr(data_a, data_b)[0]:.3f})')
    plt.scatter(data_a, data_c, label=f'Same Structure ($\\rho$ = {stats.pearsonr(data_a, data_c)[0]:.3f})')
    min_val = min(min(data_a), min(data_b), min(data_c))
    max_val = max(max(data_a), max(data_b), max(data_c))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    plt.xlabel(lbl)
    plt.legend(loc='upper right')
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
    # larger fonts
    plt.rcParams.update({'font.size': 14})

    samples_prod, systems_prod = parse('../prod/data')

    properties = []
    elements = set()
    for i in samples_prod:
        elements |= set(samples_prod[i].get_chemical_symbols())
    print(elements)
    for i in samples_prod:
        el_ratio = {}
        for el in elements:
            el_ratio[el] = samples_prod[i].get_chemical_symbols().count(el) / len(samples_prod[i])
        # print(i, el_ratio)
        # print(systems[i]['properties'])
        # print()
        props = systems_prod[i]['properties']
        props['id'] = i
        mass = sum([atomic_masses[atomic_numbers[s]] for s in samples_prod[i].get_chemical_symbols()])
        props['density [g/cm^3]'] = mass / samples_prod[i].get_volume() / kg * m**3 * 1000 / 100**3
        for el in elements:
            props[f'{el} ratio'] = el_ratio[el]

        properties.append(props)
    write('samples.extxyz', samples_prod.values())
    json.dump(properties, open('properties.json', 'w'), indent=4)
    print("N samples: ", len(properties))




    samples_test, systems_test = parse('./data')
    samples_test_reheat, systems_test_reheat = parse('../test-reheat/data')

    idx = list(set(samples_prod.keys()) & set(samples_test.keys()))
    formulas = [samples_prod[i].symbols for i in idx]

    if True:
        # Temperature plots
        tgs_a = [systems_prod[i]['properties']['Tg [K]'] for i in idx]
        tbs_a = [systems_prod[i]['properties']['Tb [K]'] for i in idx]
        tgs_b = [systems_test[i]['properties']['Tg [K]'] for i in idx]
        tbs_b = [systems_test[i]['properties']['Tb [K]'] for i in idx]
        tmps_a = [systems_prod[i]['properties']['Tm [K]'] for i in idx]
        tmps_b = [systems_test[i]['properties']['Tm [K]'] for i in idx]
        
        plot_property(tgs_a, tgs_b, formulas, 'Glass Transition Temperature', 'Temperature (K)')
        plot_correlation(tgs_a, tgs_b, 'Glass Transition Temperature [K]')
        plot_property(tbs_a, tbs_b, formulas, 'Evaporation Temperature', 'Temperature (K)')
        plot_correlation(tbs_a, tbs_b, 'Evaporation Temperature [K]')
        plot_property(tmps_a, tmps_b, formulas, 'Melting Temperature', 'Temperature (K)')
        plot_correlation(tmps_a, tmps_b, 'Melting Temperature [K]')

        # # xB plot
        # xbs_a = [systems_a[i]['properties']['xB'] for i in idx]
        # xbs_b = [systems_b[i]['properties']['xB'] for i in idx]
        # plot_property(xbs_a, xbs_b, formulas, 'Boron Fraction', 'xB')
        # plot_correlation(xbs_a, xbs_b, 'Boron Fraction')

        # Combined elastic constants plot
        plot_elastic_constants(systems_prod, systems_test, idx, formulas)
        
        # Elastic constants correlations
        for const in ['C11', 'C12', 'C44']:
            vals_a = [systems_prod[i]['properties'][f'{const} [GPa]'] for i in idx]
            vals_b = [systems_test[i]['properties'][f'{const} [GPa]'] for i in idx]
            latex_const = '$' + const.replace('C', 'C_{') + '}$'
            plot_correlation(vals_a, vals_b, f'{latex_const} [GPa]')

        # [remaining properties]
        alphas_a = [systems_prod[i]['properties']['alpha'] for i in idx]
        alphas_b = [systems_test[i]['properties']['alpha'] for i in idx]
        plot_property(alphas_a, alphas_b, formulas, 'Thermal Expansion Coefficient', 'α')
        plot_correlation(alphas_a, alphas_b, 'Thermal Expansion Coefficient')

        cp_jkgks_a = [systems_prod[i]['properties']['Cp [J/kg/K]'] for i in idx]
        cp_jkgks_b = [systems_test[i]['properties']['Cp [J/kg/K]'] for i in idx]
        plot_property(cp_jkgks_a, cp_jkgks_b, formulas, 'Specific Heat Capacity', 'Cp (J/kg/K)')
        plot_correlation(cp_jkgks_a, cp_jkgks_b, 'Specific Heat Capacity [J/kg/K]')

        cp_jmolks_a = [systems_prod[i]['properties']['Cp [J/Mol/K]'] for i in idx]
        cp_jmolks_b = [systems_test[i]['properties']['Cp [J/Mol/K]'] for i in idx]
        plot_property(cp_jmolks_a, cp_jmolks_b, formulas, 'Molar Heat Capacity', 'Cp (J/mol/K)')
        plot_correlation(cp_jmolks_a, cp_jmolks_b, 'Heat Capacity [J/mol/K]')



    # now compare thermal properties between test and test-reheat
    print("Comparing test and test-reheat")
    idx = list(set(samples_test.keys()) & set(samples_test_reheat.keys()) & set(samples_prod.keys()))
    formulas = [samples_test[i].symbols for i in idx]

    alphas_a = [systems_prod[i]['properties']['alpha'] for i in idx]
    alphas_b = [systems_test[i]['properties']['alpha'] for i in idx]
    alphas_c = [systems_test_reheat[i]['properties']['alpha'] for i in idx]
    plot_property(alphas_b, alphas_c, formulas, 'Thermal Expansion Coefficient', 'α')
    plot_correlation(alphas_b, alphas_c, 'Thermal Expansion Coefficient')
    plot_correlation_combined(alphas_b, alphas_a, alphas_c, 'Thermal Expansion Coefficient')

    cp_jkgks_a = [systems_prod[i]['properties']['Cp [J/kg/K]'] for i in idx]
    cp_jkgks_b = [systems_test[i]['properties']['Cp [J/kg/K]'] for i in idx]
    cp_jkgks_c = [systems_test_reheat[i]['properties']['Cp [J/kg/K]'] for i in idx]
    plot_property(cp_jkgks_b, cp_jkgks_c, formulas, 'Specific Heat Capacity', 'Cp (J/kg/K)')
    plot_correlation(cp_jkgks_b, cp_jkgks_c, 'Specific Heat Capacity')
    plot_correlation_combined(cp_jkgks_b, cp_jkgks_a, cp_jkgks_c, 'Specific Heat Capacity [J/kg/K]')

    cp_jmolks_a = [systems_prod[i]['properties']['Cp [J/Mol/K]'] for i in idx]
    cp_jmolks_b = [systems_test[i]['properties']['Cp [J/Mol/K]'] for i in idx]
    cp_jmolks_c = [systems_test_reheat[i]['properties']['Cp [J/Mol/K]'] for i in idx]
    plot_property(cp_jmolks_b, cp_jmolks_c, formulas, 'Molar Heat Capacity', 'Cp (J/mol/K)')
    plot_correlation(cp_jmolks_b, cp_jmolks_c, 'Heat Capacity [J/mol/K]')
    plot_correlation_combined(cp_jmolks_b, cp_jmolks_a, cp_jmolks_c, 'Heat Capacity [J/mol/K]')
