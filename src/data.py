import json
import os
from random import random
from collections import Counter

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import read, write as write_ase
from ase.io.lammpsdata import read_lammps_data
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

from neighborlists import Neighborlist
from utils import positions_into_cell, string2slice
from models.modules.bmp import formal_charges, inv_element_map

SAVE_ROOT_DIR = os.environ.get('SAVE_ROOT_DIR', 'cache')
save_dirs = {
    "settings": os.path.join(SAVE_ROOT_DIR, 'settings'),
    "infer": os.path.join(SAVE_ROOT_DIR, 'infer'),
    "models": os.path.join(SAVE_ROOT_DIR, 'models'),
    "log": os.path.join(SAVE_ROOT_DIR, 'log')
}


class MaterialDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing material structures.
    
    This class handles loading material structures from various sources (LAMMPS, ExtXYZ, or 
    synthetic examples) and provides preprocessing capabilities including atom type filtering,
    ghost atom addition for density control, and charge balancing.
    
    Attributes:
        samples (list[Sample]): List of Sample objects representing material structures.
    """

    def __init__(self, source, select_atom_types=None, target_density=None, index=':', save_dir=None):
        """Initialize MaterialDataset.
        
        Args:
            source (dict): Dictionary specifying data source with keys:
                - 'name' (str): Source type ('lammps', 'extxyz', '2D_example', '3D_example', '3D_empty')
                - 'params' (dict): Source-specific parameters
            select_atom_types (list[int], optional): Filter to only include specified atom types.
                Defaults to None (include all atoms).
            target_density (float, optional): Target density for adding ghost atoms (atoms/Å³).
                Defaults to None (no ghost atoms added).
            index (str, optional): Slice notation for selecting subset of samples.
                Defaults to ':' (all samples).
            save_dir (str, optional): Directory to save processed samples as ExtXYZ.
                Defaults to None (no saving).
        """
        super().__init__()
        source_name = source['name']
        if source_name == 'lammps':
            samples = self.read_lammps_dataset(**source['params'])
        elif source_name == 'extxyz':
            samples = self.read_extxyz_dataset(**source['params'])
        elif source_name == '2D_example':
            samples = self.create_example_2d_dataset(**source['params'])
        elif source_name == '3D_example':
            samples = self.create_example_3d_dataset(**source['params'])
        elif source_name == '3D_empty':
            samples = self.create_empty_3d_dataset(**source['params'])
        else:
            raise NotImplementedError(f'Unknown dataset type: {source_name}')

        samples = samples[string2slice(index)]

        if select_atom_types is not None:
            for i, sample in tqdm(enumerate(samples), desc='Filtering atom types'):
                mask = np.isin(sample.elements, select_atom_types)
                samples[i] = Sample(
                    sample.elements[mask], sample.positions[mask, :], sample.lattice, sample.pbc)

        if target_density is not None:
            samples = [self.add_ghost_atoms(sample, target_density) for sample in samples]

        self.samples = samples

        # Save samples to ExtXYZ file if save_dir is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            write_ase(os.path.join(save_dir, 'samples.extxyz'), [sample.to_ase_atoms() for sample in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def read_lammps_dataset(self, root_dir, style):
        """Read material structures from LAMMPS data files.
        
        Args:
            root_dir (str): Root directory containing subdirectories with LAMMPS files.
                Each subdirectory should contain a 'final.data' file.
            style (str): LAMMPS atom style (e.g., 'atomic', 'charge', 'full').
            
        Returns:
            list[Sample]: List of Sample objects parsed from LAMMPS files.
        """
        samples = []
        for inst_name in tqdm(os.listdir(root_dir), desc='Loading raw files'):
            file_path = os.path.join(root_dir, inst_name, 'final.data')
            if os.path.exists(file_path):
                sample = read_lammps_data(open(file_path, 'r'), style=style)
                samples.append(Sample.from_ase_atoms(sample))
        return samples

    def read_extxyz_dataset(self, file, properties_file=None, index=':'):
        """Read material structures from Extended XYZ format files.
        
        Args:
            file (str): Path to ExtXYZ file containing atomic structures.
            properties_file (str, optional): Path to JSON/YAML file with material properties
                for conditioning. Properties are matched to structures by index.
            index (str, optional): Slice notation for selecting subset of structures.
                Defaults to ':' (all structures).
            
        Returns:
            list[Sample]: List of Sample objects with attached properties.
        """
        samples = read(file, index=index)

        if properties_file is not None:
            if properties_file.endswith('.json'):
                props = json.load(open(properties_file, 'r'))
            elif properties_file.endswith('.yml') or properties_file.endswith('.yaml'):
                props = yaml.safe_load(open(properties_file, 'r'))
            else:
                raise NotImplementedError(
                    'The properties file has an unsupported format.')
            if len(props) != len(samples):
                raise Exception(
                    'Number of properties does not match number of structures')
            samples = [Sample.from_ase_atoms(
                atoms, properties=prop) for atoms, prop in zip(samples, props)]
        else:
            samples = [Sample.from_ase_atoms(atoms) for atoms in samples]
        return samples

    def create_example_2d_dataset(self, pattern, n_samples, nx, ny, noise=0., bond_length=4.):
        """Create a toy 2D dataset.
        """
        samples = []
        for i in tqdm(range(n_samples), desc='Generating 2D test data'):
            if pattern == 'grid':
                sample = MaterialDataset.make_test_structure_2d_grid(
                    nx, ny, noise, bond_length)
            else:
                raise NotImplementedError(
                    f'2D pattern {pattern} is not implemented')
            samples.append(sample)
        return samples

    def make_test_structure_2d_grid(self, nx, ny, noise, bond_length):
        x_grid = np.linspace(0., nx, nx + 1)[:-1]
        y_grid = np.linspace(0., ny, ny + 1)[:-1]
        x_pos, y_pos = np.meshgrid(x_grid, y_grid, indexing='ij')
        z_pos = torch.zeros((nx * ny,))
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        pos = np.stack((x_pos, y_pos, z_pos), axis=1) + 0.5
        lat = np.diag(np.array([nx, ny, 1.]))
        pos = pos * bond_length
        lat = lat * bond_length
        pos = pos + np.random.randn(*pos.shape) * noise
        atom_types = nx * ny * [1]
        return Sample(torch.LongTensor(atom_types), torch.FloatTensor(pos), torch.FloatTensor(lat),
                      pbc=(True, True, False))

    def create_example_3d_dataset(self, pattern, n_samples, nx, ny, nz, noise=0., bond_length=4.):
        """Create a toy 3D dataset.
        """
        samples = []
        for i in tqdm(range(n_samples), desc='Generating 3D test data'):
            if pattern == 'grid':
                sample = self.make_test_structure_3d_grid(
                    nx, ny, nz, noise, bond_length)
            else:
                raise NotImplementedError(
                    f'2D pattern {pattern} is not implemented')
            samples.append(sample)
        return samples

    def make_test_structure_3d_grid(self, nx, ny, nz, noise, bond_length):
        x_grid = np.linspace(0., nx, nx + 1)[:-1]
        y_grid = np.linspace(0., ny, ny + 1)[:-1]
        z_grid = np.linspace(0., nz, nz + 1)[:-1]
        x_pos, y_pos, z_pos = np.meshgrid(
            x_grid, y_grid, z_grid, indexing='ij')
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        z_pos = z_pos.flatten()
        pos = np.stack((x_pos, y_pos, z_pos), axis=1) + 0.5
        lat = np.diag(np.array([nx, ny, nz])) * 1.
        pos = pos * bond_length
        lat = lat * bond_length
        pos = pos + np.random.randn(*pos.shape) * noise
        atom_types = nx * ny * nz * [1]
        dx = np.random.rand(1, 3) - 0.5
        dx = dx @ lat
        pos += dx
        return Sample(torch.LongTensor(atom_types), torch.FloatTensor(pos), torch.FloatTensor(lat))

    def add_ghost_atoms(self, sample, target_density):
        """
        Add ghost atoms to a sample to reach the target density.

        Args:
            sample (Sample): The original sample.
            target_density (float): The target density to achieve.

        Returns:
            Sample: A new sample with ghost atoms added.
        """
        volume = torch.abs(torch.linalg.det(sample.lattice))
        current_num_atoms = sample.get_num_atoms()
        target_num_atoms = int(target_density * volume)
        num_ghost_atoms = max(0, target_num_atoms - current_num_atoms)

        if num_ghost_atoms == 0:
            return sample

        # Create ghost atoms
        ghost_elements = torch.zeros(num_ghost_atoms, dtype=torch.long)
        ghost_positions = torch.rand(num_ghost_atoms, 3) @ sample.lattice

        # Combine original and ghost atoms
        new_elements = torch.cat([sample.elements, ghost_elements])
        new_positions = torch.cat([sample.positions, ghost_positions])

        # Create a new sample with ghost atoms
        return Sample(
            elements=new_elements,
            positions=new_positions,
            lattice=sample.lattice,
            pbc=sample.pbc,
            properties=sample.properties
        )
    
    def write_charge_balanced(self, tgt_file):
        """
        Filter out the charge-unbalanced samples and write the charge-balanced ones to a new file.
        """
        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        
        charge_balanced_samples = [(i, s) for i, s in enumerate(self.samples) if s.is_charge_balanced()]
        write_ase(tgt_file, [s.to_ase_atoms() for _, s in charge_balanced_samples])
        idx_file = f"{os.path.splitext(tgt_file)[0]}_idx.txt"
        indices = [str(i) for i, _ in charge_balanced_samples]
        with open(idx_file, 'w') as f:
            f.write(','.join(indices))

    def make_empty_3d_sample(self, lx, ly, lz, properties=None, pbc=(True, True, True)):
        """
        Create an empty 3D sample with specified cell dimensions and an orthogonal lattice.
        
        Args:
            lx (float): Length of the cell in x direction.
            ly (float): Length of the cell in y direction.
            lz (float): Length of the cell in z direction.
            properties (dict, optional): Dictionary of properties to assign to the sample.
            pbc (tuple, optional): Periodic boundary conditions for each axis.
            
        Returns:
            Sample: An empty sample with no atoms and specified cell dimensions.
        """
        # Create empty tensors for elements and positions (0 atoms)
        elements = torch.zeros((0,), dtype=torch.long)
        positions = torch.zeros((0, 3), dtype=torch.float)
        
        # Create the lattice tensor
        lattice = torch.zeros((3, 3), dtype=torch.float)
        lattice[0, 0] = lx
        lattice[1, 1] = ly
        lattice[2, 2] = lz
        
        # Create and return the empty sample
        return Sample(
            elements=elements,
            positions=positions,
            lattice=lattice,
            pbc=pbc,
            properties=properties
        )
    
    def create_empty_3d_dataset(self, lx, ly, lz, n_samples, properties_args):
        """
        Create a dataset of empty 3D samples with specified cell dimensions.

        Args:
            lx (float): Length of the cell in x direction.
            ly (float): Length of the cell in y direction.
            lz (float): Length of the cell in z direction.
            n_samples (int): Number of samples to create.
            properties_args (list): List of dictionaries of property arguments. Each item in the list is a dictionary of the form:
                - prop_name (str): Name of the property.
                - mode (str): Mode of property generation, one of 'uniform', 'linear', 'fixed', 'null'.
                Plus additional arguments for the property generation. For both 'uniform' and 'linear' mode:
                - min_val (float): Minimum value of the property.
                - max_val (float): Maximum value of the property.
                For 'fixed' mode:
                - value (float): Value of the property.
        """
        samples = []
        for n in range(n_samples):
            properties = {}
            for prop_args in properties_args:
                if prop_args['mode'] == 'uniform':
                    # Uniform random
                    properties[prop_args['prop_name']] = torch.tensor(random() * (prop_args['max_val'] - prop_args['min_val']) + prop_args['min_val'])
                elif prop_args['mode'] == 'linear':
                    # Linear interpolation between two values
                    properties[prop_args['prop_name']] = torch.tensor(prop_args['min_val'] + n * (prop_args['max_val'] - prop_args['min_val']) / max(n_samples - 1, 1))
                elif prop_args['mode'] == 'fixed':
                    properties[prop_args['prop_name']] = torch.tensor(prop_args['value'])
                elif prop_args['mode'] == 'null':
                    properties[prop_args['prop_name']] = None

            sample = self.make_empty_3d_sample(lx, ly, lz, properties=properties)
            samples.append(sample)
        return samples

class Sample:
    """Represents a single 3D material structure with atoms and properties.
    
    Core data structure for material samples in the diffusion model. Handles atomic
    positions, elements, periodic boundary conditions, and material properties used
    for conditional generation. Supports efficient neighbor list computation and
    E(3)-equivariant operations.
    
    Attributes:
        elements (torch.LongTensor): Atomic numbers, shape (n_atoms,).
        positions (torch.FloatTensor): Atomic positions in Angstroms, shape (n_atoms, 3).
        lattice (torch.FloatTensor): Unit cell vectors, shape (3, 3).
        pbc (tuple[bool]): Periodic boundary conditions for each axis.
        neighborlist (Neighborlist): Efficient neighbor finding data structure.
        element_emb (torch.FloatTensor, optional): Learned element embeddings.
        properties (dict[str, torch.Tensor], optional): Material properties for conditioning.
    """

    def __init__(self, elements, positions, lattice, pbc=(True, True, True),
                 neighborlist=None, init_r_cut=None,
                 element_emb=None, properties=None):
        """
        Args:
            elements (torch.LongTensor): element numbers corresponding to the atoms, with shape (n_atom).
            positions (torch.FloatTensor): positions correspondng to the atoms, with shape (n_atom, 3).
            lattice (torch.FloatTensor): the unit cell size, with shape (3, 3).
            pbc (tuple, optional): periodic boundary conditions, set for each axis respectively.
            init_r_cut (float): cutoff used for initial neighbor list construction. Neighbors are pruned to r_cut, but a larger cutoff avoids frequent rebuild of the list during inference.
            element_emb (torch.FloatTensor): Embedding of the element
            properties (dict[str, torch.FloatTensor]): Dictionary containing any additional properties of the sample (can be used for conditioning the model)
        """
        # Geometry
        self.elements = elements
        self.positions = positions
        self.lattice = lattice
        self.pbc = pbc

        if neighborlist is None:
            self.neighborlist = Neighborlist(self.lattice, self.pbc, init_r_cut)
        else:
            self.neighborlist = neighborlist
            neighborlist.set_init_r_cut(init_r_cut)

        # Additional properties
        self.element_emb = element_emb
        self.properties = properties

    def to(self, device):
        return Sample(
            self.elements.to(device),
            self.positions.to(device),
            self.lattice.to(device),
            self.pbc,
            self.neighborlist.to(device),
            self.neighborlist.init_r_cut,
            None if self.element_emb is None else self.element_emb.to(device),
            None if self.properties is None else {k: self.properties[k].to(device) if self.properties.get(k, None) is not None else None for k in self.properties})

    @staticmethod
    def from_ase_atoms(atoms: Atoms, properties=None):
        """Create Sample object from ASE Atoms object.
        
        Args:
            atoms (ase.Atoms): ASE Atoms object containing atomic structure.
            properties (dict, optional): Dictionary of material properties for conditioning.
                Keys are property names, values are property tensors or None for null properties.
                
        Returns:
            Sample: New Sample object with data extracted from ASE Atoms.
        """
        return Sample(
            torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
            torch.tensor(atoms.get_positions()).float(),
            torch.tensor(np.array(atoms.get_cell(complete=True))).float(),
            atoms.get_pbc(),
            properties=None if properties is None else {p: torch.tensor(properties[p]) if properties[p] is not None else None for p in properties})

    def to_ase_atoms(self):
        """Create ASE Atoms object from this Sample.
        
        Returns:
            ase.Atoms: ASE Atoms object with positions, elements, cell, and PBC
                from this Sample. Properties are not transferred.
        """
        return Atoms(
            numbers=self.elements.detach().cpu().numpy(),
            positions=self.positions.detach().cpu().numpy(),
            cell=self.lattice.detach().cpu().numpy(),
            pbc=self.pbc
        )

    @torch.no_grad()
    def update_attrs(self, lattice=None, positions=None, elements=None, element_emb=None, properties=None):
        """Create a new Sample based on updated attributes.
        """
        new_lattice = lattice if lattice is not None else self.lattice
        new_positions = positions if positions is not None else self.positions
        new_elements = elements if elements is not None else self.elements
        new_element_emb = element_emb if element_emb is not None else self.element_emb
        new_properties = properties if properties is not None else self.properties

        return Sample(
            elements=new_elements,
            positions=new_positions,
            lattice=new_lattice,
            pbc=self.pbc,
            neighborlist=self.neighborlist if lattice is None else None,
            init_r_cut=self.neighborlist.init_r_cut if lattice is None else None,
            element_emb=new_element_emb,
            properties=new_properties)

    def null_properties(self, properties_to_null=None):
        """Set some or all of the properties to null

        Args:
            properties_to_null (_type_, optional): Which properties to null, if None, all are nulled. Defaults to None.

        Returns:
            Structure: The structure with the nulled properties
        """
        if properties_to_null is None:
            return self.update_attrs(properties={x: None for x in self.properties})
        else:
            return self.update_attrs(
                properties={x: None if x in properties_to_null else self.properties[x] for x in self.properties})

    def null_properties_random(self, null_prob=0.1, null_prop_all=0.1):
        if random() < null_prop_all:
            # Null all properties
            return self.null_properties(properties_to_null=None)
        else:
            # Null individual properties
            return self.null_properties(properties_to_null=[x for x in self.properties if random() < null_prob])

    def set_init_r_cut(self, init_r_cut):
        self.neighborlist.set_init_r_cut(init_r_cut)

    def update_edges(self, r_cut):
        self.neighborlist.update(self.positions, r_cut)

    def get_positions(self):
        return self.positions

    def get_elements(self):
        return self.elements

    def get_element_emb(self):
        return self.element_emb

    def get_property_arr(self, prop_name, null_placeholder=None):
        """
        Generate a tensor containing the property prop_name.
        The array is extended to the number of atoms, so it can be used as edge feature for the model.
        Returns None, if the property has been nulled.
        If null_placeholder is provided, it is used in case the property is null. 
        Additionally, a mask will be returned, that is true, in case of a null property. 

        Args:
            prop_name (str): Name of the requested property
            null_placeholder (FloatTensor): A placeholder value, in case the property has been nulled.

        Returns:
            FloatTensor: The requested property, extended to the number of atoms
        """
        p = self.properties[prop_name]
        if null_placeholder is None:
            if p is None:
                return None
            else:
                return p.unsqueeze(0).expand(self.get_num_atoms(), -1)
        else:
            if p is None:
                p = null_placeholder
            null_mask = torch.tensor([p is None], device=self.positions.device).expand(self.get_num_atoms())
            return p.unsqueeze(0).expand(self.get_num_atoms(), -1), null_mask

    def get_num_atoms(self):
        return len(self.elements)

    def get_edges(self, r_cut):
        return self.neighborlist.get_edges(self.positions, r_cut)

    def randomize_uniform(self):
        """Generate a random Sample that follows uniform distribution.
        """
        x = torch.rand_like(self.positions)
        x = x @ self.lattice
        return self.update_attrs(positions=x)

    def back_to_cell(self):
        """Wraps out-bound atoms back to the cell.
        """
        return self.update_attrs(positions=positions_into_cell(self.positions, self.lattice))

    def rotate(self, R):
        """Rotate the sample by a rotation matrix R.
        """
        return self.update_attrs(positions=self.positions @ R.T, lattice=self.lattice @ R.T)

    def remove_mean(self, x):
        return x - torch.mean(x, dim=0).unsqueeze(0)

    def get_batch_size(self):
        return 1

    def get_batch_indices(self):
        return torch.zeros(self.get_num_atoms(), dtype=torch.long)

    def is_charge_balanced(self):
        """
        Returns:
            bool: True if sum of 'valence_electrons' == 0, else False.
                  Based on ASE reference data for each element's ground state.
        """
        ase_atoms = self.to_ase_atoms()
        ele_count = Counter(ase_atoms.get_chemical_symbols())
        total_charge = sum([formal_charges[inv_element_map[e]] * ele_count[e] for e in ele_count if e != 'X'])

        return (total_charge == 0)

    def plot(self, plt_fig, r_cut, fold_back=False):
        """
        Plots a 3D visualization of a single sample from the dataset, including atom positions and their connections 
        based on a specified cutoff distance (r_cut). The plot includes optional folding back of atoms into the 
        primary simulation cell for better visualization.

        Args: 
            plt_fig (matplotlib.figure.Figure): The matplotlib figure object on which the plot will be drawn.
            r_cut (float): The cutoff distance for calculating edges between atoms. Edges are drawn if the distance
            between two atoms is less than or equal to this value.
            fold_back (bool, optional): If True, atom positions are folded back into the primary simulation cell
            to ensure all positions are within the cell boundaries. Defaults to False.

        Example usage:
        >>> fig = plt.figure(figsize=(10, 10))
        >>> sample.plot(fig, sample_i=0, r_cut=2.5, fold_back=True)
        >>> plt.show()
        """
        sample = self
        if fold_back:
            sample = sample.back_to_cell()
        sample.update_edges(r_cut)
        atom_coord = sample.get_positions().detach().cpu().numpy()
        atom_type = sample.get_elements().detach().cpu().numpy() + 1  # why +1?
        cell = sample.lattice.detach().cpu().numpy()
        # todo: for now this assumes orthorhombic cells
        cell_len = np.linalg.norm(cell, axis=1)

        cmap = plt.get_cmap('viridis')
        ax = plt_fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim((0, cell_len[0]))
        ax.set_ylim((0, cell_len[1]))
        ax.set_zlim((0, cell_len[2]))

        row, col, offset = [item.cpu().numpy()
                            for item in sample.get_edges(r_cut)]
        for i, (r, c, o) in enumerate(zip(row, col, offset)):
            alpha = (np.linalg.norm(
                atom_coord[r, :] - atom_coord[c, :] - o) % r_cut) / r_cut
            ax.plot(xs=[atom_coord[r, 0], atom_coord[c, 0] + o[0]],
                    ys=[atom_coord[r, 1], atom_coord[c, 1] + o[1]],
                    zs=[atom_coord[r, 2], atom_coord[c, 2] + o[2]],
                    color=cmap(atom_coord[r, 1] / cell_len[1]),
                    alpha=alpha, zorder=1, linewidth=2)
        for type_i in np.unique(atom_type):
            atom_coord_type = atom_coord[atom_type == type_i]
            ax.scatter(atom_coord_type[:, 0], atom_coord_type[:, 1],
                       atom_coord_type[:, 2], zorder=2)


class Batch:
    """Container for batching multiple Sample objects for efficient processing.
    
    Handles concatenation of atomic data from multiple samples while maintaining
    sample boundaries for proper graph construction and property tracking. Essential
    for batched training and inference in the diffusion model.
    
    Attributes:
        samples (list[Sample]): List of individual Sample objects.
        positions (torch.FloatTensor): Concatenated positions from all samples.
        elements (torch.LongTensor): Concatenated element types from all samples.
        element_emb (torch.FloatTensor, optional): Concatenated element embeddings.
    """

    def __init__(self, samples):
        """Initialize batch container with list of samples.
        
        Args:
            samples (list[Sample]): List of Sample objects to batch together.
                All samples should have compatible structure for concatenation.
        """
        self.samples = samples
        self.positions = torch.concat(
            [s.get_positions() for s in self.samples])
        self.elements = torch.concat([s.get_elements() for s in self.samples])
        element_embs = [s.get_element_emb() for s in self.samples]
        if not None in element_embs:
            self.element_emb = torch.concat(element_embs)
        else:
            self.element_emb = None

    def to(self, device):
        return Batch([x.to(device) for x in self.samples])

    def to_ase_atoms(self):
        """Create a list of ASE Atoms objects from this batch.
        """
        return [s.to_ase_atoms() for s in self.samples]

    def get_num_atoms(self):
        """Get the total number of atoms from all samples in this batch.
        """
        return sum([s.get_num_atoms() for s in self.samples])

    def get_positions(self):
        """Get the concatenated positions of all samples in this batch.

        Returns:
            torch.FloatTensor: concatenated positions of all samples, with shape (n_atoms, 3).
        """
        return self.positions

    def get_elements(self):
        return self.elements

    def get_element_emb(self):
        return self.element_emb

    def get_property_arr(self, prop_name, null_placeholder=None):
        if null_placeholder is None:
            return torch.concat([s.get_property_arr(prop_name) for s in self.samples])
        else:
            prop_arrs = [s.get_property_arr(prop_name, null_placeholder) for s in self.samples]
            return torch.concat([p for p, mask in prop_arrs]), torch.concat([mask for p, mask in prop_arrs])

    def set_init_r_cut(self, init_r_cut):
        for s in self.samples:
            s.set_init_r_cut(init_r_cut)

    def update_edges(self, r_cut):
        for sample in self.samples:
            sample.update_edges(r_cut)

    def get_edges(self, r_cut):
        """Get edges in all samples in this batch. The row and col indices have batch offsets built-in.
        """
        rows, cols, offsets = [], [], []
        atoms_i = 0
        for s in self.samples:
            r, c, o = s.get_edges(r_cut)
            rows.append(r + atoms_i)
            cols.append(c + atoms_i)
            offsets.append(o)
            atoms_i += s.get_num_atoms()
        return (torch.concat(rows), torch.concat(cols), torch.concat(offsets))

    def update_attrs(self, positions=None, elements=None, element_emb=None):
        """Update the attributes of all samples in this batch.
        The new attributes should have the same form as the original ones.
        """
        new_elements = elements if elements is not None else self.get_elements()
        new_element_emb = element_emb if element_emb is not None else self.get_element_emb()
        new_positions = positions if positions is not None else self.get_positions()

        atoms_i = 0
        new_samples = []
        for s in self.samples:
            last_atoms_i = atoms_i
            atoms_i = atoms_i + s.get_num_atoms()
            new_sample = s.update_attrs(
                elements=new_elements[last_atoms_i:atoms_i],
                positions=new_positions[last_atoms_i:atoms_i, :],
                element_emb=new_element_emb[last_atoms_i:atoms_i] if element_emb is not None else None)
            new_samples.append(new_sample)
        return Batch(new_samples)

    def null_properties(self, properties_to_null=None):
        return Batch([s.null_properties(properties_to_null) for s in self.samples])

    def null_properties_random(self, null_prob=0.1, null_prop_all=0.0):
        return Batch([s.null_properties_random(null_prob, null_prop_all) for s in self.samples])

    def remove_mean(self, x):
        mean = torch.zeros_like(x)
        batch_indices = self.get_batch_indices()
        for i in range(self.get_batch_size()):
            mask = batch_indices == i
            mean[mask, :] += torch.mean(x[mask, :], dim=0).unsqueeze(0)
        return x - mean

    def randomize_uniform(self):
        """Get a batch of random Samples that all follow the uniform distribution.
        """
        samples = [s.randomize_uniform() for s in self.samples]
        return Batch(samples)

    def get_batch_size(self):
        return len(self.samples)

    def get_batch_indices(self):
        return torch.concat(
            [torch.ones((s.get_num_atoms(),), dtype=torch.long) * i for i, s in enumerate(self.samples)])
    
    def rotate(self, R):
        return Batch([s.rotate(R) for s in self.samples])

    def is_charge_balanced(self):
        """
        Returns:
            torch.BoolTensor of shape (batch_size,):
            True where each sample is charge-balanced, False otherwise.
        """
        mask_list = []
        for s in self.samples:
            mask_list.append(s.is_charge_balanced())  # returns Python bool
        return torch.tensor(mask_list, dtype=torch.bool)

    def get_sub_batch(self, sample_indices):
        """
        Create a new Batch containing only the samples at `sample_indices`.
        """
        new_samples = [self.samples[i] for i in sample_indices]
        return Batch(new_samples)

    def update_sub_batch(self, sample_indices, sub_batch):
        """
        Update samples in-place for the specified `sample_indices`.
        Then refresh the concatenated attributes.
        """
        for local_i, global_i in enumerate(sample_indices):
            self.samples[global_i] = sub_batch.samples[local_i]
        self._refresh_cache()

    def _refresh_cache(self):
        """
        Internal helper to rebuild self.positions, self.elements, self.element_emb
        from the updated self.samples list.
        """
        self.positions = torch.concat([s.get_positions() for s in self.samples], dim=0)
        self.elements = torch.concat([s.get_elements() for s in self.samples], dim=0)

        element_embs = [s.get_element_emb() for s in self.samples]
        if any(e is None for e in element_embs):
            self.element_emb = None
        else:
            self.element_emb = torch.concat(element_embs, dim=0)

class MaterialCollateFn:
    """Custom collate function for batching material samples in DataLoader.
    
    Handles the special batching requirements of material structures, including
    proper concatenation of variable-sized samples and maintenance of sample
    boundaries for graph operations.
    
    Attributes:
        device (str): Target device for tensor placement ('cuda' or 'cpu').
    """
    
    def __init__(self, device):
        """Initialize collate function.
        
        Args:
            device (str): Device to place batched tensors on.
        """
        self.device = device

    def __call__(self, sample_batch):
        """Collate list of samples into a Batch object.

        Args:
            sample_batch (list[Sample]): List of Sample objects to batch.

        Returns:
            Batch: A batch of samples with ghost atoms added if target_density is set.
        """
        return Batch(sample_batch).to(self.device)
