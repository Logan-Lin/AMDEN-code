import ase
import numpy as np
import torch
from ase.neighborlist import NeighborList as AseNeighborList
from torch_nl import compute_neighborlist, ase2data
from utils import positions_into_cell



class Neighborlist:
    """Efficient neighbor list computation for periodic systems.
    
    Manages neighbor finding for atomic systems with periodic boundary conditions.
    Pre-computes and caches neighbor lists at initialization cutoff, then provides
    efficient edge extraction at smaller cutoffs during model evaluation.
    
    Essential for graph neural networks operating on material systems where
    atoms interact with neighbors within a cutoff radius across periodic boundaries.
    
    Attributes:
        lattice (torch.Tensor): Unit cell matrix, shape (3, 3).
        pbc (tuple[bool]): Periodic boundary conditions for each axis.
        init_r_cut (float, optional): Initial cutoff for neighbor list construction.
        use_torch_nl (bool): Whether to use torch_nl backend for efficiency.
    """
    
    def __init__(self, lattice, pbc, init_r_cut=None, use_torch_nl=True):
        """Initialize neighbor list manager.
        
        Args:
            lattice (torch.Tensor): Unit cell matrix, shape (3, 3).
            pbc (tuple[bool]): Periodic boundary conditions for each axis.
            init_r_cut (float, optional): Initial cutoff for neighbor list construction.
            use_torch_nl (bool, optional): Use torch_nl backend. Defaults to True.
        """
        self.lattice = lattice
        self.pbc = pbc
        self.init_r_cut = init_r_cut # this will be used for NL construction, if provided

        self.orig_pos = None # positions from which current list was built
        self.r_cut = None # r_cut with which it was built
        self.edges = None # the current NL edges

        self.use_torch_nl = use_torch_nl
    
    def set_init_r_cut(self, init_r_cut):
        self.init_r_cut = init_r_cut 

    def to(self, device):
        nl = Neighborlist(self.lattice.to(device), self.pbc, self.init_r_cut)
        if self.orig_pos is not None:
            nl.orig_pos = self.orig_pos.to(device)
        nl.r_cut = self.r_cut
        if self.edges is not None:
            nl.edges = (x.to(device) for x in self.edges)
        return nl

    @torch.no_grad()
    def update(self, positions, r_cut):
        nat = positions.shape[0]
        # NOTE: torch_nl also just uses a for loop over batches, so there is no advantage to be gained here. 
        if self.use_torch_nl:
            positions_c = positions_into_cell(positions, self.lattice) # need to be wrapped for torch_nl
            neighbors, batch_indices, offset_indices = compute_neighborlist(
                    r_cut,
                    positions_c,
                    self.lattice,
                    torch.tensor(self.pbc).to(positions.device),
                    torch.zeros(nat, dtype=torch.long).to(positions.device),
                    self_interaction = False)
            rows = neighbors[0, :]
            cols = neighbors[1, :]
            d_cell = positions_c - positions
            offsets = offset_indices @ self.lattice - (d_cell[rows] - d_cell[cols])
        else:
            atoms = ase.Atoms(
                numbers=np.ones(nat, dtype=np.int32),
                positions=positions.detach().cpu().numpy(),
                cell=self.lattice.detach().cpu().numpy(),
                pbc=self.pbc
            )
            nl = AseNeighborList(
                [r_cut / 2.] * nat,
                self_interaction=False,
                bothways=False,
                skin=0.)
            nl.update(atoms)
            neighbors, offset_indices = nl.get_neighbors(slice(None))
            rows, cols, offsets = [], [], []
            lat = self.lattice.cpu()
            for i, (neis, offs) in enumerate(zip(neighbors, offset_indices)):
                rows.extend([i] * len(neis))
                cols.extend(neis)
                if self.lattice is None:
                    offsets.append(torch.zeros(
                        len(neis), 2, dtype=positions.dtype))
                else:
                    offsets.append((torch.tensor(offs * 1.0).float()
                                   @ lat).to(positions.dtype))
            rows = torch.LongTensor(rows)
            cols = torch.LongTensor(cols)
            offsets = torch.cat(offsets, dim=0) if len(
                offsets) > 0 else torch.zeros(0, positions.shape[1])

            rows = rows.to(positions.device)
            cols = cols.to(positions.device)
            offsets = offsets.to(positions.device)
            rows, cols, offsets = torch.cat((rows, cols)), torch.cat((cols, rows)), torch.cat((offsets, -1 * offsets))

        self.r_cut = r_cut
        self.orig_pos = positions.clone()
        self.edges = (rows, cols, offsets)

    @torch.no_grad()
    def get_edges(self, positions, r_cut):
        if self.edges is None:
            self.update(positions, r_cut if self.init_r_cut is None else self.init_r_cut)
        else:
            disp = torch.linalg.norm(positions - self.orig_pos, dim=1)
            d_max = torch.max(disp).detach().item() # maximum distance any atom has been moved since edges were built
            curr_r_cut = self.r_cut - 2 * d_max # worst case, two atoms move towards each other by the max distance, hence, we reduce r_cut by that amount
            if curr_r_cut < r_cut:
                self.update(positions, r_cut if self.init_r_cut is None else self.init_r_cut)
        
        # trim the edges that are > r_cut
        row, col, offs = self.edges
        edge_len = torch.linalg.norm(
            positions[row, :] - positions[col, :] - offs, dim=1)
        keep_edge = edge_len <= r_cut
        return row[keep_edge], col[keep_edge], offs[keep_edge, :]
