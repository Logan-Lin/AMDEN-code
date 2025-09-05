import torch
from torch import nn

from models.embeddings import OneHotElementEmbedding
from models.networks.egnn import EGNN
from models.denoisers.egnn import EgnnDenoiser
from data import Batch


def check_nan(tensor, location):
    """Check if tensor contains NaN and log the location where NaN is detected."""
    if isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            if t is not None and torch.isnan(t).any():
                raise ValueError(f"NaN detected in {location} (tuple element {i})")
    elif tensor is not None and torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {location}")
    return tensor


class EgnnDerivDenoiser(EgnnDenoiser):
    
    def __init__(self, 
                 r_cut, 
                 kB_T=1.0, atomic_energy_scale=1.0, 
                 elements=None, properties=None, d_prop_embed=None, node_attrs=False, icg_mode=None, prop_embed_ln=False,
                 **egnn_args):

        super().__init__(r_cut, 
                         elements=elements, 
                         properties=properties, 
                         d_prop_embed=d_prop_embed, 
                         node_attrs=node_attrs, 
                         icg_mode=icg_mode,
                         prop_embed_ln=prop_embed_ln,
                         **egnn_args)
        self.kB_T = kB_T
        self.atomic_energy_scale = atomic_energy_scale

    # overrides the parent method
    def init_egnn(self, egnn_args):
        in_node_nf = 1 # time
        out_node_nf = 1 # atomic energy
    
        if self.elements is not None:
            in_node_nf = in_node_nf + self.element_embedding.dim

        if self.properties is not None:
            if self.use_property_embedding: 
                in_node_nf = in_node_nf + sum([v.get('d_prop_embed', self.d_prop_embed) for _, v in self.properties.items()])
            else:
                in_node_nf = in_node_nf + sum([v.get('dim', self.d_prop_embed) for _, v in self.properties.items()])

        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            r_cut=self.cutoff_radius,
            out_node_nf=out_node_nf,
            nodes_att_dim=in_node_nf if self.node_attrs else 0,
            **egnn_args)

    # score = d/dx ln(p(x))
    # noise = -sigma * score
    # p(x) = exp(-E(x)/(kB_T)) # by definition
    # score = d/dx -E(x) / (kB_T)
    # noise = d/dx sigma * E(x) / (kB_T)
    # N(x) = sigma * E(x) / kB_T # this is what we call noise_energy
    # It follows...
    # score = d/dx -N(x) / sigma 
    # noise = d/dx N(x) 
    def noise_energy_fn(self, sample, edges, x, element_emb, t):
        h = self.assemble_h(sample, element_emb, t)

        h_out, x_out = self.egnn(h, x, edges, node_attr=h if self.node_attrs else None)

        # https://openreview.net/pdf?id=9AS-TF2jRNb
        # (N_atoms, N_coordinates, N_DIMENSIOS) -> mean of coordinates, sum xyz to get per atom contributions
        pos_nenergy = torch.sum(torch.mean(0.5 * (x.unsqueeze(1) - x_out)**2, dim=1), dim=1)

        # Use the nn output as well
        # Might actually be necessary, because we dont want all x with dE/dx to have the same energy
        atomic_nenergy = h_out.squeeze(1) * self.atomic_energy_scale
        # get the batching information
        batch_idx = sample.get_batch_indices().to(h.device)
        batch_size = sample.get_batch_size() 
        # sum up atomic energies on per system basis
        nenergy = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        nenergy = nenergy.index_add(0, batch_idx, pos_nenergy + atomic_nenergy) 
        nenergy = nenergy / self.kB_T 
        return nenergy
    
    # converts nenergy to energy
    def get_energy_and_forces(self, sample, t, sigma_t, calc_energy=True, calc_forces=True, calc_dE_del=False):
        nenergy, dE_dx, dE_del = self.get_noise_energy(sample, t, calc_E=calc_energy, calc_dE_dx=calc_forces, calc_dE_del=calc_dE_del)
        energy = nenergy / sigma_t if calc_energy else None
        forces = -1.0 * dE_dx / sigma_t[sample.get_batch_indices()].unsqueeze(1) if calc_forces else None
        dE_del = dE_del / sigma_t[sample.get_batch_indices()].unsqueeze(1) if calc_dE_del else None
        return energy, forces, dE_del


    # the noise energy is energy * sigma_t
    def get_noise_energy(self, sample, t, calc_E=True, calc_dE_dx=False, calc_dE_del=False):
        edges = self._get_edges(sample)
        x = sample.get_positions()
        element_emb = sample.get_element_emb()

        nenergy = None
        dE_dx = None
        dE_del = None

        if calc_E:
            nenergy = self.noise_energy_fn(sample, edges, x, element_emb, t)

        # Since the energies of all samples in the batch are independent, we just sum them up when calculating derivatives
        if calc_dE_dx:
            dE_dx = torch.autograd.functional.jacobian(lambda x: torch.sum(self.noise_energy_fn(sample, edges, x, element_emb, t)), x, create_graph=self.training)
            if self.training:
                check_nan(dE_dx, f"{self.__class__.__name__}.get_noise_energy - dE_dx")

        if calc_dE_del:
            dE_del = torch.autograd.functional.jacobian(lambda element_emb: torch.sum(self.noise_energy_fn(sample, edges, x, element_emb, t)), element_emb, create_graph=self.training)
            if self.training:
                check_nan(dE_del, f"{self.__class__.__name__}.get_noise_energy - dE_del")
        return nenergy, dE_dx, dE_del

    # calculates the full derivative d/dt ln(p(x, t, sigma(t)))
    def get_dlnp_dt(self, sample, t, sigma_fn):
        edges = self._get_edges(sample)
        x = sample.get_positions()
        element_emb = sample.get_element_emb()
        dlnp_dt = torch.autograd.functional.jacobian(lambda t: torch.sum(-1. / sigma_fn(t) * self.noise_energy_fn(sample, edges, x, element_emb, t)), t, create_graph=self.training)
        return dlnp_dt

    # overrides the parent method, predicts the noise (eps = -sigma * s)
    def forward(self, sample, t):
        _, dE_dx, dE_del = self.get_noise_energy(sample, t, calc_E=False, calc_dE_dx=True, calc_dE_del=self.element_embedding is not None)
        return (dE_dx, dE_del)



