from models.denoisers.base import BaseDenoiser
from models.networks.egnn import EGNN


class EgnnDenoiser(BaseDenoiser):
    def __init__(self, r_cut, elements=None, properties=None,
                 d_prop_embed=None, node_attrs=False, icg_mode=None, prop_embed_ln=False, **egnn_args):
        super().__init__(r_cut, elements, properties, d_prop_embed, node_attrs, icg_mode, prop_embed_ln)
        self.init_egnn(egnn_args)

    def init_egnn(self, egnn_args):
        in_node_nf = 1  # time
        out_node_nf = 0
    
        if self.elements is not None:
            in_node_nf = in_node_nf + self.element_embedding.dim
            out_node_nf = self.element_embedding.dim

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

    def forward(self, sample, t):
        edges = self._get_edges(sample)
        h = self.assemble_h(sample, sample.get_element_emb(), t)
        x = sample.get_positions()

        h_out, x_out = self.egnn(h, x, edges, node_attr=h if self.node_attrs else None)
        x_out = x_out[:, 0, :]

        noise_pos = x - x_out
        noise_pos = sample.remove_mean(noise_pos)

        noise_els = None
        if self.element_embedding:
            noise_els = h_out

        return (noise_pos, noise_els)

