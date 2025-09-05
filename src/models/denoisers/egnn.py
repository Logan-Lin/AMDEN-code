from models.denoisers.base import BaseDenoiser
from models.networks.egnn import EGNN


class EgnnDenoiser(BaseDenoiser):
    """E(3)-equivariant graph neural network denoiser for material diffusion.
    
    Implements an EGNN-based denoiser that predicts noise in atomic positions and
    element embeddings while maintaining E(3) equivariance (rotation and translation
    invariance). This is crucial for learning physical systems where the energy and
    structure should be independent of global rotations and translations.
    
    The denoiser takes noisy material configurations at timestep t and predicts the
    noise that was added, enabling iterative denoising in the reverse diffusion process.
    
    Attributes:
        egnn (EGNN): The underlying E(3)-equivariant graph neural network.
    """
    
    def __init__(self, r_cut, elements=None, properties=None,
                 d_prop_embed=None, node_attrs=False, icg_mode=None, prop_embed_ln=False, **egnn_args):
        """Initialize EGNN denoiser.
        
        Args:
            r_cut (float): Cutoff radius for neighbor interactions in Angstroms.
            elements (list[int], optional): List of element types to embed.
            properties (dict, optional): Property specifications for conditioning.
            d_prop_embed (int, optional): Embedding dimension for properties.
            node_attrs (bool, optional): Whether to use node attributes. Defaults to False.
            icg_mode (str, optional): Independent Condition Guidance mode.
                Options: None (standard), 'embed' (noisy embedding), 'prop' (noisy property).
            prop_embed_ln (bool, optional): Apply LayerNorm to property embeddings.
            **egnn_args: Additional arguments passed to EGNN network.
        """
        super().__init__(r_cut, elements, properties, d_prop_embed, node_attrs, icg_mode, prop_embed_ln)
        self.init_egnn(egnn_args)

    def init_egnn(self, egnn_args):
        """Initialize the underlying EGNN network with appropriate input/output dimensions.
        
        Args:
            egnn_args (dict): Configuration for EGNN network architecture.
        """
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
        """Forward pass to predict noise in positions and optionally elements.
        
        Args:
            sample (Sample): Noisy material sample at timestep t.
            t (torch.FloatTensor): Diffusion timestep, shape (batch_size,).
            
        Returns:
            tuple: (noise_pos, noise_els) where:
                - noise_pos: Predicted position noise, shape (n_atoms, 3)
                - noise_els: Predicted element embedding noise or None
        """
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

