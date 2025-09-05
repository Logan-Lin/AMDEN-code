import torch
from torch import nn
from models.embeddings import OneHotElementEmbedding


class BaseDenoiser(nn.Module):
    """Base class for all denoisers implementing common functionality."""
    
    def __init__(self, cutoff_radius, elements=None, properties=None, d_prop_embed=None, node_attrs=False,
                 icg_mode=None, prop_embed_ln=False):
        """
        Args:
            cutoff_radius (float): Cutoff radius for atomic interactions
            elements (list, optional): List of elements to embed. Defaults to None.
            properties (dict, optional): Dictionary of property specifications. Defaults to None.
            d_prop_embed (int, optional): Default embedding dimension for properties. Defaults to None.
            node_attrs (bool, optional): Whether to use node attributes. Defaults to False.
            icg_mode (str, optional): Independent Condition Guidance (ICG) mode. Defaults to None.
                - None: Use conventional regressor-free guidance instead of ICG.
                - 'embed': Sample noisy condition from the embedding space.
                - 'prop': Sample noisy condition from the property space.
        """
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.elements = elements
        self.node_attrs = node_attrs
        self.properties = properties
        self.d_prop_embed = d_prop_embed
        self.icg_mode = icg_mode
        if icg_mode == 'embed':
            assert prop_embed_ln, "Layernorm of property embedding must be used if ICG mode is 'embed'"

        if self.elements is not None:
            self.element_embedding = OneHotElementEmbedding(self.elements)
        else:
            self.element_embedding = None

        self.use_property_embedding = self.d_prop_embed is not None

        if self.properties is not None and self.use_property_embedding:
            self.prop_embed = nn.ModuleDict({
                k: nn.Sequential(
                    nn.Linear(v['dim'], v.get('d_prop_embed', self.d_prop_embed)),
                    nn.LayerNorm(v.get('d_prop_embed', self.d_prop_embed)) if prop_embed_ln else nn.Identity()
                )
                for k, v in self.properties.items()
            })
            self.null_prop = nn.ParameterDict({
                k: nn.Parameter(torch.randn(v.get('d_prop_embed', self.d_prop_embed)), requires_grad=True)
                for k, v in self.properties.items()
            })

    def predict_noise(self, sample, t):
        """Wrapper for forward pass."""
        return self(sample, t)

    @torch.compiler.disable
    def _get_edges(self, sample):
        """Get edges with compiler disabled."""
        return sample.get_edges(self.cutoff_radius)

    def embed_properties(self, sample):
        """Embed sample properties."""
        props = []
        for prop_name in self.properties:
            prop, null_mask = sample.get_property_arr(
                prop_name,
                null_placeholder=torch.zeros(
                    self.properties[prop_name]['dim'], 
                    device=sample.positions.device
                )
            )
            
            if 'offset' in self.properties[prop_name]:
                prop = prop - self.properties[prop_name]['offset']
            if 'scale' in self.properties[prop_name]:
                prop = prop / self.properties[prop_name]['scale']

            if not self.use_property_embedding:
                if torch.any(null_mask):
                    raise Exception('To use null properties, property embedding must be used')
                props.append(prop)
            else:
                prop_emb_arr = self.prop_embed[prop_name](prop)
                if self.icg_mode == None:
                    prop_emb_arr[null_mask, :] = self.null_prop[prop_name]
                elif self.icg_mode == 'embed':
                    prop_emb_arr[null_mask, :] = torch.randn(
                        prop_emb_arr[null_mask, :].shape,
                        device=prop_emb_arr.device
                    )
                elif self.icg_mode == 'prop':
                    # Suppose we always scale the input properties using the offset and scale.
                    noisy_prop = self.prop_embed[prop_name](torch.randn_like(prop))
                    prop_emb_arr[null_mask, :] = noisy_prop[null_mask, :]
                props.append(prop_emb_arr)
        return props

    def get_properties(self):
        """Get properties dictionary."""
        return self.properties

    def assemble_h(self, sample, element_emb, t):
        """Assemble node features for GNN input.
        
        Combines timestep, element embeddings, and property embeddings into
        node feature vectors for the graph neural network.
        
        Args:
            sample (Sample): Material sample containing properties and batch indices.
            element_emb (torch.Tensor): Element embeddings, shape (n_atoms, embed_dim).
            t (torch.Tensor): Diffusion timesteps, shape (batch_size,).
            
        Returns:
            torch.Tensor: Assembled node features, shape (n_atoms, total_feature_dim).
        """
        batch_idx = sample.get_batch_indices().to(sample.get_positions().device)
        h = t[batch_idx].unsqueeze(-1)
        
        if self.element_embedding is not None:
            h = torch.concat((h, element_emb), dim=1)

        if self.properties is not None:
            props = self.embed_properties(sample)
            h = torch.concat([h] + props, dim=1)
        return h

    def forward(self, sample, t):
        """
        Forward pass to be implemented by subclasses.
        Should return (position_noise, element_noise).
        """
        raise NotImplementedError 