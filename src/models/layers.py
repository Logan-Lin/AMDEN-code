import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import Node2Vec


class SinusoidsEmbedding(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        """Sinusoidal embedding for squared distances (assumes input is squared)
            max_res: Maximum wavelength 
            min_res: Minimum wavelength
            div_factor: ratio between following wavelengths/frequencies
        """
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * \
                           div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):  # NOTE: x is the squared distance (use t**2 when embedding time)
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class FourierEncode(nn.Module):
    """A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        Args:
            x (FloatTensor): input features for encoding with shape (batch_size).

        Returns:
            FloatTensor: encoded features with shape (batch_size, embed_size).
        """
        x = x.unsqueeze(-1)
        encode = x * self.omega.reshape(1, -1) + self.bias.reshape(1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode


class Element2Vec(nn.Module):
    """Embed each element into an latent vector by utilizing the node2vec model.
    """

    def __init__(self, element_edges, embed_dim, **node2vec_param):
        """
        Args:
            element_edges (LongTensor): the set of all edge indices between elements with shape (2, N),
            denoting the virtual graph connecting elements.
            The virtual graph should represents the relationships between elements, for example,
            their relative position in the periodic table.
            embed_dim (int): the size of each embedding vector.
            node2vec_param (dict): detailed parameters controlling the training process of node2vec. 
            https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.models.Node2Vec.html.
        """
        super().__init__()

        self.node2vec = Node2Vec(
            element_edges, embedding_dim=embed_dim, **node2vec_param)

    def forward(self, elements):
        """
        Args:
            elements (LongTensor): the element indices.

        Returns:
            FloatTensor: the embedding vectors of the given elements.
        """
        return self.node2vec(elements)


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class GaussianSmearing(nn.Module):
    """Gaussian Smearing module for approximating the effects of temperature on electronic states in simulations.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(nn.Module):
    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer('freq_k', torch.arange(
            1, num_basis_k + 1).float())
        self.register_buffer('freq_l', torch.arange(
            1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # (num_angles, num_basis_k)
        s = torch.sin(angle.view(-1, 1) * self.freq_k.view(1, -1))
        # (num_angles, num_basis_l)
        c = torch.cos(angle.view(-1, 1) * self.freq_l.view(1, -1))
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(nn.Module):
    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer('freq_k', torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        # (num_angles, num_basis)
        return torch.cos(angle.view(-1, 1) * self.freq_k.view(1, -1))
