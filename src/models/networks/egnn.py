import torch
from torch import nn


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False, residual=True):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.residual = residual

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col, _ = edge_index
        # aggregate and flatten out the n_coords dimension
        agg = unsorted_segment_sum(edge_attr.unsqueeze(1), row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method).squeeze(1)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        if self.residual:
            out = x + self.node_mlp(agg)
        else:
            out = self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col, _ = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 n_coords=1):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        self.n_coords = n_coords
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, self.n_coords ** 2, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col, _ = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        # predict weighting for all pairs of coordinates
        phi = self.coord_mlp(input_tensor).reshape([-1, self.n_coords, self.n_coords])
        if self.tanh:
            trans = torch.tanh(torch.einsum('ijk,ijl->ilk', coord_diff, phi)) * self.coords_range
        else:
            trans = torch.einsum('ijk,ijl->ilk', coord_diff, phi)
        if edge_mask is not None:
            trans = trans * edge_mask.unsqueeze(1)
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=1, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 tanh=False, coords_range=15, norm_constant=1,
                 normalization_factor=100, aggregation_method='sum', r_cut=1., n_coords=1,
                 distance_embedding=None, residual=True,
                 nodes_att_dim=0):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.r_cut = r_cut
        self.n_coords = n_coords
        self.distance_embedding = distance_embedding
        self.residual = residual
        self.nodes_att_dim = nodes_att_dim

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method,
                                              residual=self.residual,
                                              nodes_att_dim=self.nodes_att_dim))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf,
                                                       act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,
                                                       n_coords=self.n_coords))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None, node_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)

        if self.distance_embedding is not None:
            distances = self.distance_embedding(distances)

        edge_attr = torch.cat([distances.flatten(1, 2), edge_attr.flatten(1, 2)], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_attr=node_attr,
                                               node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, r_cut, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, sin_embedding_time=False, normalization_factor=100, aggregation_method='sum',
                 n_coords=1, residual=True, nodes_att_dim=0):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = 0
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        # the sum in the aggregation is divided by this - should probably be roughly the number of neighbors
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method  # only sum makes sense for our use case
        self.r_cut = r_cut
        self.n_coords = n_coords  # https://ml4astro.github.io/icml2023/assets/68.pdf

        if sin_embedding:
            self.distance_embedding = ...
            raise NotImplementedError("Sin Embedding of distances is not implemented")
        else:
            self.distance_embedding = lambda x: torch.tanh(x / self.r_cut ** 2) * 2 - 1

        edge_feat_nf = 1 + n_coords  # original edge length + one for each coordinate

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        if out_node_nf > 0:
            self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        else:
            self.embedding_out = None
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               r_cut=self.r_cut,
                                                               distance_embedding=self.distance_embedding,
                                                               n_coords=self.n_coords,
                                                               residual=residual,
                                                               nodes_att_dim=nodes_att_dim))
        self.to(self.device)

    # def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):

    def forward(self, h, x, edges, node_attr=None):

        distances, _ = coord2diff(x.unsqueeze(1), edges)
        edge_mask = self.cutoff_function(torch.sqrt(distances)).squeeze(1)
        node_mask = None

        # all the coordinates are initialized with atom positions
        x = x.unsqueeze(1).expand(-1, self.n_coords, -1)

        if self.distance_embedding is not None:
            distances = self.distance_embedding(distances)

        h = self.embedding(h)

        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances, node_attr=node_attr)

        if self.embedding_out:
            h = self.embedding_out(h)
            # Important, the bias of the last linear might be non-zero
            if node_mask is not None:
                h = h * node_mask
        else:
            h = None

        return h, x

    def cutoff_function(self, r):
        return torch.where(r <= self.r_cut, torch.tanh(1. - r / self.r_cut) ** 2 * 2., 0.)


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


def coord2diff(x, edge_index, norm_constant=1.0):
    row, col, offset = edge_index
    coord_diff = x[row] - x[col] - offset.unsqueeze(1)
    radial = torch.sum((coord_diff) ** 2, 2).unsqueeze(2)
    norm = torch.sqrt(radial + 1)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1), data.size(2))
    result = data.new_zeros(result_shape)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, data.size(1), data.size(2))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
