import torch
from ase.symbols import symbols2numbers
from torch import nn


class OneHotElementEmbedding(nn.Module):
    def __init__(self, elements):
        super().__init__()
        self.n_elements = len(elements)
        self.dim = self.n_elements  # dimension of the element embedding
        element_idx = torch.full((120,), -1, dtype=torch.long)  # for all elements
        inverse_element_idx = torch.zeros(self.n_elements,
                                          dtype=torch.long)
        for i_el, el in enumerate(elements):
            if type(el) is str:
                el_nr = symbols2numbers(el)[0]
            else:
                el_nr = el
            element_idx[el_nr] = i_el
            inverse_element_idx[i_el] = el_nr

            # register them as buffers, so they are saved with the model and pushed to gpu with the .to(...) function
        self.register_buffer('element_idx', element_idx)
        self.register_buffer('inverse_element_idx', inverse_element_idx)

    def embed(self, elements):
        emb = nn.functional.one_hot(self.element_idx[elements], self.n_elements).float()
        return emb

    def unembed(self, element_emb):
        return self.inverse_element_idx[torch.argmax(element_emb, dim=1)]
