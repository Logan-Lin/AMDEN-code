import torch
from ase.symbols import symbols2numbers
from torch import nn


class OneHotElementEmbedding(nn.Module):
    """One-hot encoding for element types.
    
    Converts atomic numbers to one-hot vectors for input to neural networks.
    Maintains mappings between atomic numbers and embedding indices, enabling
    both forward embedding and reverse un-embedding operations.
    
    The embedding dimension equals the number of unique element types, providing
    a simple but effective representation for material composition.
    
    Attributes:
        n_elements (int): Number of unique element types.
        dim (int): Embedding dimension (equals n_elements).
        element_idx (torch.LongTensor): Maps atomic numbers to embedding indices.
        inverse_element_idx (torch.LongTensor): Maps embedding indices back to atomic numbers.
    """
    
    def __init__(self, elements):
        """Initialize one-hot element embedding.
        
        Args:
            elements (list): List of element types (atomic numbers or symbols).
                Can contain integers (atomic numbers) or strings (element symbols).
        """
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
        """Convert element indices to one-hot embeddings.
        
        Args:
            elements (torch.LongTensor): Atomic numbers, shape (n_atoms,).
            
        Returns:
            torch.FloatTensor: One-hot embeddings, shape (n_atoms, n_elements).
        """
        emb = nn.functional.one_hot(self.element_idx[elements], self.n_elements).float()
        return emb

    def unembed(self, element_emb):
        """Convert one-hot embeddings back to element indices.
        
        Uses argmax to find the most likely element type from the embedding.
        Primarily used during inference to convert predicted embeddings back
        to discrete element types.
        
        Args:
            element_emb (torch.FloatTensor): Element embeddings, shape (n_atoms, n_elements).
            
        Returns:
            torch.LongTensor: Atomic numbers, shape (n_atoms,).
        """
        return self.inverse_element_idx[torch.argmax(element_emb, dim=1)]
