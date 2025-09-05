import torch.nn.functional as F
from torch import nn


class EpsilonDiff:
    """Supervising based on the predicted and ground truth noise (epsilon) of one diffusion step."""
    def __init__(self, norm_type, element_norm_type=None, position_weight=1.0, element_weight=1.0, dpdt_weight=0.):
        """
        Args:
            norm_type: type of normalization on the positions (l1, l2, huber, or kl).
            element_norm_type: type of normalization on the elements.
            position_weight: weight applied on the positions.
            element_weight: weight applied on the elements.
        """
        self.position_weight = position_weight
        self.element_weight = element_weight
        self.dpdt_weight = dpdt_weight   # For now, this class just holds the weight - it is used in the pipeline.
        if element_norm_type is None:
            element_norm_type = norm_type
        
        self.pos_loss_func = self._get_loss_func(norm_type)
        self.el_loss_func = self._get_loss_func(element_norm_type)

    def __call__(self, noise_pos, pred_noise_pos, noise_els, pred_noise_els, **kwargs):
        loss_pos = self.pos_loss_func(noise_pos, pred_noise_pos)
        if noise_els is not None and pred_noise_els is not None:
            loss_el = self.el_loss_func(noise_els, pred_noise_els)
        else:
            loss_el = 0.0

        loss_tot = self.position_weight * loss_pos + self.element_weight * loss_el
        return loss_tot, (loss_pos, loss_el)
    
    @staticmethod
    def _get_loss_func(name):
        if name == 'l1':
            loss_func = F.l1_loss
        elif name == 'l2':
            loss_func = F.mse_loss
        elif name == 'huber':
            # Huber loss is a combination of L1 and L2 losses and is widely used in diffusion models.
            loss_func = F.smooth_l1_loss
        elif name == 'kl':
            # KL loss is good for comparing two distributions.
            loss_func = nn.KLDivLoss(reduction='batchmean')
        else:
            raise NotImplementedError(f'Unknown loss function: {name}')
        
        return loss_func
