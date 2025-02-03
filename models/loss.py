import torch
import torch.nn.functional as F
from src import config  

def contrastive_loss(output1, output2, target):
    """
    Вычисляет контрастивный (contrastive) loss.

    Args:
        output1 (torch.Tensor): Выход первого ответвления Siamese сети.
        output2 (torch.Tensor): Выход второго ответвления Siamese сети.
        target (torch.Tensor): Целевой тензор (0 для схожих пар, 1 для несхожих пар).

    Returns:
        torch.Tensor: Вычисленное значение контрастивного loss.
    """    
    margin = config.MARGIN
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = torch.mean(
        (1-target) * torch.pow(euclidean_distance, 2) +
        (target) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    )
    return loss