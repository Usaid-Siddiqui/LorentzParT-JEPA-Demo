from torch import nn

from .conservation_loss import ConservationLoss
from .embedding_loss import EmbeddingLoss


LOSS_REGISTRY = {
    'conservation_loss': ConservationLoss,
    'embedding_loss': EmbeddingLoss,
    'cross_entropy_loss': nn.CrossEntropyLoss,
    'bce_with_logits_loss': nn.BCEWithLogitsLoss,
    'mse_loss': nn.MSELoss,
    # Add more loss functions here as needed
}