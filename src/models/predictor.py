"""
Particle Predictor for JEPA pretraining.

Given the encoder's output over all particle positions (with the target particle
zeroed out), the predictor extracts the representation at the masked position,
projects it through a narrow bottleneck transformer, and returns a predicted
embedding to be compared against the target encoder's output at that position.

Architecture (following I-JEPA):
  - input_proj  : encoder_dim → predictor_dim  (bottleneck)
  - pos_embed   : learnable per-particle positional embedding
  - mask_token  : learnable token replacing the masked position
  - transformer : N lightweight blocks at predictor_dim
  - output_proj : predictor_dim → encoder_dim

The bottleneck (predictor_dim < encoder_dim) prevents the context encoder from
collapsing by forcing it to summarise particle information rather than just
copying representations through.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ParticlePredictor(nn.Module):
    """
    Predicts the target encoder's embedding at a masked particle position
    from the context encoder's full sequence output.

    Parameters
    ----------
    encoder_dim : int
        Output dimension of the LorentzParTEncoder (default 128).
    predictor_dim : int
        Hidden dimension of the predictor's bottleneck (default 64).
    num_heads : int
        Number of attention heads in the predictor transformer (default 4).
    num_layers : int
        Number of transformer encoder layers (default 4).
    max_num_particles : int
        Maximum sequence length; used to size the positional embedding table.
    dropout : float
        Dropout rate for the predictor transformer layers (default 0.1).
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        predictor_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        max_num_particles: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        # Project from encoder dimension down to predictor bottleneck
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # Learnable positional embedding: one vector per particle slot
        self.pos_embed = nn.Embedding(max_num_particles, predictor_dim)

        # Learnable mask token (shared across positions)
        self.mask_token = nn.Parameter(torch.zeros(predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Lightweight transformer operating in predictor_dim space
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(predictor_dim)

        # Project predictions back up to encoder dimension for loss computation
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, encoder_output: Tensor, mask_idx: Tensor) -> Tensor:
        """
        Parameters
        ----------
        encoder_output : Tensor, shape (B, N, encoder_dim)
            Full sequence output from the context encoder (masked input).
        mask_idx : Tensor, shape (B,) dtype long
            Index of the masked particle in each batch item.

        Returns
        -------
        prediction : Tensor, shape (B, encoder_dim)
            Predicted embedding for the masked particle position.
        """
        B, N, _ = encoder_output.shape
        device = encoder_output.device

        # Project all positions to predictor dimension
        x = self.input_proj(encoder_output)           # (B, N, predictor_dim)

        # Add positional embeddings to all positions
        positions = torch.arange(N, device=device)    # (N,)
        x = x + self.pos_embed(positions).unsqueeze(0)  # (B, N, predictor_dim)

        # Replace the masked position with mask_token + its positional embedding
        batch_idx = torch.arange(B, device=device)
        mask_pos_embed = self.pos_embed(mask_idx)     # (B, predictor_dim)
        x[batch_idx, mask_idx] = self.mask_token.unsqueeze(0) + mask_pos_embed

        # Run through transformer
        x = self.transformer(x)                        # (B, N, predictor_dim)
        x = self.norm(x)

        # Extract prediction at the masked position only
        pred = x[batch_idx, mask_idx]                  # (B, predictor_dim)

        # Project back to encoder dimension
        pred = self.output_proj(pred)                  # (B, encoder_dim)

        return pred
