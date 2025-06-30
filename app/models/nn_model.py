"""Simple feed-forward neural network for classification."""
from typing import Tuple
import torch
from torch import nn


class Classifier(nn.Module):
    """A basic neural network classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> float:
    """Perform a single training step."""
    model.train()
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
