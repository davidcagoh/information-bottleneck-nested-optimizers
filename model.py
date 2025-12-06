import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small MLP for Tishby-style experiments.

    Architecture:
      - input: flattened 784
      - hidden: 12 -> 10 -> 8 -> 6 -> 4 (tanh activations)
      - output: 10 units (logits)

    The forward method returns (out, [h1, h2, h3]) to remain compatible
    with existing MI-estimation code that expects a list of hidden representations.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, 6)
        self.fc5 = nn.Linear(6, 4)
        self.fc6 = nn.Linear(4, 10)
        self.tanh = nn.Tanh()

    def forward(self, x):
      x = x.view(x.size(0), -1)
      h1 = self.tanh(self.fc1(x))
      h2 = self.tanh(self.fc2(h1))
      h3 = self.tanh(self.fc3(h2))
      h4 = self.tanh(self.fc4(h3))
      h5 = self.tanh(self.fc5(h4))
      out = self.fc6(h5)
      return out, [h1, h2, h3, h4, h5]
