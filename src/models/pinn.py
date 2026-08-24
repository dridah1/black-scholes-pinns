import torch.nn as nn


class BlackScholesPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Black-Scholes equation.
    """

    def __init__(self, input_dim=2, hidden_dim=50, num_hidden_layers=9):
        super().__init__()

        layers = [
            nn.BatchNorm1d(input_dim)
        ]

        for _ in range(num_hidden_layers):
            layers.append(
                nn.Linear(
                    hidden_dim if _ > 0 else input_dim,
                    hidden_dim
                )
            )
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
