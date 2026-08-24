import torch


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Networks.
    """

    def __init__(
        self,
        model,
        optimizer,
        device="cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self):
        """
        Perform one optimization step.

        The complete loss function will be added
        after we separate the PDE, boundary, and
        terminal-condition losses.
        """

        self.optimizer.zero_grad()

        # Loss calculation will be added here.

        raise NotImplementedError
