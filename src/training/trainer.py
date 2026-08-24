import torch

from src.training.losses import total_loss


class PINNTrainer:
    """
    Trainer for the Black-Scholes Physics-Informed Neural Network.
    """

    def __init__(
        self,
        model,
        optimizer,
        x_collocation,
        x_terminal,
        x_boundary,
        sigma,
        r,
        K,
        dividend=0.0,
        pde_weight=100.0,
        terminal_weight=1.0,
        boundary_weight=1.0,
        device="cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.x_collocation = x_collocation.to(device)
        self.x_terminal = x_terminal.to(device)
        self.x_boundary = x_boundary.to(device)

        self.sigma = sigma
        self.r = r
        self.K = K
        self.dividend = dividend

        self.pde_weight = pde_weight
        self.terminal_weight = terminal_weight
        self.boundary_weight = boundary_weight

        self.device = device

        self.history = {
            "total_loss": [],
            "pde_loss": [],
            "terminal_loss": [],
            "boundary_loss": []
        }

    def train_step(self):
        """
        Perform one optimization step.
        """

        self.model.train()

        self.optimizer.zero_grad()

        (
            loss,
            loss_f,
            loss_terminal,
            loss_boundary
        ) = total_loss(
            model=self.model,
            x_collocation=self.x_collocation,
            x_terminal=self.x_terminal,
            x_boundary=self.x_boundary,
            sigma=self.sigma,
            r=self.r,
            K=self.K,
            dividend=self.dividend,
            pde_weight=self.pde_weight,
            terminal_weight=self.terminal_weight,
            boundary_weight=self.boundary_weight
        )

        loss.backward()

        self.optimizer.step()

        return (
            loss.item(),
            loss_f.item(),
            loss_terminal.item(),
            loss_boundary.item()
        )

    def train(self, epochs):
        """
        Train the PINN for a specified number of epochs.
        """

        for epoch in range(epochs):

            (
                loss,
                loss_f,
                loss_terminal,
                loss_boundary
            ) = self.train_step()

            self.history["total_loss"].append(loss)
            self.history["pde_loss"].append(loss_f)
            self.history["terminal_loss"].append(loss_terminal)
            self.history["boundary_loss"].append(loss_boundary)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"Total: {loss:.6e} | "
                    f"PDE: {loss_f:.6e} | "
                    f"Terminal: {loss_terminal:.6e} | "
                    f"Boundary: {loss_boundary:.6e}"
                )

        return self.history
