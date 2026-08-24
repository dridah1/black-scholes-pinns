"""
Main entry point for the Black-Scholes PINN project.
"""


def main():

    # =========================
    # Black-Scholes parameters
    # =========================

    K = 100.0              # Strike price
    r = 0.05               # Risk-free interest rate
    sigma = 0.20           # Volatility
    dividend = 0.0         # Dividend yield
    T = 1.0                # Maturity

    # =========================
    # Computational domain
    # =========================

    S_min = 0.0
    S_max = 200.0

    t_min = 0.0
    t_max = T

    # =========================
    # Training parameters
    # =========================

    num_collocation = 5000
    num_terminal = 1000
    num_boundary = 1000

    epochs = 5000
    learning_rate = 1e-3

    print("Black-Scholes PINN experiment")
    print(f"Strike price: {K}")
    print(f"Volatility: {sigma}")
    print(f"Risk-free rate: {r}")
    print(f"Dividend yield: {dividend}")
    print(f"Maturity: {T}")


if __name__ == "__main__":
    main()
