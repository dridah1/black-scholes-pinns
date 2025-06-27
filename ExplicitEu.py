import numpy as np

class ExplicitEu:
    def __init__(self, S0, K, r, T, sigma, Smax, M, N, is_call=True):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.Smax = Smax
        self.M = M
        self.N = N
        self.is_call = is_call
        self.grid = None

    def payoff(self, S):
        return np.maximum(S - self.K, 0) if self.is_call else np.maximum(self.K - S, 0)

    def price(self):
        dt = self.T / self.N
        dS = self.Smax / self.M
        S = np.linspace(0, self.Smax, self.M + 1)
        V = self.payoff(S)
        V_old = V.copy()

        j = np.arange(1, self.M)
        alpha = 0.5 * dt * ((self.sigma ** 2) * j**2 - self.r * j)
        beta = 1 - dt * ((self.sigma ** 2) * j**2 + self.r)
        gamma = 0.5 * dt * ((self.sigma ** 2) * j**2 + self.r * j)

        for _ in range(self.N):
            V_new = V_old.copy()
            for i in j:
                V_new[i] = alpha[i - 1] * V_old[i - 1] + beta[i - 1] * V_old[i] + gamma[i - 1] * V_old[i + 1]

            # Boundary conditions
            V_new[0] = 0 if self.is_call else self.K * np.exp(-self.r * (_ * dt))
            V_new[-1] = self.Smax - self.K * np.exp(-self.r * (_ * dt)) if self.is_call else 0
            V_old = V_new

        self.grid = np.tile(V_old[:, None], (1, self.N + 1))
