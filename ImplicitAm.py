import numpy as np

class ImplicitAmBer:
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

        alpha = 0.5 * dt * ((self.sigma ** 2) * (np.arange(self.M + 1) ** 2) - self.r * np.arange(self.M + 1))
        beta = 1 + dt * ((self.sigma ** 2) * (np.arange(self.M + 1) ** 2) + self.r)
        gamma = 0.5 * dt * ((self.sigma ** 2) * (np.arange(self.M + 1) ** 2) + self.r * np.arange(self.M + 1))

        A = np.zeros((self.M - 1, self.M - 1))
        for i in range(1, self.M):
            if i > 1:
                A[i - 1, i - 2] = -alpha[i]
            A[i - 1, i - 1] = beta[i]
            if i < self.M - 1:
                A[i - 1, i] = -gamma[i]

        for t in reversed(range(self.N)):
            rhs = V[1:-1]
            if self.is_call:
                rhs[-1] += gamma[self.M - 1] * (self.Smax - self.K * np.exp(-self.r * (self.T - t * dt)))
            else:
                rhs[0] += alpha[1] * (self.K * np.exp(-self.r * (self.T - t * dt)))

            V_new = np.linalg.solve(A, rhs)
            V[1:-1] = np.maximum(V_new, self.payoff(S[1:-1]))

        self.grid = np.tile(V[:, None], (1, self.N + 1))
