import numpy as np

class ImplicitEu:
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
        A = np.zeros((self.M - 1, self.M - 1))

        for i in range(1, self.M):
            a = 0.5 * dt * (self.sigma ** 2 * i ** 2 - self.r * i)
            b = 1 + dt * (self.sigma ** 2 * i ** 2 + self.r)
            c = 0.5 * dt * (self.sigma ** 2 * i ** 2 + self.r * i)
            if i > 1:
                A[i - 1, i - 2] = -a
            A[i - 1, i - 1] = b
            if i < self.M - 1:
                A[i - 1, i] = -c

        for _ in range(self.N):
            rhs = V[1:-1]
            V_inner = np.linalg.solve(A, rhs)
            V[1:-1] = V_inner
            V[0] = 0 if self.is_call else self.K * np.exp(-self.r * (_ * dt))
            V[-1] = self.Smax - self.K * np.exp(-self.r * (_ * dt)) if self.is_call else 0

        self.grid = np.tile(V[:, None], (1, self.N + 1))
