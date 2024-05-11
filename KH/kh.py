import numpy as np
from .objective_function import ObjectiveFunction
import numpy as np

class KrillHerd:
    def __init__(self, NK, MI, dim, lb, ub):
        self.NK = NK
        self.MI = MI
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.X = np.random.rand(NK, dim) * (ub - lb) + lb
        self.K = np.array([ObjectiveFunction.evaluate(x) for x in self.X])
        self.Kib = np.copy(self.K)
        self.Xib = np.copy(self.X)
        self.Kgb = np.min(self.K)
        self.Xgb = self.X[np.argmin(self.K)]
        self.RR = np.zeros((dim, NK))

    def update_food_location(self):
        Sf = np.zeros(self.dim)
        Xf = np.zeros((self.dim, self.NK))
        for i in range(self.dim):
            for j in range(self.NK):
                Sf[i] += self.X[j][i] / self.K[j]
            Xf[i] = Sf[i] / self.NK
        return Xf

    def evaluate_fitness(self, X):
        Kf = np.zeros(self.NK)
        for i in range(self.NK):
            for j in range(self.dim):
                if X[i][j] > self.ub[j]:
                    X[i][j] = self.ub[j]
                if X[i][j] < self.lb[j]:
                    X[i][j] = self.lb[j]
            Kf[i] = ObjectiveFunction.evaluate(X[i])
        return Kf

    def update_positions(self, Kf, Xf):
        for i in range(self.NK):
            if self.MI >= 2 and Kf[self.MI-1] < Kf[self.MI]:
                Xf[self.MI][i] = Xf[self.MI-1][i]
        return Xf

    def calculate_parameters(self, Xf, Kf):
        Kw_Kgb = np.max(self.K) - self.Kgb
        w = 0.1 + 0.8 * (1 - self.MI / self.MI)
        Nmax = 1.0
        Vf = 1.0
        Food_multiplier = np.zeros(self.dim)
        Food_multiplier_trans = np.zeros(self.dim)
        Sum_attraction_multipliers = np.zeros(self.dim)
        alpha_n = np.zeros(self.dim)
        for i in range(self.NK):
            alpha_b_numerator = 0
            for j in range(self.dim):
                Rf = Xf[j][self.MI] - self.X[i][j]
                Rgb = self.Xgb[j] - self.X[i][j]
                alpha_b_numerator += (Rgb * Rgb)
                Food_multiplier_trans[j] = Rf * Rf
                Food_multiplier[j] += Food_multiplier_trans[j]
                Sum_attraction_multipliers[j] += (self.X[j][i] - self.X[i][j]) ** 2
            if self.Kgb < self.K[i]:
                alpha_b = -2 * (1 + np.random.rand() * (self.MI / self.MI)) * (self.Kgb - self.K[i]) / Kw_Kgb / np.sqrt(alpha_b_numerator) * Rgb
            else:
                alpha_b = 0
            nn = 0
            ds = np.sum(self.RR) / (5 * self.NK)
            for n in range(self.NK):
                condition = self.RR < ds
                nn += 1 if np.sum(condition) == self.NK and n != i else 0
                if nn <= 4 and self.K[i] != self.K[n]:
                    for j in range(self.dim):
                        RR_multiplier[j] = self.RR[j][n]
                    alpha_n -= (self.K[n] - self.K[i]) / Kw_Kgb / np.sqrt(self.RR) * RR_multiplier
        return w, Nmax, Vf, Food_multiplier, Sum_attraction_multipliers, alpha_n
