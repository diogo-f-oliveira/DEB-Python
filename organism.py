import numpy as np


class Organism:
    def __init__(self, E_G, P_Am, v, P_M, kappa, k_J, kap_R, E_Hb, E_Hp, kap_X=0.8, kap_P=0.1, E_0=1e6, mu_X=525_000,
                 mu_E=550_000, mu_P=480_000, d_V=0.2, w_V=23.9295, **additional_parameters):
        self.E_G = E_G  # Specific cost for structure (J/cm^3)
        self.P_Am = P_Am  # Surface-specific maximum assimilation rate (J/d.cm^2)
        self.v = v  # Energy conductance (cm/d)
        self.P_M = P_M  # Volume-specific somatic maintenance rate (J/d.cm^3)
        self.kappa = kappa  # Allocation to soma (-)
        self.k_J = k_J  # Maturity maintenance rate coefficient (d^-1)
        self.kap_R = kap_R  # Reproduction efficiency (-)
        self.E_Hb = E_Hb  # Maturity at birth (J)
        self.E_Hp = E_Hp  # Maturity at puberty (J)
        self.E_0 = E_0  # Initial reserve (J)
        self.kap_X = kap_X  # Digestion efficiency (-)
        self.kap_P = kap_P  # Defecation efficiency (-)
        self.mu_X = mu_X
        self.mu_E = mu_E
        self.mu_P = mu_P
        self.d_V = d_V
        self.w_V = w_V

        self.nM = np.array([[1, 0, 0, 1], [0, 2, 0, 2], [2, 1, 2, 1], [0, 0, 0, 2]])
        self.nO = np.array([[1, 1, 1, 1], [1.8, 1.8, 1.8, 1.8], [0.5, 0.5, 0.5, 0.5], [0.15, 0.15, 0.15, 0.15]])

        for name, value in additional_parameters.items():
            setattr(self, name, value)

    @property
    def E_m(self):
        return self.P_Am / self.v

    @property
    def g(self):
        return self.E_G / (self.kappa * self.E_m)

    @property
    def k_M(self):
        return self.P_M / self.E_G

    @property
    def eta_O(self):
        return np.array([[-1 / (self.kap_X * self.mu_X), 0, 0],
                         [0, 0, self.d_V / (self.E_G * self.w_V)],
                         [1 / self.mu_E, -1 / self.mu_E, -1 / self.mu_E],
                         [self.kap_P / (self.mu_P * self.kap_X), 0, 0]])

    @property
    def eta_M(self):
        return -np.linalg.inv(self.nM) @ self.nO @ self.eta_O


animals = {
    'shark': dict(E_G=5212.32, P_Am=558.824, v=0.02774, P_M=34.3632, kappa=0.84851, k_J=0.002, kap_R=0.95, E_Hb=7096,
                  E_Hp=300600, E_0=174_619),
    'muskox': dict(E_G=7842.44, P_Am=1053.62, v=0.13958, P_M=18.4042, kappa=0.82731, k_J=0.00087827, kap_R=0.95,
                   E_Hb=1.409e+7, E_Hp=3.675e+8, E_Hx=5.136e+07, t_0=18.2498, f_milk=1)
}
