class Animal:
    def __init__(self, E_G, P_Am, v, P_M, kappa, k_J, k_R, E_Hb, E_Hp):
        self.E_G = E_G  # Specific cost for structure (J/cm^3)
        self.P_Am = P_Am  # Surface-specific maximum assimilation rate (J/d.cm^2)
        self.v = v  # Energy conductance (cm/d)
        self.P_M = P_M  # Volume-specific somatic maintenance rate (J/d.cm^3)
        self.kappa = kappa  # Allocation to soma (-)
        self.k_J = k_J  # Maturity maintenance rate coefficient (d^-1)
        self.k_R = k_R  # Reproduction efficiency (-)
        self.E_Hb = E_Hb  # Maturity at birth (J)
        self.E_Hp = E_Hp  # Maturity at puberty (J)
        self.E_0 = 1e6  # Initial reserve (J)

    @property
    def E_m(self):
        return self.P_Am / self.v

    @property
    def g(self):
        return self.E_G / (self.kappa * self.E_m)

    @property
    def k_M(self):
        return self.P_M / self.E_G


animals = {
    'shark': dict(E_G=5212.32, P_Am=558.824, v=0.02774, P_M=34.3632, kappa=0.84851, k_J=0.002, k_R=0.95, E_Hb=7096,
                  E_Hp=300600),
    'squalus_acanthias': dict(E_G=5212.32, P_Am=558.824, v=0.02774, P_M=34.3632, kappa=0.84851, k_J=0.002, k_R=0.95,
                              E_Hb=7096, E_Hp=300600)
}
