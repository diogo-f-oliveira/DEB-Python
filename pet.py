import numpy as np
from math import exp


class Pet:
    """
    class Pet:

    Stores all DEB theory parameters of an organism.
    Includes methods defined as properties for compound parameters (parameters that are function of core parameters).
    Rate parameters are dependent on temperature and their returned value is multiplied by the temperature correction
    factor TC.
    Units:
        Energy       -> J (Joule)
        Length       -> cm (centimeter)
        Time         -> d (day)
        Temperature  -> K (Kelvin)
        Mass         -> g (grams)
        Nº Molecules -> mol (moles)
    """
    temperature_affected = ('P_Am', 'v', 'P_M', 'P_T', 'k_J')

    def __init__(self, E_G, P_Am, v, P_M, kappa, k_J, kap_R, E_Hb, E_Hp, P_T=0, kap_X=0.8, kap_P=0.1, E_0=1e6,
                 V_0=1e-12, mu_X=525_000, mu_E=550_000, mu_P=480_000, d_V=0.2, w_V=23.9295, T_A=8000, T_ref=293.15,
                 T=310.85, **additional_parameters):
        self.E_G = E_G  # Specific cost for Structure (J/cm^3)
        self._P_Am = P_Am  # Surface-specific maximum assimilation rate (J/d.cm^2)
        self._v = v  # Energy conductance (cm/d)
        self._P_M = P_M  # Volume-specific somatic maintenance rate (J/d.cm^3)
        self._P_T = P_T  # Surface-specific somatic maintenance rate (J/d.cm^3)
        self.kappa = kappa  # Allocation to soma (-)
        self._k_J = k_J  # Maturity maintenance rate coefficient (d^-1)
        self.kap_R = kap_R  # Reproduction efficiency (-)
        self.E_Hb = E_Hb  # Maturity at birth (J)
        self.E_Hp = E_Hp  # Maturity at puberty (J)
        self.E_0 = E_0  # Initial Reserve (J)
        self.V_0 = V_0  # Initial Structure (cm^3)
        self.kap_X = kap_X  # Digestion efficiency (-)
        self.kap_P = kap_P  # Defecation efficiency (-)
        self.T_A = T_A  # Arrhenius temperature (K)
        self.T_ref = T_ref  # Reference temperature (K)
        self.T = T  # Temperature correction factor (K)
        self.mu_X = mu_X  # Chemical potential of Food (J/mol)
        self.mu_E = mu_E  # Chemical potential of Reserve (J/mol)
        self.mu_P = mu_P  # Chemical potential of Faeces (J/mol)
        self.d_V = d_V  # Specific density of Structure (g/cm^3)
        self.w_V = w_V  # Molecular weight of Structure (g/mol)

        # Chemical indices of mineral compounds
        self.nM = np.array([[1, 0, 0, 1], [0, 2, 0, 2], [2, 1, 2, 1], [0, 0, 0, 2]])
        # Chemical indices of organic compounds
        self.nO = np.array([[1, 1, 1, 1], [1.8, 1.8, 1.8, 1.8], [0.5, 0.5, 0.5, 0.5], [0.15, 0.15, 0.15, 0.15]])

        # Set any extra parameters that are not required for the STD model
        for name, value in additional_parameters.items():
            setattr(self, name, value)

    @property
    def E_m(self):
        """Computes the maximum energy density (J/cm^3)."""
        return self.P_Am / self.v

    @property
    def g(self):
        """Computes the energy investment ratio (-)."""
        return self.E_G / (self.kappa * self.E_m)

    @property
    def k_M(self):
        """Computes the somatic maintenance rate coefficient (d^-1)."""
        return self._P_M / self.E_G * self.TC

    @property
    def eta_O(self):
        """Computes the matrix of coefficients that couple mass fluxes of organic compounds to energy fluxes."""
        return np.array([[-1 / (self.kap_X * self.mu_X), 0, 0],
                         [0, 0, self.d_V / (self.E_G * self.w_V)],
                         [1 / self.mu_E, -1 / self.mu_E, -1 / self.mu_E],
                         [self.kap_P / (self.mu_P * self.kap_X), 0, 0]])

    @property
    def eta_M(self):
        """Computes the  matrix of coefficients that couple mass fluxes of mineral compounds to energy fluxes."""
        return -np.linalg.inv(self.nM) @ self.nO @ self.eta_O

    @property
    def TC(self):
        """Computes the temperature correction factor TC (-)."""
        return exp(self.T_A / self.T_ref - self.T_A / self.T)

    def __getattr__(self, item):
        """Returns the value of temperature affected parameters corrected with the temperature correction factor TC.
        Has no effect on other parameters."""
        if item in self.temperature_affected:
            return getattr(self, f'_{item}') * self.TC
        else:
            raise AttributeError  # Ensures that the behaviour for undefined parameters works as expected


# Dictionary with parameters for several organisms. Usage with Pet class is: Pet(**animals[pet_name])
animals = {
    'shark': dict(E_G=5212.32, P_Am=558.824, v=0.02774, P_M=34.3632, kappa=0.84851, k_J=0.002, kap_R=0.95, E_Hb=7096,
                  E_Hp=300600, E_0=174_619, T=282.15),
    'muskox': dict(E_G=7842.44, P_Am=1053.62, v=0.13958, P_M=18.4042, kappa=0.82731, k_J=0.00087827, kap_R=0.95,
                   E_Hb=1.409e+7, E_Hp=3.675e+8, E_Hx=5.136e+7, t_0=18.2498, f_milk=1, T=310.85),
    'human': dict(E_G=7879.55, P_Am=118.992, v=0.031461, P_M=2.5826, kappa=0.78656, k_J=0.00026254, kap_R=0.95,
                  E_Hb=4.81e+6, E_Hp=8.726e+7, E_Hx=1.346e+7, t_0=26.8217, f_milk=1)
}

if __name__ == '__main__':
    muskox = Pet(**animals['muskox'])
