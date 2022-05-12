import numpy as np
from math import exp


class Pet:
    # TODO: Better input structure of parameters
    # TODO: Include chemical variables as inputs
    # TODO: __str__ method for Pet description
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
                 V_0=1e-12, mu_X=525_000, mu_V=500_000, mu_E=550_000, mu_P=480_000, d_V=0.2, w_V=23.9295, T_A=8000,
                 T_ref=293.15,
                 T=310.85, n_waste=(1, 2, 1, 2), **additional_parameters):
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
        self.mu_X = mu_X  # Chemical potential of Food X (J/mol)
        self.mu_V = mu_V  # Chemical potential of Structure V (J/mol)
        self.mu_E = mu_E  # Chemical potential of Reserve E (J/mol)
        self.mu_P = mu_P  # Chemical potential of Feces P (J/mol)
        self.d_V = d_V  # Specific density of Structure (g/cm^3)
        self.w_V = w_V  # Molecular weight of Structure (g/mol)

        # Chemical indices of mineral compounds (CO2, H20, O2, N-Waste)
        self.nM = np.array([[1, 0, 0, n_waste[0]],
                            [0, 2, 0, n_waste[1]],
                            [2, 1, 2, n_waste[2]],
                            [0, 0, 0, n_waste[3]]])
        # Chemical indices of organic compounds (X, V, E, P)
        self.nO = np.array([[1, 1, 1, 1],
                            [1.8, 1.8, 1.8, 1.8],
                            [0.5, 0.5, 0.5, 0.5],
                            [0.15, 0.15, 0.15, 0.15]])

        # Set any extra parameters that are not required for the STD model
        for name, value in additional_parameters.items():
            setattr(self, name, value)

    @property
    def E_m(self):
        """Maximum energy density (J/cm^3)."""
        return self.P_Am / self.v

    @property
    def g(self):
        """Energy investment ratio (-)."""
        return self.E_G / (self.kappa * self.E_m)

    @property
    def k_M(self):
        """Somatic maintenance rate coefficient k_M (d^-1)."""
        return self._P_M / self.E_G * self.TC

    @property
    def k(self):
        """Maintenance ratio (-)."""
        return self._k_J / self._k_M

    @property
    def L_m(self):
        """Maximum length L_m (cm)."""
        return self.kappa * self._P_Am / self._P_M

    @property
    def kap_G(self):
        """Growth efficiency (-)."""
        return self.mu_V * self.M_V / self.E_G

    @property
    def M_V(self):
        """Volume-specific mass of structure (mol/cm^3)."""
        return self.d_V / self.w_V

    @property
    def p_Xm(self):
        """Specific maximum ingestion rate (J/d.cm^2)"""
        return self.p_Am / self.kap_X

    @property
    def eta_O(self):
        """Computes the matrix of coefficients that couple mass fluxes of organic compounds to energy fluxes."""
        return np.array([[-1 / (self.kap_X * self.mu_X), 0, 0],
                         [0, 0, self.d_V / (self.E_G * self.w_V)],
                         [1 / self.mu_E, -1 / self.mu_E, -1 / self.mu_E],
                         [self.kap_P / (self.mu_P * self.kap_X), 0, 0]])

    @property
    def eta_M(self):
        """Computes the matrix of coefficients that couple mass fluxes of mineral compounds to energy fluxes."""
        return -np.linalg.inv(self.nM) @ self.nO @ self.eta_O

    @property
    def TC(self):
        """Temperature correction factor TC (-)."""
        return exp(self.T_A / self.T_ref - self.T_A / self.T)

    def check_validity(self):
        """
        Checks that the parameters of the Pet are within the allowable part of the parameter space of the standard DEB
        model.
        :return: true if the parameters are valid, false otherwise
        """
        # All parameters must be positive
        if self.kap_P < 0 or self.kap_X < 0 or self._P_M < 0 or self._P_Am < 0 or self._v < 0 or self._P_T < 0 or \
                self.kappa < 0 or self.E_G < 0 or self._k_J < 0 or self.E_Hb < 0 or self.E_Hp < 0 or self.kap_R < 0 or \
                self.T_A < 0:
            return False
        # Maturity at puberty must be higher than maturity at birth
        if self.E_Hb >= self.E_Hp:
            return False
        # Efficiencies must be lower than one
        if self.kap_X >= 1 or self.kap_P >= 1 or self.kap_R >= 1 or self.kap_G >= 1 or self.kappa >= 1:
            return False
        # Constraint to reach puberty
        if (1 - self.kappa) * self._p_Am * (self.L_m ** 2) > self._k_J * self.E_Hp:
            return False
        return True

    def __getattr__(self, item):
        """Returns the value of temperature affected parameters corrected with the temperature correction factor TC.
        Has no effect on other parameters."""
        if item in self.temperature_affected:
            return getattr(self, f'_{item}') * self.TC
        else:
            raise AttributeError  # Ensures that the behaviour for undefined parameters works as expected

    def __str__(self):
        for name, value in self.__dict__.items():
            print(f"{name}: {value}")


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
