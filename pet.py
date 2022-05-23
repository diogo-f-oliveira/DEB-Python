import numpy as np
from math import exp
from composition import Composition


class Pet:
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
    temperature_affected = ('p_Am', 'v', 'p_M', 'p_T', 'k_J')

    def __init__(self, E_G, p_Am, v, p_M, kappa, k_J, kap_R, E_Hb, E_Hp, comp=None, p_T=0, kap_X=0.8, kap_P=0.1,
                 E_0=1e6, V_0=1e-12, T_A=8000, T_ref=293.15, T=298.15, del_M=1, **additional_parameters):
        self.E_G = E_G  # Specific cost for Structure (J/cm^3)
        self._p_Am = p_Am  # Surface-specific maximum assimilation rate (J/d.cm^2)
        self._v = v  # Energy conductance (cm/d)
        self._p_M = p_M  # Volume-specific somatic maintenance rate (J/d.cm^3)
        self._p_T = p_T  # Surface-specific somatic maintenance rate (J/d.cm^3)
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
        self.del_M = del_M  # Shape coefficient (-)

        # Chemical composition
        if comp is None:
            self.comp = Composition()
        elif isinstance(comp, (list, tuple)):
            self.comp = Composition(*comp)
        elif isinstance(comp, dict):
            self.comp = Composition(**comp)
        elif isinstance(comp, Composition):
            self.comp = comp
        else:
            raise Exception("Invalid Composition input.")

        # Set any extra parameters that are not required for the STD model
        for name, value in additional_parameters.items():
            setattr(self, name, value)

    def check_validity(self):
        # TODO: return or print the reason for invalidity
        # TODO: test invalid params
        """
        Checks that the parameters of the Pet are within the allowable part of the parameter space of the standard DEB
        model.
        :return: true if the parameters are valid, false otherwise
        """
        # All parameters must be positive
        if self.kap_P < 0 or self.kap_X < 0 or self._p_M < 0 or self._p_Am < 0 or self._v < 0 or self._p_T < 0 or \
                self.kappa < 0 or self.E_G < 0 or self._k_J < 0 or self.E_Hb < 0 or self.E_Hp < 0 or self.kap_R < 0 or \
                self.T_A < 0:
            return False
        # Maturity at puberty must be higher than maturity at birth
        if self.E_Hb >= self.E_Hp:
            return False
        # Efficiencies must be lower than one
        if self.kap_X >= 1 or self.kap_P >= 1 or self.kap_R >= 1 or self.kap_G >= 1 or self.kappa >= 1:
            return False
        # Constraint to reach birth
        if (1 - self.kappa) * self._p_Am * (self.L_m ** 2) <= self._k_J * self.E_Hb:
            return False
        # Constraint to reach puberty
        if (1 - self.kappa) * self._p_Am * (self.L_m ** 2) <= self._k_J * self.E_Hp:
            return False
        # Supply stress outside the supply-demand spectrum
        if self.s_s >= 4/27:
            return False
        return True

    def convert_to_physical_length(self, V):
        """
        Converts a structure value to physical length using the shape coefficient del_M.
        :param V: structure
        :return: physical length
        """
        return V ** (1 / 3) / self.del_M

    def __getattr__(self, item):
        """Returns the value of temperature affected parameters corrected with the temperature correction factor TC.
        Has no effect on other parameters."""
        if item in self.temperature_affected:
            return getattr(self, f'_{item}') * self.TC
        else:
            raise AttributeError  # Ensures that the behaviour for undefined parameters works as expected

    def __str__(self):
        # for name, value in self.__dict__.items():
        #     print(f"{name}: {value}")
        return

    def aggregated_chemical_reactions(self):
        """
        Returns a dictionary with the aggregated chemical reactions complete with stoichiometry coefficients.
        :return: dictionary with the three aggregated chemical reactions as strings
        """
        assimilation = f'{self.gamma_O[0, 0]:.4} X + {self.gamma_M[2, 0]:.4} O2 -> E + {-self.gamma_M[0, 0]:.4} CO2 ' \
                       f'+ {-self.gamma_M[1, 0]:.4} H20 + {-self.gamma_M[3, 0]:.4} {self.comp.N.chemical_formula} + ' \
                       f'{-self.gamma_O[3, 0]:.4} P'

        dissipation = f'E + {self.gamma_M[2, 1]:.4} O2 ->  {-self.gamma_M[0, 1]:.4} CO2 + {-self.gamma_M[1, 1]:.4} H20 ' \
                      f'+ {-self.gamma_M[3, 1]:.4} {self.comp.N.chemical_formula}'

        growth = f'E + {self.gamma_M[2, 2]:.4} O2 ->  {-self.gamma_O[1, 2]:.4} V + {-self.gamma_M[0, 2]:.4} CO2 + ' \
                 f'{-self.gamma_M[1, 2]:.4} H20 + {-self.gamma_M[3, 2]:.4} {self.comp.N.chemical_formula}'

        return {'assimilation': assimilation, 'dissipation': dissipation, 'growth': growth}

    def print_reactions(self):
        reactions = self.aggregated_chemical_reactions()

        for reaction_name, formula in reactions.items():
            print(f"{reaction_name.capitalize()}: {formula}")

    @property
    def E_m(self):
        """Maximum energy density (J/cm^3)."""
        return self._p_Am / self._v

    @property
    def g(self):
        """Energy investment ratio (-)."""
        return self.E_G / (self.kappa * self.E_m)

    @property
    def k_M(self):
        """Somatic maintenance rate coefficient k_M (d^-1)."""
        return self._p_M / self.E_G * self.TC

    @property
    def k(self):
        """Maintenance ratio (-)."""
        return self._k_J / self._k_M

    @property
    def L_m(self):
        """Maximum length L_m (cm)."""
        return self.kappa * self._p_Am / self._p_M

    @property
    def kap_G(self):
        """Growth efficiency (-)."""
        return self.comp.V.mu * self.M_V / self.E_G

    @property
    def M_V(self):
        """Volume-specific mass of structure (mol/cm^3)."""
        return self.comp.V.d / self.comp.V.w

    @property
    def p_Xm(self):
        """Specific maximum ingestion rate (J/d.cm^2)"""
        return self.p_Am / self.kap_X

    @property
    def eta_O(self):
        # TODO: Rewrite using the yield coefficients for better understanding
        """Computes the matrix of coefficients that couple mass fluxes of organic compounds to energy fluxes."""
        return np.array([[-1 / (self.kap_X * self.comp.X.mu), 0, 0],
                         [0, 0, self.comp.V.d / (self.E_G * self.comp.V.w)],
                         [1 / self.comp.E.mu, -1 / self.comp.E.mu, -1 / self.comp.E.mu],
                         [self.kap_P / (self.comp.P.mu * self.kap_X), 0, 0]])

    @property
    def eta_M(self):
        """Computes the matrix of coefficients that couple mass fluxes of mineral compounds to energy fluxes."""
        return -np.linalg.inv(self.comp.n_M) @ self.comp.n_O @ self.eta_O

    @property
    def TC(self):
        """Temperature correction factor TC (-)."""
        return exp(self.T_A / self.T_ref - self.T_A / self.T)

    @property
    def y_XE(self):
        """Yield of food on reserve (-)."""
        return self.comp.E.mu / (self.comp.X.mu * self.kap_X)

    @property
    def y_PE(self):
        """Yield of feces on reserve (-)."""
        return self.y_XE * self.kap_P * self.comp.X.mu / self.comp.P.mu

    @property
    def y_VE(self):
        """Yield of structure on reserve (-)."""
        return self.comp.E.mu * self.M_V / self.E_G

    @property
    def y_PX(self):
        """Yield of feces on food (-)."""
        return self.kap_P * self.comp.X.mu / self.comp.P.mu

    @property
    def gamma_O(self):
        """Computes the matrix of stoichiometry coefficients for organic compounds in the assimilation, dissipation and
        growth aggregated chemical reactions."""
        return np.array([[self.y_XE, 0, 0],
                         [0, 0, -self.y_VE],
                         [-1, 1, 1],
                         [-self.y_PE, 0, 0]])

    @property
    def gamma_M(self):
        """Computes the matrix of stoichiometry coefficients for mineral compounds in the assimilation, dissipation and
        growth aggregated chemical reactions."""
        return -np.linalg.inv(self.comp.n_M) @ self.comp.n_O @ self.gamma_O

    @property
    def s_s(self):
        """Supply Stress (-)."""
        return self._k_J * self.E_Hp * (self._p_M ** 2) / (self._p_Am ** 3)


# Dictionary with parameters for several commom organisms. Usage with Pet class is: Pet(**animals[pet_name])
animals = {
    'shark': dict(E_G=5212.32, p_Am=558.824, v=0.02774, p_M=34.3632, kappa=0.84851, k_J=0.002, kap_R=0.95, E_Hb=7096,
                  E_Hp=300600, E_0=174_619, T=282.15),
    'muskox': dict(E_G=7842.44, p_Am=1053.62, v=0.13958, p_M=18.4042, kappa=0.82731, k_J=0.00087827, kap_R=0.95,
                   E_Hb=1.409e+7, E_Hp=3.675e+8, E_Hx=5.136e+7, t_0=18.2498, f_milk=1, T=310.85),
    'human': dict(E_G=7879.55, p_Am=118.992, v=0.031461, p_M=2.5826, kappa=0.78656, k_J=0.00026254, kap_R=0.95,
                  E_Hb=4.81e+6, E_Hp=8.726e+7, E_Hx=1.346e+7, t_0=26.8217, f_milk=1),
    'bos_taurus_alentejana': dict(E_G=8261.79, p_Am=2501.03, v=0.107224, p_M=42.2556, kappa=0.976264, k_J=0.0002,
                                  kap_R=0.95, E_Hb=2071229.972, E_Hp=30724119.81, E_Hx=15139260.45, t_0=109.4715964,
                                  f_milk=1, del_M=0.349222),
}

if __name__ == '__main__':
    muskox = Pet(**animals['muskox'])
