from .composition import Composition, RuminantComposition
from .state import State

import numpy as np
from math import exp
from copy import deepcopy


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
    temperature_affected = ('p_Am', 'v', 'p_M', 'p_T', 'k_J')
    T_REF = 293.15  # Reference temperature (K)

    def __init__(self, p_Am, kap, v, p_M, E_G, k_J, E_Hb, E_Hp, kap_R, state=None, comp=None, p_T=0, kap_X=0.8,
                 kap_P=0.1, T_typical=T_REF, E_0=1e6, V_0=1e-10, T_A=8000, del_M=1, **additional_parameters):

        # Parameters
        self.E_G = E_G  # Specific cost for Structure (J/cm^3)
        self._p_Am = p_Am  # Surface-specific maximum assimilation rate (J/d.cm^2)
        self._v = v  # Energy conductance (cm/d)
        self._p_M = p_M  # Volume-specific somatic maintenance rate (J/d.cm^3)
        self._p_T = p_T  # Surface-specific somatic maintenance rate (J/d.cm^2)
        self.kap = kap  # Allocation fraction to soma (-)
        self._k_J = k_J  # Maturity maintenance rate constant (d^-1)
        self.E_Hb = E_Hb  # Maturity at birth (J)
        self.E_Hp = E_Hp  # Maturity at puberty (J)
        self.kap_R = kap_R  # Reproduction efficiency (-)
        self.E_0 = E_0  # Initial Reserve (J)
        self.V_0 = V_0  # Initial Structure (cm^3)
        self.kap_X = kap_X  # Digestion efficiency (-)
        self.kap_P = kap_P  # Defecation efficiency (-)
        self.T_A = T_A  # Arrhenius temperature (K)
        self.T_typical = T_typical  # Typical body temperature (K)
        self.del_M = del_M  # Shape coefficient (-)

        # Model inputs are saved in State
        if state is None:
            state = State()
            state.T = T_typical
        self.state = state

        # Chemical composition
        if comp is None:
            self.comp = Composition(food=self.state.food_comp)
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
        """
        Checks that the parameters of the Pet are within the allowable part of the parameter space of the standard DEB
        model.
        :return: true if the parameters are valid, false otherwise
        """
        # All parameters must be positive
        if self.kap_P < 0 or self.kap_X < 0 or self._p_M < 0 or self._p_Am < 0 or self._v < 0 or self._p_T < 0 or \
                self.kap < 0 or self.E_G < 0 or self._k_J < 0 or self.E_Hb < 0 or self.E_Hp < 0 or self.kap_R < 0 or \
                self.T_A < 0:
            return False, "All parameters must be positive."
        # Maturity at puberty must be higher than maturity at birth
        if self.E_Hb >= self.E_Hp:
            return False, "Maturity at puberty must be higher than maturity at birth."
        # Efficiencies must be lower than one
        if self.kap_X >= 1 or self.kap_P >= 1 or self.kap_R >= 1 or self.kap_G >= 1 or self.kap >= 1:
            return False, "Efficiencies must be lower than one."
        return True, "All good!"

    def check_viability(self):
        """
        Checks that the organism is capable of reaching birth and puberty at maximum food level.
        :return: true if the organism is capable of reaching birth and puberty, false otherwise
        """
        # Constraint to reach birth
        if (1 - self.kap) * self._p_Am * (self.L_m ** 2) <= self._k_J * self.E_Hb:
            return False, "Impossible to reach birth."
        # Constraint to reach puberty
        if (1 - self.kap) * self._p_Am * (self.L_m ** 2) <= self._k_J * self.E_Hp:
            return False, "Impossible to reach puberty."
        # Supply stress outside the supply-demand spectrum
        if self.s_s >= 4 / 27:
            return False, "Supply stress outside the supply-demand spectrum."
        return True, "All good!"

    def compute_physical_length(self, V):
        """
        Converts a structure value to physical length using the shape coefficient del_M.
        :param V: scalar or vector of structure
        :return: scalar or vector of physical length
        """
        return V ** (1 / 3) / self.del_M

    @property
    def physical_length(self):
        return self.state.V ** (1 / 3) / self.del_M

    def compute_physical_volume(self, V, E, E_R):
        """
        Converts structure, reserve and reproduction buffer into physical volume.
        :param V: scalar or vector of structure
        :param E: scalar or vector of reserve
        :param E_R: scalar or vector of reproduction buffer
        :return: scalar or vector of physical volume
        """
        return V + (E + E_R) * self.comp.E.w / self.comp.E.d / self.comp.E.mu

    @property
    def physical_volume(self):
        E, V, E_R = self.state.E, self.state.V, self.state.E_R
        return V + (E + E_R) * self.comp.E.w / self.comp.E.d / self.comp.E.mu

    def compute_wet_weight(self, V, E, E_R):
        """
        Converts structure, reserve and reproduction buffer into wet weight_col. Assumes density of reserve d_V equal to 1
        :param V: scalar or vector of structure
        :param E: scalar or vector of reserve
        :param E_R: scalar or vector of reproduction buffer
        :return: scalar or vector of wet weight_col
        """
        # TODO: Compute wet weight with reproduction buffer (check DEB comments)
        # TODO: Check assumption of d_V = 1, seems like it is wet density of structure
        scaled_reserved_density = (E / V) / self.E_m
        return V * (1 + scaled_reserved_density * self.omega)

    @property
    def wet_weight(self):
        E, V, E_R = self.state.E, self.state.V, self.state.E_R
        scaled_reserved_density = (E / V) / self.E_m
        return V * (1 + scaled_reserved_density * self.omega)

    def compute_dry_weight(self, V, E, E_R):
        """
        Converts structure, reserve and reproduction buffer into dry_weight.
        :param V: scalar or vector of structure
        :param E: scalar or vector of reserve
        :param E_R: scalar or vector of reproduction buffer
        :return: scalar or vector of wet weight_col
        """
        return 1 * V + (E + E_R) * self.comp.E.w / self.comp.E.mu

    @property
    def dry_weight(self):
        E, V, E_R = self.state.E, self.state.V, self.state.E_R
        return 1 * V + (E + E_R) * self.comp.E.w / self.comp.E.mu

    def __getattr__(self, item):
        """Returns the value of temperature affected parameters corrected with the temperature correction factor TC.
        Has no effect on other parameters."""
        if item in self.temperature_affected:
            return getattr(self, f'_{item}') * self.TC
        else:
            raise AttributeError  # Ensures that the behaviour for undefined parameters works as expected

    # TODO: __setattr__ for temperature affected parameters

    def __str__(self):
        """Called when print(Pet) is called. Returns a description of the Pet, including the most relevant parameters
        and chemical reactions."""
        # TODO: Add symbols in front of description or replace
        description = f"Parameters at T={self.state.T} K\n" \
                      f"Surface-specific maximum assimilation rate: {self.p_Am:.6} (J/d.cm^2)\n" \
                      f"Allocation fraction to soma: {self.kap} (-)\n" \
                      f"Energy conductance: {self.v:.6} (cm/d)\n" \
                      f"Volume-specific somatic maintenance rate: {self.p_M:.6} (J/d.cm^3)\n" \
                      f"Specific cost for structure: {self.E_G} (J/cm^3)\n" \
                      f"Maturity maintenance rate constant: {self.k_J:.6} (d^-1)\n" \
                      f"Maturity at birth: {self.E_Hb} (J)\n" \
                      f"Maturity at puberty: {self.E_Hp} (J)\n" \
                      f"Reproduction efficiency: {self.kap_R} (-)\n\n" \
                      f"Chemical Reactions:\n"
        reactions = self.aggregated_chemical_reactions()

        for reaction_name, formula in reactions.items():
            description += f"{reaction_name.capitalize()}: {formula}\n"
        return description

    def aggregated_chemical_reactions(self):
        # TODO: Return equations formatted in LaTeX
        """
        Returns a dictionary with the aggregated chemical reactions complete with stoichiometry coefficients.
        :return: dictionary with the three aggregated chemical reactions as strings
        """
        reactions_right_side = []
        reactions_left_side = []
        for i in range(3):
            right = ''
            left = ''
            for y, symbol in zip(self.gamma_M[:, i], self.comp.mineral_symbols):
                if y < 0:
                    left += f'{-y:.4} {symbol} + '
                elif y > 0:
                    right += f'{y:.4} {symbol} + '
            reactions_left_side.append(left)
            reactions_right_side.append(right)

        assimilation = f'{-self.gamma_O[0, 0]:.4} X + {reactions_left_side[0][:-3]} -> ' \
                       f'E + {reactions_right_side[0][:-3]} + {self.gamma_O[3, 0]:.4} P'

        dissipation = f'E + {reactions_left_side[1][:-3]} -> {reactions_right_side[1][:-3]}'

        growth = f'E + {reactions_left_side[2][:-3]} ->  {self.gamma_O[1, 2]:.4} V + {reactions_right_side[2][:-3]}'

        return {'assimilation': assimilation, 'dissipation': dissipation, 'growth': growth}

    def print_reactions(self):
        """Prints the aggregated chemical reactions."""
        reactions = self.aggregated_chemical_reactions()

        for reaction_name, formula in reactions.items():
            print(f"{reaction_name.capitalize()}: {formula}")

    # TODO: scaled functional response based on p_X from state

    @property
    def E_m(self):
        """Maximum energy density (J/cm^3)."""
        return self._p_Am / self._v

    @property
    def g(self):
        """Energy investment ratio (-)."""
        return self.E_G / (self.kap * self.E_m)

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
        return self.kap * self._p_Am / self._p_M

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
        """Specific maximum ingestion rate (J/d.cm^2)."""
        return self.p_Am / self.kap_X

    @property
    def f(self):
        """Scaled functional response (-)."""
        return self.state.p_X / (self.state.V ** (2 / 3) * self.p_Xm)

    @property
    def TC(self):
        """Temperature correction factor TC (-)."""
        return exp(self.T_A / self.T_REF - self.T_A / self.state.T)

    def r_B(self, f):
        """Von Bertalanffy growth rate r_B (d^-1)."""
        return self.k_M / (1 + f / self.g) / 3

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
        return np.array([[-self.y_XE, 0, 0],
                         [0, 0, self.y_VE],
                         [1, -1, -1],
                         [self.y_PE, 0, 0]])

    @property
    def gamma_M(self):
        """Computes the matrix of stoichiometry coefficients for mineral compounds in the assimilation, dissipation and
        growth aggregated chemical reactions."""
        return -np.linalg.inv(self.comp.n_M) @ self.comp.n_O @ self.gamma_O

    @property
    def eta_O(self):
        """Computes the matrix of coefficients that couple mass fluxes of organic compounds to energy fluxes."""
        return self.gamma_O / self.comp.E.mu

    @property
    def eta_M(self):
        """Computes the matrix of coefficients that couple mass fluxes of mineral compounds to energy fluxes."""
        return self.gamma_M / self.comp.E.mu

    @property
    def s_s(self):
        """Supply Stress (-)."""
        return self._k_J * self.E_Hp * (self._p_M ** 2) / (self._p_Am ** 3)

    @property
    def L_T(self):
        """Heating Length (cm)."""
        return self._p_T / self._p_M

    def L_inf(self, f):
        """Ultimate Length (cm)."""
        return f * self.L_m - self.L_T

    @property
    def omega(self):
        """Contribution of ash free dry mass of reserve to total ash free dry biomass (-)"""
        return self.E_m * self.comp.E.w / self.comp.E.d / self.comp.E.mu

    @property
    def kap_N(self):
        return self.comp.N.mu / self.comp.X.mu * self.gamma_M[3, 0] / self.y_XE

    @property
    def kap_Q(self):
        return 1 - self.kap_X - self.kap_P - self.kap_N


# TODO: Update Pet class to use new Environment and State paradigm
class Ruminant(Pet):
    """
        class Ruminant:

        Child class of Pet.
        Assumes assimilation occurs in two sub transformations, one that produces CO2 and another that produces CH4. The
        assimilation reaction is an average of both sub transformations, weighted by the parameter xi_C.
        """

    def __init__(self, p_Am, kap, v, p_M, E_G, k_J, E_Hb, E_Hp, kap_R, xi_C, comp=None, p_T=0, kap_X=0.8,
                 kap_P=0.1, E_0=1e6, V_0=1e-12, T_A=8000, T_ref=293.15, T_typical=298.15, del_M=1,
                 **additional_parameters):

        super().__init__(p_Am, kap, v, p_M, E_G, k_J, E_Hb, E_Hp, kap_R, comp=comp, p_T=p_T, kap_X=kap_X, kap_P=kap_P,
                         E_0=E_0, V_0=V_0, T_A=T_A, T_ref=T_ref, T_typical=T_typical, del_M=del_M,
                         **additional_parameters)
        # Chemical composition
        if comp is None:
            self.comp = RuminantComposition()
        elif isinstance(comp, (list, tuple)):
            self.comp = RuminantComposition(*comp)
        elif isinstance(comp, dict):
            self.comp = RuminantComposition(**comp)
        elif isinstance(comp, RuminantComposition):
            self.comp = comp
        else:
            raise Exception("Invalid Composition input. Must be of class RuminantComposition.")

        self.xi_C = xi_C  # Rumination fraction (-)

    @property
    def gamma_M(self):
        """Computes the matrix of stoichiometry coefficients for mineral compounds in the assimilation, dissipation and
        growth aggregated chemical reactions."""
        gamma_M = np.pad(self.gamma_M_CO2, ((0, 1), (0, 0)))
        gamma_M[:, 0] = gamma_M[:, 0] * (1 - self.xi_C) + \
                        np.pad(self.gamma_M_CH4[:, 0], (1, 0)) * self.xi_C
        return gamma_M

    @property
    def gamma_M_CO2(self):
        """Computes the matrix of stoichiometry coefficients for mineral compounds in the assimilation, dissipation and
        growth aggregated chemical reactions assuming methane (CH4) production does not occur."""
        return -np.linalg.inv(self.comp.n_M[:, :-1]) @ self.comp.n_O @ self.gamma_O

    @property
    def gamma_M_CH4(self):
        """Computes the matrix of stoichiometry coefficients for mineral compounds in the assimilation, dissipation and
        growth aggregated chemical reactions assuming methane (CH4) production occurs instead of CO2."""
        return -np.linalg.inv(self.comp.n_M[:, 1:]) @ self.comp.n_O @ self.gamma_O

    @property
    def eta_M_CO2(self):
        """Computes the matrix of coefficients that couple mass fluxes of mineral compounds to energy fluxes, assuming
        methane (CH4) production does not occur."""
        return self.gamma_M_CO2 / self.comp.E.mu

    @property
    def eta_M_CH4(self):
        """Computes the matrix of coefficients that couple mass fluxes of mineral compounds to energy fluxes assuming
        methane (CH4) production occurs instead of CO2."""
        return self.gamma_M_CH4 / self.comp.E.mu

    @property
    def gamma_M_ext(self):
        """Computes the matrix of stoichiometry coefficients for mineral compounds in the assimilation, dissipation and
        growth aggregated chemical reactions, as well as the sub transformations of assimilation that produce CO2 and
        CH4."""
        gamma_M = np.zeros((5, 5))
        gamma_M[1:5, 1] = self.gamma_M_CH4
        gamma_M[0:4, 2:] = self.gamma_M_CO2
        gamma_M[:, 0] = gamma_M[:, 2] * (1 - self.xi_C) + gamma_M[:, 1] * self.xi_C
        return gamma_M

    @property
    def kap_M(self):
        return self.comp.M.mu / self.comp.X.mu * self.gamma_M[4, 0] / self.y_XE

    @property
    def kap_Q(self):
        return 1 - self.kap_X - self.kap_P - self.kap_N - self.kap_M


# Dictionary with parameters for several common organisms. Usage with Pet class is: Pet(**animals[pet_name])
animals = {
    'shark': dict(E_G=5212.32, p_Am=558.824, v=0.02774, p_M=34.3632, kap=0.84851, k_J=0.002, kap_R=0.95, E_Hb=7096,
                  E_Hp=300600, E_0=174_619, T_typical=282.15),
    'muskox': dict(E_G=7842.44, p_Am=1053.62, v=0.13958, p_M=18.4042, kap=0.82731, k_J=0.00087827, kap_R=0.95,
                   E_Hb=1.409e+7, E_Hp=3.675e+8, E_Hx=5.136e+7, t_0=18.2498, f_milk=1, T_typical=310.85),
    'human': dict(E_G=7879.55, p_Am=118.992, v=0.031461, p_M=2.5826, kap=0.78656, k_J=0.00026254, kap_R=0.95,
                  E_Hb=4.81e+6, E_Hp=8.726e+7, E_Hx=1.346e+7, t_0=26.8217, f_milk=1),
    'bos_taurus_alentejana': dict(E_G=8261.79, p_Am=2501.03, v=0.107224, p_M=42.2556, kap=0.976264, k_J=0.002,
                                  kap_R=0.95, E_Hb=2071229.972, E_Hp=30724119.81, E_Hx=15139260.45, t_0=109.4715964,
                                  f_milk=1, del_M=0.349222, kap_X=0.3, rum_fraction=0.3, T_typical=311.75),
    'bos_taurus_angus': dict(E_G=7960, p_Am=1584.86, v=0.08859, p_M=21.81, kap=0.9633,
                             k_J=0.002, kap_R=0.95, E_Hb=4.335e+06, E_Hp=2.608e+07, E_Hx=2.083e+07,
                             t_0=84.66, f_milk=1, del_M=0.4481, kap_X=0.09051, rum_fraction=0.3, T_typical=311.75),
    'bos_taurus_limousin': dict(E_G=8839.515768, p_Am=4927.102094, v=0.167427252, p_M=88.18065702, kap=0.9783141238,
                                k_J=0.002, kap_R=0.95, E_Hb=2234700.015, E_Hp=91160096.48, E_Hx=34941987.33,
                                t_0=188.2442861, f_milk=1, del_M=0.3686000612, kap_X=0.3, rum_fraction=0.3,
                                T_typical=311.75),
    'bos_taurus_charolais': dict(E_G=8885.233383, p_Am=4583.16235, v=0.06126486348, p_M=96.28243777, kap=0.9771477494,
                                 k_J=0.002, kap_R=0.95, E_Hb=2031342.633, E_Hp=26960202.1, E_Hx=17219252.94,
                                 t_0=88.84268928, f_milk=1, del_M=0.2688914292, kap_X=0.3, rum_fraction=0.3,
                                 T_typical=311.75),
    'bos_taurus_mertolenga': dict(E_G=8880, p_Am=1787, v=0.2361, p_M=24.16, kap=0.9528, k_J=0.002, kap_R=0.95,
                                  E_Hb=5.948e+06, E_Hp=1.467e+08, E_Hx=4.451e+07, t_0=229, f_milk=1, del_M=0.4,
                                  kap_X=0.1815, rum_fraction=0.3, T_typical=311.75),
    'sheep': dict(E_G=7838.53, p_Am=3473.31, v=0.18751, p_M=96.0278, kap=0.82933, k_J=0.002, kap_R=0.95,
                  E_Hb=4.154e+06, E_Hp=2.397e+08, E_Hx=6.797e+07, t_0=80.9564, f_milk=1, del_M=0.2688914292, kap_X=0.8,
                  rum_fraction=0.3, T_typical=311.75),
    'ovis_aries_lacaune': dict(E_G=7828, p_Am=2289.55, v=1.845, p_M=57.54, kap=0.955, k_J=0.002, kap_R=0.95,
                               E_Hb=1.656e+06, E_Hp=6.298e+07, E_Hx=6.345e+06, t_0=126.5, f_milk=1, del_M=0.2688914292,
                               kap_X=0.8, rum_fraction=0.3, T_typical=311.75),
    'Danio_rerio': dict(E_G=5267.41, p_Am=150.724, v=0.0196, p_M=243.001, kap=0.35756, k_J=0.002, kap_R=0.95,
                        E_Hb=0.7925, E_Hj=37.26, E_Hp=2361, E_0=1.81032, T_typical=273.15, del_M=0.14246,
                        comp=Composition.standard(n_waste='ammonia', d=0.2)),

}
