import numpy as np


class Compound:
    """
    class Compound:

        Stores the properties of a compound, including the chemical indices, the specific density, the chemical
        potential, the enthalpy of formation, the molecular weight_col and the chemical formula.
    """
    ATOMS = ['C', 'H', 'O', 'N']
    MOLECULAR_WEIGHTS = [12, 1, 16, 14]

    def __init__(self, n, d, mu, h, name, chemical_formula=None):
        """
        Instantiates a compound. The chemical indices, the specific density, the chemical potential, the enthalpy of
        formation and name of the compound are required. Molecular weight_col is computed from the chemical indices. If a
        chemical formula is not provided, it is built from the chemical indices.
        :param n: sequence of length 4 containing the the chemical indices in the order (C, H, O, N)
        :param d: specific density (g/cm^3)
        :param mu: chemical potential (J/mol)
        :param h: enthalpy of formation (J/mol)
        :param name: name of the compound
        :param chemical_formula: chemical formula
        """
        if len(n) != len(self.ATOMS):
            raise Exception("Chemical indices must be have length 4 corresponding to (C, H, O, N)")
        if not isinstance(n, np.ndarray):
            n = np.array(n)

        self.n = n  # Chemical indices (C, H, O, N)
        self.d = d  # Specific density of Compound (g/cm^3)
        self.mu = mu  # Chemical potential of Compound (J/mol) or (J/C-mol)
        self.h = h  # Enthalpy of formation (J/mol) or (J/C-mol)
        self.w = np.dot(n, self.MOLECULAR_WEIGHTS)  # Molecular weight_col (g/mol) or (g/C-mol)

        self.name = name
        if chemical_formula is not None:
            self._chemical_formula = chemical_formula

    @property
    def chemical_formula(self):
        """Returns the chemical formula. If a chemical formula was provided in the creation of the compound, that is
        returned. Otherwise, it is built from the chemical indices."""
        if self._chemical_formula is not None:
            return self._chemical_formula
        string = ''
        for atom, chem_index in zip(self.ATOMS, self.n):
            if chem_index == 1:
                string += f"{atom} "
            elif chem_index > 0:
                string += f"{atom}_{chem_index} "
        return string

    def __str__(self):
        return self.chemical_formula

    @property
    def description(self):
        # TODO: Print the name, and each property with units
        return

    @classmethod
    def carbon_dioxide(cls):
        """Constructor for carbon dioxide (CO2)."""
        return cls(n=(1, 0, 2, 0), d=0.1, mu=0, h=-393_520, name='Carbon Dioxide', chemical_formula='CO2')

    @classmethod
    def methane(cls):
        """Constructor for methane (CH4)."""
        return cls(n=(1, 4, 0, 0), d=0.1, mu=0, h=-74_900, name='Methane', chemical_formula='CH4')

    @classmethod
    def water(cls):
        """Constructor for water (H2O)."""
        return cls(n=(0, 2, 1, 0), d=0.1, mu=0, h=-285_830, name='Water', chemical_formula='H2O')

    @classmethod
    def oxygen(cls):
        """Constructor for oxygen (O2)."""
        return cls(n=(0, 0, 2, 0), d=0.1, mu=0, h=0, name='Oxygen', chemical_formula='O2')

    @classmethod
    def urea(cls):
        """Constructor for urea (CO(NH2)2)."""
        return cls(n=(1, 4, 1, 2), d=0.1, mu=0, h=0, name='Urea', chemical_formula='CO(NH2)2')

    @classmethod
    def ammonia(cls):
        """Constructor for ammonia (NH3)."""
        return cls(n=(0, 3, 0, 1), d=0.1, mu=0, h=-46_100, name='Ammonia', chemical_formula='NH3')

    @classmethod
    def food(cls, n=(1, 1.8, 0.5, 0.15), d=0.3, mu=525_000, h=-117_300):
        """Constructor for food (X). Properties have default values, but can be overridden."""
        return cls(n=n, d=d, mu=mu, h=h, name='Food')

    @classmethod
    def structure(cls, n=(1, 1.8, 0.5, 0.15), d=0.3, mu=500_000, h=-117_300):
        """Constructor for structure (V). Properties have default values, but can be overridden."""
        return cls(n=n, d=d, mu=mu, h=h, name='Structure')

    @classmethod
    def reserve(cls, n=(1, 1.8, 0.5, 0.15), d=0.3, mu=550_000, h=-117_300):
        """Constructor for reserve (E). Properties have default values, but can be overridden."""
        return cls(n=n, d=d, mu=mu, h=h, name='Reserve')

    @classmethod
    def feces(cls, n=(1, 1.8, 0.5, 0.15), d=0.3, mu=480_000, h=-117_300):
        """Constructor for food (P). Properties have default values, but can be overridden."""
        return cls(n=n, d=d, mu=mu, h=h, name='Feces')


class Composition:
    """
    class Composition:
        Stores the composition of an organism: 4 mineral compounds and 4 organic compounds. Assumes that 3 minerals are:
        carbon dioxide, water and oxygen. Has methods for returning matrices used in computations.
    """

    def __init__(self, n_waste=None, food=None, structure=None, reserve=None, feces=None):
        """
        Instantiates a Composition. If any argument is not provided, standard composition of DEB theory is assumed, with
        urea as nitrogenous waste.
        :param n_waste: Compound class for nitrogenous waste
        :param food: Compound class for food
        :param structure: Compound class for structure
        :param reserve: Compound class for reserve
        :param feces: Compound class for feces
        """
        self.C = Compound.carbon_dioxide()
        self.H = Compound.water()
        self.O = Compound.oxygen()

        if n_waste is None or n_waste == 'urea':
            self.N = Compound.urea()
        elif n_waste == 'ammonia':
            self.N = Compound.ammonia()
        elif isinstance(n_waste, Compound):
            self.N = n_waste
        else:
            raise Exception("Invalid input for N-Waste")

        self.X = self.set_organic_compound(food, Compound.food, 'food')
        self.V = self.set_organic_compound(structure, Compound.structure, 'structure')
        self.E = self.set_organic_compound(reserve, Compound.reserve, 'reserve')
        self.P = self.set_organic_compound(feces, Compound.feces, 'feces')

    @staticmethod
    def set_organic_compound(compound, constructor, name):
        if compound is None:
            return constructor()
        elif isinstance(compound, (float, int)):  # Assumes that the value to change is the specific density
            return constructor(d=compound)
        elif isinstance(compound, (tuple, list)):
            return constructor(*compound)
        elif isinstance(compound, dict):
            return constructor(**compound)
        elif isinstance(compound, Compound):
            return compound
        else:
            raise Exception(f"Invalid input for {name}")

    @property
    def n_O(self):
        """Matrix of chemical indices of organic compounds."""
        return np.array([self.X.n, self.V.n, self.E.n, self.P.n]).T

    @property
    def n_M(self):
        """Matrix of chemical indices of mineral compounds."""
        return np.array([self.C.n, self.H.n, self.O.n, self.N.n]).T

    @property
    def h_O(self):
        """Vector of enthalpies of formation of organic compounds."""
        return np.array([self.X.h, self.V.h, self.E.h, self.P.h])

    @property
    def h_M(self):
        """Vector of enthalpies of formation of mineral compounds."""
        return np.array([self.C.h, self.H.h, self.O.h, self.N.h])

    @property
    def organic_symbols(self):
        return 'X', 'V', 'E', 'P'

    @property
    def mineral_symbols(self):
        return self.C.chemical_formula, self.H.chemical_formula, self.O.chemical_formula, self.N.chemical_formula

    def __str__(self):
        # TODO: Print the name and formula of each compound.
        return ' '


class RuminantComposition(Composition):
    def __init__(self, food=None, structure=None, reserve=None, feces=None):
        super().__init__(n_waste='urea', food=food, structure=structure, reserve=reserve, feces=feces)
        self.M = Compound.methane()

    @property
    def n_M(self):
        return np.array([self.C.n, self.H.n, self.O.n, self.N.n, self.M.n]).T

    @property
    def h_M(self):
        return np.array([self.C.h, self.H.h, self.O.h, self.N.h, self.M.h])

    @property
    def mineral_symbols(self):
        return self.C.chemical_formula, self.H.chemical_formula, self.O.chemical_formula, self.N.chemical_formula, \
               self.M.chemical_formula
