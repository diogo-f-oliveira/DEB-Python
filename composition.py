import numpy as np


class Compound:
    ATOMS = ['C', 'H', 'O', 'N']
    MOLECULAR_WEIGHTS = [12, 1, 16, 14]

    def __init__(self, n, d, mu, h, name, chemical_formula=None):
        if len(n) != len(self.ATOMS):
            raise Exception("Chemical indices must be have length 4 corresponding to (C, H, O, N)")
        if not isinstance(n, np.ndarray):
            n = np.array(n)

        self.n = n  # Chemical indices (C, H, O, N)
        self.d = d  # Specific density of Compound (g/cm^3)
        self.mu = mu  # Chemical potential of Compound (J/mol) or (J/C-mol)
        self.h = h  # Enthalpy of formation (J/mol) or (J/C-mol)
        self.w = np.dot(n, self.MOLECULAR_WEIGHTS)  # Molecular weight (g/mol) or (g/C-mol)

        self.name = name
        if chemical_formula is not None:
            self._chemical_formula = chemical_formula

    @property
    def chemical_formula(self):
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
        # TODO: Return the name, and each property with units
        return self.chemical_formula

    @classmethod
    def carbon_dioxide(cls):
        return cls(n=(1, 0, 2, 0), d=0.1, mu=0, h=-393_520, name='Carbon Dioxide', chemical_formula='CO2')

    @classmethod
    def water(cls):
        return cls(n=(0, 2, 1, 0), d=0.1, mu=0, h=-285_830, name='Water', chemical_formula='H2O')

    @classmethod
    def oxygen(cls):
        return cls(n=(0, 0, 2, 0), d=0.1, mu=0, h=0, name='Oxygen', chemical_formula='O2')

    @classmethod
    def urea(cls):
        return cls(n=(1, 4, 1, 2), d=0.1, mu=0, h=0, name='Urea', chemical_formula='CO(NH2)2')

    @classmethod
    def ammonia(cls):
        return cls(n=(0, 3, 0, 1), d=0.1, mu=0, h=-46_100, name='Ammonia', chemical_formula='NH3')

    @classmethod
    def food(cls, n=(1, 1.8, 0.5, 0.15), d=0.1, mu=525_000, h=-117_300):
        return cls(n=n, d=d, mu=mu, h=h, name='Food')

    @classmethod
    def structure(cls, n=(1, 1.8, 0.5, 0.15), d=0.1, mu=500_000, h=-117_300):
        return cls(n=n, d=d, mu=mu, h=h, name='Structure')

    @classmethod
    def reserve(cls, n=(1, 1.8, 0.5, 0.15), d=0.1, mu=550_000, h=-117_300):
        return cls(n=n, d=d, mu=mu, h=h, name='Reserve')

    @classmethod
    def feces(cls, n=(1, 1.8, 0.5, 0.15), d=0.1, mu=480_000, h=-117_300):
        return cls(n=n, d=d, mu=mu, h=h, name='Feces')


class Composition:
    def __init__(self, n_waste=None, food=None, structure=None, reserve=None, feces=None):
        self.C = Compound(n=(1, 0, 2, 0), d=0.1, mu=0, h=-393_520, name='Carbon Dioxide')
        self.H = Compound(n=(0, 2, 1, 0), d=0.1, mu=0, h=-285_830, name='Water')
        self.O = Compound(n=(0, 0, 2, 0), d=0.1, mu=0, h=0, name='Oxygen')

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
        return np.array([self.X.n, self.V.n, self.E.n, self.P.n]).T

    @property
    def n_M(self):
        return np.array([self.C.n, self.H.n, self.O.n, self.N.n]).T

    @property
    def h_O(self):
        return np.array([self.X.h, self.V.h, self.E.h, self.P.h])

    @property
    def h_M(self):
        return np.array([self.C.h, self.H.h, self.O.h, self.N.h])


if __name__ == '__main__':
    comp = Composition(structure=0.3)
