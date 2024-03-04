from .solution import TimeIntervalSol

import matplotlib.pyplot as plt

tex_par_symbols = {
    'p_Am': r'\{\dot{p}_{Am}\}',
    'p_M': r'\dot{p}_M',
    'v': r'\dot{v}',
    'kap': r'\kappa',
    'kap_X': r'\kappa_X',
    'E_G': '[E_G]',
    'E_Hb': 'E_H^b',
    'E_Hx': 'E_H^x',
    'E_Hp': 'E_H^p',
    'k_J': r'\dot{k}_J',
    'kap_R': r'\kappa_R',
    'kap_P': r'\kappa_P',
    'kap_G': r'\kappa_G',
    'r_B': r'\dot{r}_B',
    'omega': r'\omega',
    'xi_C': r'\xi_C',
    'h_a': r'\ddot{h}_a',
    't_0': r't_0',
    'del_M': r'\delta_M'
}
tex_par_units = {
    'p_Am': r'J/d \cdot cm^2',
    'p_M': r'J/d \cdot cm^3',
    'v': r'cm/d',
    'kap': r'-',
    'kap_X': r'-',
    'E_G': 'J/cm^3',
    'E_Hb': 'J',
    'E_Hx': 'J',
    'E_Hp': 'J',
    'k_J': r'd^-1',
    'kap_R': r'-',
    'kap_P': r'-',
    'kap_G': r'-',
    'r_B': r'd^-1',
    'omega': r'-',
    'xi_C': r'-',
    'h_a': r'd^{-2}',
    't_0': r'd',
    'del_M': r'-'
}

par_descriptions_en = {
    'p_Am': 'Surface-specific maximum assimilation rate',
    'p_M': 'Volume-specific somatic maintenance rate',
    'v': 'Energy conductance',
    'kap': 'Allocation fraction to soma',
    'kap_X': 'Digestion efficiency',
    'E_G': 'Specific cost for Structure',
    'E_Hb': 'Maturity at birth',
    'E_Hx': 'Maturity at weaning',
    'E_Hp': 'Maturity at puberty',
    'k_J': 'Maturity maintenance rate constant',
    'kap_R': 'Reproduction efficiency',
    'kap_P': 'Defecation efficiency',
    'kap_G': 'Growth efficiency',
    'r_B': r'von Bertalanffy growth rate',
    'omega': 'Contribution of ash free dry mass of reserve to total ash free dry biomass',
    'xi_C': 'Contribution of methane subtransformation to assimilation',
    'h_a': 'Weibull aging acceleration',
    't_0': 'Diapause',
    'del_M': 'Shape coefficient'
}


# TODO: description dict for pars in Portuguese
class Plotter:
    """
    class Plotter:

    Plots the evolution of a Pet from the result of a simulation.
    """

    def __init__(self, sol):
        """Instantiates a Plotter from a TimeIntervalSol solution."""
        if not isinstance(sol, TimeIntervalSol):
            raise Exception("Invalid solution type.")
        self.sol = sol

    def plot_state_vars(self):
        """Plots the evolution of the state variables."""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), tight_layout=True, num="State Variables")

        self.plot_vs_time(axes[0, 0], self.sol.E, 'Reserve (E)', 'J')
        self.plot_vs_time(axes[0, 1], self.sol.V, 'Structure (V)', 'cm$^3$')
        self.plot_vs_time(axes[1, 0], self.sol.E_H, 'Maturity $(E_H)$', 'J')
        self.plot_vs_time(axes[1, 1], self.sol.E_R, 'Reproduction Buffer $(E_R)$', 'J')

        fig.show()

    def plot_powers(self):
        """Plots the evolution of the powers."""
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), tight_layout=True, num="Powers")

        self.plot_vs_time(axes[0, 0], self.sol.p_A, 'Assimilation Power', 'J/d', title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[0, 1], self.sol.p_C, 'Mobilization Power', 'J/d', title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[0, 2], self.sol.p_S, 'Somatic Maintenance Power', 'J/d', title_fontsize=15,
                          label_fontsize=15)
        self.plot_vs_time(axes[1, 0], self.sol.p_G, 'Growth Power', 'J/d', title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[1, 1], self.sol.p_J, 'Maturity Maintenance Power', 'J/d', title_fontsize=15,
                          label_fontsize=15)
        self.plot_vs_time(axes[1, 2], self.sol.p_R, 'Reproduction Power', 'J/d', title_fontsize=15, label_fontsize=15)

        fig.show()

    def plot_organic_fluxes(self):
        """Plots the evolution of the organic fluxes."""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), tight_layout=True, num="Organic Fluxes")

        self.plot_vs_time(axes[0, 0], self.sol.organic_fluxes[0], 'Food Flux', 'mol/d')
        self.plot_vs_time(axes[0, 1], self.sol.organic_fluxes[1], 'Reserve Flux', 'mol/d')
        self.plot_vs_time(axes[1, 0], self.sol.organic_fluxes[2], 'Reserve and Reproduction Buffer Flux', 'mol/d')
        self.plot_vs_time(axes[1, 1], self.sol.organic_fluxes[3], 'Feces Flux', 'mol/d')

        fig.show()

    def plot_mineral_fluxes(self):
        """Plots the evolution of the mineral fluxes."""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), tight_layout=True, num="Mineral Fluxes")

        self.plot_vs_time(axes[0, 0], self.sol.mineral_fluxes[0], 'CO$_2$ Flux', 'mol/d')
        self.plot_vs_time(axes[0, 1], self.sol.mineral_fluxes[1], 'H$_2$O Flux', 'mol/d')
        self.plot_vs_time(axes[1, 0], self.sol.mineral_fluxes[2], 'O$_2$ Flux', 'mol/d')
        self.plot_vs_time(axes[1, 1], self.sol.mineral_fluxes[3], 'N-Waste Flux', 'mol/d')

        fig.show()

    def plot_entropy_generation(self):
        """Plots the evolution of entropy generation."""
        fig, axes = plt.subplots(figsize=(16, 9), tight_layout=True, num="Entropy Generation")

        self.plot_vs_time(axes, self.sol.entropy, 'Entropy Generation', 'J/K')

        fig.show()

    def plot_real_variables(self):
        """Plots the evolution of real variables such as physical length."""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), tight_layout=True, num="Real Variables")

        self.plot_vs_time(axes[0, 0], self.sol.physical_length, 'Physical Length', 'cm')
        self.plot_vs_time(axes[0, 1], self.sol.physical_volume, 'Physical Volume', 'cm$^3$')
        self.plot_vs_time(axes[1, 0], self.sol.wet_weight, 'Wet Weight', 'g')
        self.plot_vs_time(axes[1, 1], self.sol.dry_weight, 'Dry Weight', 'g')

        fig.show()

    def plot_emissions(self):
        """Plots the GHG emissions in CO2 equivalent."""
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), tight_layout=True, num="GHG Emissions")

        self.plot_vs_time(axes, self.sol.mineral_fluxes[0] / self.sol.wet_weight, 'CO2 Emissions', 'g$_{CO_2}')

    def plot_vs_time(self, ax, variable, variable_name='', unit='J', label_fontsize=16, title_fontsize=20):
        """
        Plots a variable vs time.
        :param ax: Matplotlib Axes instance
        :param variable: array of the variable to plot vs time
        :param variable_name: Name of the variable
        :param unit: Unit of the variable
        :param label_fontsize: Font size of axis labels (default: 16)
        :param title_fontsize: Font size of title (default: 20)
        """
        ax.plot(self.sol.t, variable)

        self.plot_stage_transitions(ax)

        ax.set_xlabel('Time [d]', fontsize=label_fontsize)
        ax.set_ylabel(f'{variable_name} [{unit}]', fontsize=label_fontsize)
        ax.set_title(f'{variable_name} vs Time', fontsize=title_fontsize)
        ax.grid()

    def plot_stage_transitions(self, ax):
        """
        Adds vertical lines to a plot to represent stage transitions.
        :param ax: Matplotlib Axes instance
        """
        if self.sol.time_of_birth:
            ax.axvline(x=self.sol.time_of_birth, linestyle=':', color='k')
        if self.sol.time_of_puberty:
            ax.axvline(x=self.sol.time_of_puberty, linestyle=':', color='k')
        if self.sol.time_of_weaning:
            ax.axvline(x=self.sol.time_of_weaning, linestyle=':', color='k')
