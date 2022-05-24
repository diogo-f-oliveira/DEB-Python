import matplotlib.pyplot as plt
from solution import TimeIntervalSol


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

        self.plot_vs_time(axes[0, 0], self.sol.p_A, 'Assimilation Power', 'J/d',title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[0, 1], self.sol.p_C, 'Mobilization Power', 'J/d',title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[0, 2], self.sol.p_S, 'Somatic Maintenance Power', 'J/d',title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[1, 0], self.sol.p_G, 'Growth Power', 'J/d',title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[1, 1], self.sol.p_J, 'Maturity Maintenance Power', 'J/d',title_fontsize=15, label_fontsize=15)
        self.plot_vs_time(axes[1, 2], self.sol.p_R, 'Reproduction Power', 'J/d',title_fontsize=15, label_fontsize=15)

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
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), tight_layout=True, num="Real Variables")

        self.plot_vs_time(axes[0], self.sol.physical_length, 'Physical Length', 'cm')

        fig.show()

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
        ax.axvline(x=self.sol.time_of_birth, linestyle=':', color='k')
        ax.axvline(x=self.sol.time_of_puberty, linestyle=':', color='k')
        if self.sol.time_of_weaning:
            ax.axvline(x=self.sol.time_of_weaning, linestyle=':', color='k')
