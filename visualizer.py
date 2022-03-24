import matplotlib.pyplot as plt
from models import Solution


class Plotter:
    label_fontsize = 16
    title_fontsize = 20

    def __init__(self, model):
        self.sol = Solution(model)

    def plot_state_vars(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), tight_layout=True, num="State Variables")

        self.plot_vs_time(axes[0, 0], self.sol.E, 'Reserve', 'J')
        self.plot_vs_time(axes[0, 1], self.sol.V, 'Structure', 'cm$^3$')
        self.plot_vs_time(axes[1, 0], self.sol.E_H, 'Maturity', 'J')
        self.plot_vs_time(axes[1, 1], self.sol.E_R, 'Reproduction Buffer', 'J')

        fig.show()

    def plot_powers(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), tight_layout=True, num="Powers")

        self.plot_vs_time(axes[0, 0], self.sol.p_A, 'Assimilation Power', 'J/d')
        self.plot_vs_time(axes[0, 1], self.sol.p_C, 'Mobilization Power', 'J/d')
        self.plot_vs_time(axes[0, 2], self.sol.p_S, 'Somatic Maintenance Power', 'J/d')
        self.plot_vs_time(axes[1, 0], self.sol.p_G, 'Growth Power', 'J/d')
        self.plot_vs_time(axes[1, 1], self.sol.p_J, 'Maturity Maintenance Power', 'J/d')
        self.plot_vs_time(axes[1, 2], self.sol.p_R, 'Reproduction Power', 'J/d')

        fig.show()

    def plot_mineral_fluxes(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), tight_layout=True, num="Mineral Fluxes")

        self.plot_vs_time(axes[0, 0], self.sol.mineral_fluxes[0], 'CO$_2$ Flux', 'mol/d')
        self.plot_vs_time(axes[0, 1], self.sol.mineral_fluxes[1], 'H$_2$O Flux', 'mol/d')
        self.plot_vs_time(axes[1, 0], self.sol.mineral_fluxes[2], 'O$_2$ Flux', 'mol/d')
        self.plot_vs_time(axes[1, 1], self.sol.mineral_fluxes[3], 'N-Waste Flux', 'mol/d')

        fig.show()

    def plot_vs_time(self, ax, variable, variable_name='', unit='J'):
        ax.plot(self.sol.t, variable)

        self.plot_stage_transitions(ax)

        ax.set_xlabel('Time [d]', fontsize=self.label_fontsize)
        ax.set_ylabel(f'{variable_name} [{unit}]', fontsize=self.label_fontsize)
        ax.set_title(f'{variable_name} vs Time', fontsize=self.title_fontsize)
        ax.grid()

    def plot_stage_transitions(self, ax):
        ax.axvline(x=self.sol.time_of_birth, linestyle=':', color='k')
        ax.axvline(x=self.sol.time_of_puberty, linestyle=':', color='k')
        if self.sol.time_of_weaning:
            ax.axvline(x=self.sol.time_of_weaning, linestyle=':', color='k')
