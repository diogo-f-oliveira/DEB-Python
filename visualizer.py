import matplotlib.pyplot as plt
from models import Solution

class Plotter:
    label_fontsize = 16
    title_fontsize = 20

    def __init__(self, model):
        self.sol = Solution(model)

    def plot_state_vars(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), tight_layout=True, num="State Variables")

        # Reserve plot
        ax = axes[0, 0]
        ax.plot(self.sol.t, self.sol.E)
        ax.set_xlabel('Time [d]', fontsize=self.label_fontsize)
        ax.set_ylabel('Reserve [J]', fontsize=self.label_fontsize)
        ax.set_title('Reserve vs Time', fontsize=self.title_fontsize)
        ax.grid()

        # Structure plot
        ax = axes[0, 1]
        ax.plot(self.sol.t, self.sol.V)
        ax.set_xlabel('Time [d]', fontsize=self.label_fontsize)
        ax.set_ylabel('Structure [cm$^3$]', fontsize=self.label_fontsize)
        ax.set_title('Structure vs Time', fontsize=self.title_fontsize)
        ax.grid()

        # Maturity plot
        ax = axes[1, 0]
        ax.plot(self.sol.t, self.sol.E_H)
        ax.set_xlabel('Time [d]', fontsize=self.label_fontsize)
        ax.set_ylabel('Maturity [J]', fontsize=self.label_fontsize)
        ax.set_title('Maturity vs Time', fontsize=self.title_fontsize)
        ax.grid()

        # Reproduction Buffer plot
        ax = axes[1, 1]
        ax.plot(self.sol.t, self.sol.E_R)
        ax.set_xlabel('Time [d]', fontsize=self.label_fontsize)
        ax.set_ylabel('Reproduction Buffer [J]', fontsize=self.label_fontsize)
        ax.set_title('Reproduction Buffer vs Time', fontsize=self.title_fontsize)
        ax.grid()

        # fig.tight_layout()
        fig.show()
