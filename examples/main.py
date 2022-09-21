import pet
import models
from visualizer import Plotter
from math import sin
import composition
import warnings
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Create Pet from parameters. Standard composition is assumed. Parameters t_0 and E_Hx are required in the STX model
    # mammal = pet.Pet(**pet.animals['bos_taurus_alentejana'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_alentejana'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_angus'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_limousin'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_charolais'])
    mammal = pet.Ruminant(**pet.animals['sheep'])
    # mammal._p_Am = mammal._p_Am * 5

    # mammal.comp.X = composition.Compound.food(n=(1, 1.5, 0.69, 0.03))
    # mammal.comp.P = composition.Compound.feces(n=(1, 1.8, 0.5, 0.01))
    # mammal.kap_X = 0.2
    print(mammal)
    print(mammal.check_viability())

    # Create STX model from Pet class
    # model = models.STX(mammal)
    model = models.RUM(mammal)

    # Simulate the organism from birth with constant food density equal to 1. The returned solution contains the full
    # state of the organism, including powers, fluxes and entropy at all simulated time steps
    # sol = model.simulate(food_function=1, t_span=(0, 10300), step_size='auto', initial_state='embryo')
    sol = model.simulate(food_function=0.7, t_span=(0, 5000), step_size='auto', initial_state='birth')

    # Simulate with changing temperature (affects rate parameters)
    # def changing_temperature(organism, t, state_vars):
    #     if state_vars[2] > organism.E_Hb:
    #         organism.T = 298.15 + 5 * sin(t / 365)

    # sol = model.simulate(food_function=1, t_span=(0, 10000), step_size='auto', initial_state='embryo',
    #                      transformation=changing_temperature)

    # Simulate with change in diet
    # def diet_change(organism, t, state_vars):
    #     if state_vars[2] >= organism.E_Hx:
    #         organism.comp.X = composition.Compound.food(n=(1, 1.5, 0.69, 0.03))
    #     else:
    #         organism.comp.X = composition.Compound.food()

    #
    # sol = model.simulate(food_function=1, t_span=(0, 10000), step_size='auto', initial_state='embryo',
    #                      transformation=diet_change)

    # mammal.print_reactions()

    # sol = model.fully_grown()
    # print(sol.entropy)
    # Find TimeInstantSol at t1_time
    # t1_time = 295
    # for i, t in enumerate(sol.t):
    #     if t > t1_time:
    #         t1 = sol[t]
    #         t1_i = i
    #         break
    #
    sol.print_growth_report()
    # print(f"DFI: {sol.daily_feed_intake:.5} g\n"
    #       f"ADG: {sol.average_daily_gain:.4} g\n"
    #       f"FCR: {sol.feed_consumption_ratio:.4}\n"
    #       f"RGR: {sol.relative_growth_rate * 100:.4} %")

    # Visualize the simulation
    viz = Plotter(sol)
    #
    viz.plot_state_vars()
    viz.plot_powers()
    # viz.plot_mineral_fluxes()
    viz.plot_real_variables()

    # viz.plot_entropy_generation()

    plt.show()
