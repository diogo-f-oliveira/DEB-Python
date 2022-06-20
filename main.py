import pet
import models
from visualizer import Plotter
from math import sin
import composition

if __name__ == '__main__':
    # Create Pet from parameters. Standard composition is assumed. Parameters t_0 and E_Hx are required in the STX model
    # mammal = pet.Pet(**pet.animals['bos_taurus_alentejana'])
    mammal = pet.Ruminant(**pet.animals['bos_taurus_alentejana'])
    # L_m = 100
    # p_Am = 998
    # kappa = 0.989
    # v = 602.3
    # mammal = pet.Pet(
    #     p_Am=998,
    #     kappa=0.989,
    #     v=602.3,
    #     p_M=p_Am * kappa / L_m,
    #     E_G=8261.79,
    #     k_J=0.002,
    #     E_Hb=2071229.972,
    #     E_Hp=30724119.81,
    #     kap_R=0.95,
    #     t_0=0.1,
    #     E_Hx=15139260.45,
    #     T=298.15,
    #     T_ref=298.15
    # )
    # mammal.comp.X = composition.Compound.food(n=(1, 1.5, 0.69, 0.03))
    # mammal.comp.C = composition.Compound.methane()
    print(mammal)
    print(mammal.check_viability())

    # Create STX model from Pet class
    # model = models.STX(mammal)
    model = models.RUM(mammal)

    # Simulate the organism from birth with constant food density equal to 1. The returned solution contains the full
    # state of the organism, including powers, fluxes and entropy at all simulated time steps
    sol = model.simulate(food_function=1, t_span=(0, 1300), step_size='auto', initial_state='embryo')

    # Simulate with changing temperature (affects rate parameters)
    def changing_temperature(organism, t, state_vars):
        if state_vars[2] > organism.E_Hb:
            organism.T = 298.15 + 5 * sin(t / 365)

    # sol = model.simulate(food_function=1, t_span=(0, 10000), step_size='auto', initial_state='embryo',
    #                      transformation=changing_temperature)


    # Simulate with change in diet
    def diet_change(organism, t, state_vars):
        if state_vars[2] >= organism.E_Hx:
            organism.comp.X = composition.Compound.food(n=(1, 1.5, 0.69, 0.03))
        else:
            organism.comp.X = composition.Compound.food()
    #
    # sol = model.simulate(food_function=1, t_span=(0, 10000), step_size='auto', initial_state='embryo',
    #                      transformation=diet_change)

    # mammal.print_reactions()

    # sol = model.fully_grown()
    # print(sol.entropy)

    # Visualize the simulation
    # viz = Plotter(sol)
    #
    # viz.plot_state_vars()
    # viz.plot_powers()
    # viz.plot_mineral_fluxes()
    # viz.plot_real_variables()
    # viz.plot_entropy_generation()
