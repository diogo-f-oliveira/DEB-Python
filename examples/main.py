import matplotlib
import matplotlib.pyplot as plt
import DEBpython.pet as pet
import DEBpython.models as models
import DEBpython.visualizer as viz

matplotlib.use('TkAgg')


import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Create Pet from parameters. Standard composition is assumed. Parameters t_0 and E_Hx are required in the STX model
    # mammal = pet.Pet(**pet.animals['bos_taurus_alentejana'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_alentejana'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_angus'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_limousin'])
    # mammal = pet.Ruminant(**pet.animals['bos_taurus_charolais'])
    # mammal = pet.Ruminant(**pet.animals['sheep'])
    # mammal = pet.Pet(**pet.animals['sheep'])

    # mammal.comp.X = composition.Compound.food(n=(1, 1.5, 0.69, 0.03))
    # mammal.comp.P = composition.Compound.feces(n=(1, 1.8, 0.5, 0.01))
    # mammal.kap_X = 0.2
    mertolenga_pars = {'p_Am': 4146.7,
                       'kap_X': 0.1374,
                       'p_M': 97.51,
                       'v': 0.05777,
                       'kap': 0.9324,
                       'E_G': 7838,
                       'E_Hb': 6.748e6,
                       'E_Hx': 3.998e7,
                       'E_Hp': 7.421e7,
                       't_0': 80,
                       'del_M': 0.2332,
                       'h_a': 5.554e-8,
                       'k_J': 0.002, 'kap_R': 0.95, 'T': 311.75}  # never forget internal temperature!
    mammal = pet.Pet(**mertolenga_pars)
    print(mammal)
    print(mammal.check_viability())

    # Create STX model from Pet class
    # model = models.STD(mammal)
    model = models.STX(mammal)
    # model = models.RUM(mammal)

    # Simulate the organism from birth with constant food density equal to 1. The returned solution contains the full
    # state of the organism, including powers, fluxes and entropy at all simulated time steps
    # sol = model.simulate(food_function=1, t_span=(0, 10300), step_size='auto', initial_state='embryo')
    init_state = model.get_state_at_maturity(mammal.E_Hp)
    init_state[0] = init_state[0] * 0.9
    print(init_state, mammal.E_m)
    sol = model.simulate(food_function=1, t_span=(0, 1200), step_size='auto', initial_state=init_state)

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
    #       f"FCR: {sol.feed_conversion_ratio:.4}\n"
    #       f"RGR: {sol.relative_growth_rate * 100:.4} %")

    # Visualize the simulation
    visual = viz.Plotter(sol)
    #
    visual.plot_state_vars()
    visual.plot_powers()
    # visual.plot_mineral_fluxes()
    visual.plot_real_variables()
    # visual.plot_emissions()

    # viz.plot_entropy_generation()
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.E/sol.V/mammal.E_m)
    ax.grid()
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("Scaled reserve density $e$ [-]")
    plt.show()
