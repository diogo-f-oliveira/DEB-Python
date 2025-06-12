import DEBpython.state as state
import DEBpython.pet as pet
import DEBpython.composition as composition
import DEBpython.models as models
import DEBpython.environment as environment
import DEBpython.visualizer as visualizer

if __name__ == '__main__':
    # warnings.filterwarnings('ignore')

    # organism = pet.Pet(**pet.animals['bos_taurus_angus'])
    # # env = environment.Environment(
    # #     pet=organism,
    # #     food_function=lambda t, pet: pet.state.V ** (2 / 3) * pet.p_Xm,
    # #     temp_function=lambda t, pet: pet.T_typical,
    # #     food_comp_function=lambda t, pet: composition.Compound.food(),
    # # )
    # env = environment.ConstantEnvironment(
    #     pet=organism,
    #     temp=organism.T_typical,
    # )
    #
    # model = models.STX(organism=organism, env=env)
    #
    # initial_state = state.State()
    # initial_state.set_state_vars((organism.E_0, organism.V_0, 0, 0))
    # sol = model.simulate(
    #     t_span=(0, 5000),
    #     initial_state=initial_state,
    # )

    temp = 25 + 273.15
    f = 1
    initial_state = state.ABJState()
    initial_state.T = temp
    organism = pet.Pet(**pet.animals['Danio_rerio'], state=initial_state)
    initial_state.set_state_vars((organism.E_0, organism.V_0, 0., 0., 1.))
    env = environment.ConstantEnvironment(
        pet=organism,
        temp=temp,
        f=f,
    )
    model = models.ABJ(organism=organism, env=env)
    sol = model.simulate(
        t_span=(0, 100),
        initial_state=initial_state,
    )

    # Visualize the simulation
    visual = visualizer.Plotter(sol)
    #
    visual.plot_state_vars()
    visual.plot_powers()
    # visual.plot_mineral_fluxes()
    visual.plot_real_variables()
    # visual.plot_emissions()
