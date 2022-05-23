import pet
import models
from visualizer import Plotter

if __name__ == '__main__':
    # shark = pet.Pet(**pet.animals['shark'])
    # shark.print_reactions()
    #
    # model = models.STD(shark)

    # Get parameters of muskox from animals dictionary
    # muskox = pet.Pet(**pet.animals['muskox'])
    cow = pet.Pet(**pet.animals['bos_taurus_alentejana'])
    cow.print_reactions()
    # print(muskox)
    # Initialize model STX for muskox
    # model = models.STX(muskox)
    model = models.STX(cow)

    # Simulate the organism from birth with constant food density equal to 1
    sol = model.simulate(food_function=1, t_span=(0, 10000), step_size='auto', initial_state='birth')

    fully_sol = model.fully_grown()

    # Visualize the simulation
    viz = Plotter(model)

    viz.plot_state_vars()
    viz.plot_powers()
    viz.plot_mineral_fluxes()
    viz.plot_real_variables()
    viz.plot_entropy_generation()

