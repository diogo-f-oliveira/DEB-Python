import animal
import models
from visualizer import Plotter

if __name__ == '__main__':
    shark = animal.Animal(E_G=5212.32, P_Am=558.824, v=0.02774, P_M=34.3632, kappa=0.84851, k_J=0.002, k_R=0.95,
                          E_Hb=7096, E_Hp=300600)
    model = models.STD(shark)

    model.solve(food_function=lambda t: 1, t_span=(0, 20000), step_size=12/24, initial_state='birth')

    # print(model.sol)

    viz = Plotter(model)
    viz.plot_state_vars()

