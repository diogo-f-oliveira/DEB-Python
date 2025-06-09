from .pet import Pet
from .composition import Compound
import numpy as np


class Environment:
    def __init__(self, pet: Pet, food_function, temp_function, food_comp_function):
        self.pet = pet
        self.state = pet.state
        self.food_function = food_function
        self.temp_function = temp_function
        self.food_comp_function = food_comp_function

    def update(self):
        self.apply_temp()
        self.apply_food_comp()
        self.apply_food()

    def apply_food(self):
        self.state.p_X = self.food_function(self.state.t, self.pet)

    def apply_temp(self):
        self.state.T = self.temp_function(self.state.t, self.pet)

    def apply_food_comp(self):
        self.state.food_comp = self.food_comp_function(self.state.t, self.pet)


class ConstantEnvironment(Environment):
    def __init__(self, pet: Pet, f=1, food_comp=None, temp=293):
        self.f = f
        if food_comp is None:
            food_comp = Compound.food()
        self.food_comp = food_comp
        self.temp = temp
        super().__init__(pet=pet,
                         food_function=self.constant_food,
                         food_comp_function=self.constant_food_comp,
                         temp_function=self.constant_temp)

    def constant_food(self, t, pet):
        return self.f * pet.state.V ** (2 / 3) * pet.p_Xm

    def constant_food_comp(self, t, pet):
        return self.food_comp

    def constant_temp(self, t, pet):
        return self.temp


class SinusoidalTemperatureEnvironment(Environment):
    def __init__(self, pet: Pet, f=1, food_comp=None, T_max=293, T_min=293, T_init=293, period=365):
        # Food parameters
        self.f = f
        if food_comp is None:
            food_comp = Compound.food()
        self.food_comp = food_comp

        # Sinusoidal temperature function parameters
        self.T_max = T_max
        self.T_min = T_min
        self.T_init = T_init
        self.period = period
        # Calculate amplitude and mean temperature
        self.amplitude = (T_max - T_min) / 2
        self.T_mean = (T_max + T_min) / 2
        # Solve for phase shift to match initial temperature
        self.phase_init = np.arcsin((self.T_init - self.T_mean) / self.amplitude)

        super().__init__(pet=pet,
                         food_function=self.constant_food,
                         food_comp_function=self.constant_food_comp,
                         temp_function=self.sinusoidal_temperature)

    def sinusoidal_temperature(self, t, pet):
        return self.T_mean + self.amplitude * np.sin((2 * np.pi / self.period) * t + self.phase_init)

    def constant_food(self, t, pet):
        return self.f * pet.state.V ** (2 / 3) * pet.p_Xm

    def constant_food_comp(self, t, pet):
        return self.food_comp
