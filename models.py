import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class STD:
    MAX_STEP_SIZE = 1
    def __init__(self, animal):
        self.animal = animal
        self.sol = None
        self.food_function = None

    def solve(self, food_function, t_span, initial_state='birth'):
        if initial_state == 'birth':
            initial_state = [self.animal.E_0, 1e-9, 0, 0]
        self.food_function = food_function
        self.sol = solve_ivp(self.state_changes, t_span, initial_state, max_step=self.MAX_STEP_SIZE)

    def state_changes(self, t, state_vars):
        E, V, E_H, E_R = state_vars  # Unpacking state variables (Reserve, Structure, Maturity, Reproduction Buffer)

        # Computing fluxes
        p_A = self.p_A(V, E_H, t)
        p_C = self.p_C(E, V)
        p_S = self.p_J(V)
        p_G = self.p_G(p_C, p_S)
        p_J = self.p_J(E_H)
        p_R = self.p_R(p_C, p_J)

        # Changes to state variables
        dE = p_A - p_C
        dV = p_G/self.animal.E_G
        # Maturity or Reproduction Buffer logic
        if E_H < self.animal.E_Hp:
            dE_H = p_R
            dE_R = 0
        else:
            dE_H = 0
            dE_R = self.animal.k_R * p_R
        return [dE, dV, dE_H, dE_R]

    # Assimilation Flux
    def p_A(self, V, E_H, t):
        if E_H < self.animal.E_Hb:
            food_level = 0
        else:
            food_level = self.food_function(t)
        return self.animal.P_Am * food_level * (V ** (2 / 3))

    # Mobilization Flux
    def p_C(self, E, V):
        return E * (self.animal.E_G * self.animal.v * (V ** (-1 / 3)) + self.animal.P_M) / \
               (self.animal.kappa * E / V + self.animal.E_G)

    # Somatic Maintenance Flux
    def p_S(self, V):
        return self.animal.P_M * V

    # Growth Flux
    def p_G(self, p_C, p_S):
        return self.animal.kappa * p_C - p_S

    # Maturity Maintenance Flux
    def p_J(self, E_H):
        return self.animal.k_J * E_H

    # Maturation/Reproduction Flux
    def p_R(self, p_C, p_J):
        return (1 - self.animal.kappa) * p_C - p_J

    def plot_state_vars(self):

