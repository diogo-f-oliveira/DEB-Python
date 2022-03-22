import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np


class STD:
    MAX_STEP_SIZE = 1 / 24

    def __init__(self, animal):
        self.animal = animal
        self.sol = None
        self.food_function = None
        self.plotter = None

    def solve(self, food_function, t_span, step_size, initial_state='birth'):
        if initial_state == 'birth':
            initial_state = [self.animal.E_0, 3e-9, 0, 0]
        self.food_function = food_function
        self.sol = solve_ivp(self.state_changes, t_span, initial_state, t_eval=np.arange(*t_span, step_size))

    def state_changes(self, t, state_vars):
        E, V, E_H, E_R = state_vars  # Unpacking state variables (Reserve, Structure, Maturity, Reproduction Buffer)

        # Computing fluxes
        p_A = self.p_A(V, E_H, t)
        p_C = self.p_C(E, V)
        p_S = self.p_S(V)
        p_G = self.p_G(p_C, p_S)
        p_J = self.p_J(E_H)
        p_R = self.p_R(p_C, p_J)

        # Changes to state variables
        dE = p_A - p_C
        dV = p_G / self.animal.E_G
        # Maturity or Reproduction Buffer logic
        if E_H < self.animal.E_Hp:
            dE_H = p_R
            dE_R = 0
        else:
            dE_H = 0
            dE_R = self.animal.k_R * p_R
        return dE, dV, dE_H, dE_R

    # Assimilation Flux
    def p_A(self, V, E_H, t):
        if type(E_H) == np.ndarray:
            for i, (structure, maturity, time) in enumerate(zip(V, E_H, t)):
                p_A = np.zeros_like(E_H)
                if maturity < self.animal.E_Hb:
                    p_A[i] = 0
                else:
                    p_A[i] = self.animal.P_Am * self.food_function(time) * (structure ** (2 / 3))
            return p_A
        else:
            if E_H < self.animal.E_Hb:
                return 0
            else:
                return self.animal.P_Am * self.food_function(t) * (V ** (2 / 3))

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
        if type(E_H) == np.ndarray:
            p_J = np.zeros_like(E_H)
            for i, maturity in enumerate(E_H):
                if maturity < self.animal.E_Hp:
                    p_J[i] = self.animal.k_J * maturity
                else:
                    p_J[i] = self.animal.k_J * self.animal.E_Hp
            return p_J
        else:
            if E_H < self.animal.E_Hp:
                return self.animal.k_J * E_H
            else:
                return self.animal.k_J * self.animal.E_Hp

    # Maturation/Reproduction Flux
    def p_R(self, p_C, p_J):
        return (1 - self.animal.kappa) * p_C - p_J



class Solution:
    def __init__(self, model):
        self.t = model.sol.t
        self.E = model.sol.y[0]
        self.V = model.sol.y[1]
        self.E_H = model.sol.y[2]
        self.E_R = model.sol.y[3]

        self.p_A = model.p_A(self.V, self.E_H, self.t)
        self.p_C = model.p_C(self.E, self.V)
        self.p_S = model.p_S(self.V)
        self.p_G = model.p_G(self.p_C, self.p_S)
        self.p_J = model.p_J(self.E_H)
        self.p_R = model.p_R(self.p_C, self.p_J)