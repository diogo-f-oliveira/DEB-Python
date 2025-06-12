from collections.abc import MutableSequence
from copy import deepcopy

import numpy as np
from scipy.integrate import simpson


class TimeInstantSol:
    """
        TimeIntervalSol class:

        Stores the complete state of the organism at a given time step, including state variables, powers,  fluxes,
        entropy and real variables such as physical length.
        """

    def __init__(self, model, t, state_vars):
        """
        Creates an instance of TimeInstantSol
        :param model: model class
        :param t: time (d)
        :param state_vars: state variables at time t in the order (E, V, E_H, E_R)
        """
        self.model_type = type(model).__name__

        self.organism = model.organism  # This is just a pointer to the organism, not a copy
        model.organism.state.set_state_vars(state_vars)
        model.organism.state.t = t
        model.env.update()
        self.state = deepcopy(model.organism.state)

        # Save state variables
        # TODO: either delete or make it independent
        # self.t = t
        # self.E, self.V, self.E_H, self.E_R = state_vars

        # Powers
        powers = model.compute_powers(compute_dissipation_power=True)
        self.p_A, self.p_C, self.p_S, self.p_G, self.p_J, self.p_R, self.p_D = powers

        # Fluxes
        self.organic_fluxes = model.organic_fluxes(self.p_A, self.p_D, self.p_G)
        self.mineral_fluxes = model.mineral_fluxes(self.p_A, self.p_D, self.p_G)
        # Entropy
        self.entropy = model.entropy_generation(self.p_A, self.p_D, self.p_G)[0]
        # Physical Length
        self.calculate_real_variables()

    def calculate_real_variables(self):
        """Computes real variables such as physical length, etc... (WIP)"""
        self.physical_length = self.organism.physical_length
        self.physical_volume = self.organism.physical_volume
        self.wet_weight = self.organism.wet_weight
        self.dry_weight = self.organism.dry_weight

        self.feed_intake = (self.organic_fluxes[0] * self.organism.comp.X.w)[0]
        self.carbon_dioxide = (self.mineral_fluxes[0] * self.organism.comp.C.w)[0]
        if self.model_type == 'RUM':
            self.methane = (self.mineral_fluxes[4] * self.organism.comp.M.w)[0]


class TimeIntervalSol(MutableSequence):
    """
    TimeIntervalSol class:

    Stores the complete solution to the integration of state equations, including state variables, powers, fluxes,
    entropy and real variables such as physical length, as well as time of stage transitions.
    """

    def __init__(self, model, ode_sol):
        """
        Creates an instance of TimeIntervalSol from a model containing a simulation of the organism.
        :param model: model class
        """
        self.model = model

        self.t = ode_sol.t
        self._time_to_index = {self.t[i]: i for i in range(len(self.t))}

        for i, state_var in enumerate(model.organism.state.STATE_VARS):
            setattr(self, state_var, ode_sol.y[i, :])
        # self.E, self.V, self.E_H, self.E_R = ode_sol.y

        self.time_instant_sols = [TimeInstantSol(self.model, self.t[i], ode_sol.y[:, i]) for i in range(len(self.t))]

        # TODO: create attributes for each of the stage transitions
        self.time_of_birth = None
        self.time_of_weaning = None
        self.time_of_puberty = None
        self.calculate_stage_transitions()

    def calculate_stage_transitions(self):
        """Calculates the time step of life stage transitions."""
        # TODO: set to be one before
        for t, E_H in zip(self.t, self.E_H):
            if not self.time_of_birth and E_H > self.model.organism.E_Hb:
                self.time_of_birth = t
            elif not self.time_of_weaning and hasattr(self.model.organism, 'E_Hx'):
                if E_H > self.model.organism.E_Hx:
                    self.time_of_weaning = t
            elif not self.time_of_puberty and E_H > self.model.organism.E_Hp:
                self.time_of_puberty = t

    def to_csv(self, variables, csv_filename=None):
        """
        Saves the time steps and variables of TimeIntervalSol to a .csv file.
        :param variables: string or list of strings of variable names
        :param csv_filename: optional, name of .csv file to save values. If omitted, the filename will be generated
        using the variable names.
        """
        if not isinstance(variables, (list, tuple)):
            variables = list(variables)
        if csv_filename is None:
            csv_filename = f"{'-'.join(variables)}.csv"
        var_data = []
        for v in variables:
            if hasattr(self, v):
                var_data.append(getattr(self, v))
        with open(csv_filename, 'w') as f:
            print(f"time,{','.join(variables)}", file=f)
            for data in zip(self.t, *var_data):
                print(f"{str(data).replace(' ', '')[1:-1]}", file=f)

    def __getitem__(self, time_step):
        # TODO: Check for time steps outside of simulation time
        # TODO: Case for slice of time steps
        if isinstance(time_step, int):
            return self.time_instant_sols[time_step]
        elif isinstance(time_step, (float, np.float64)):
            if time_step in self._time_to_index:
                return self.time_instant_sols[self._time_to_index[time_step]]
            # Interpolate if time_step doesn't exist
            else:
                ti2 = self.find_closest_time_step(time_step)
                ti1 = ti2 - 1
                sol1 = self.time_instant_sols[ti1]
                sol2 = self.time_instant_sols[ti2]
                t1, t2 = sol1.t, sol2.t
                st1 = np.array([sol1.E, sol1.V, sol1.E_H, sol1.E_R])
                st2 = np.array([sol2.E, sol2.V, sol2.E_H, sol2.E_R])

                state = (st2 - st1) / (t2 - t1) * (time_step - t1) + st1
                return TimeInstantSol(self.model, time_step, state)

    def find_closest_time_step(self, time_step):
        for i, t in enumerate(self.t):
            if t > time_step:
                t_i = i
                break
        return t_i

    @staticmethod
    def find_closest_value(variable, value):
        for i, v in enumerate(variable):
            if v > value:
                time_step = i
                break
        return time_step

    def __setitem__(self, key, value):
        return

    def __delitem__(self, key):
        # TODO: Delete a time step
        return

    def __len__(self):
        return len(self.t)

    def insert(self, index: int, value) -> None:
        return

    # TODO: See if adding attributes to self.__dict__ helps in any way
    def __getattr__(self, item):
        tsol = self.time_instant_sols[-1]
        if hasattr(tsol, item):
            if isinstance(getattr(tsol, item), np.ndarray):
                return np.concatenate([getattr(sol, item) for sol in self.time_instant_sols], axis=1)
            else:
                return np.array([getattr(sol, item) for sol in self.time_instant_sols])
        else:
            raise AttributeError

    def integrate(self, variable, t1=0, t2=-1):
        if t2 > -1:
            t2 += 1
        elif t2 == -1:
            t2 = self.__len__()
        return -simpson(variable[t1:t2], self.t[t1:t2])

    def total_feed_intake(self, t1=0, t2=-1):
        if t2 > -1:
            t2 += 1
        elif t2 == -1:
            t2 = self.__len__()
        return -simpson(self.feed_intake[t1:t2], self.t[t1:t2])

    def daily_feed_intake(self, t1=0, t2=-1):
        return self.total_feed_intake(t1, t2) / (self.t[t2] - self.t[t1])

    def average_daily_gain(self, t1=0, t2=-1):
        return (self.time_instant_sols[t2].wet_weight - self.time_instant_sols[t1].wet_weight) / \
            (self.t[t2] - self.t[t1])

    def feed_conversion_ratio(self, t1=0, t2=-1):
        return self.total_feed_intake(t1, t2) / \
            (self.time_instant_sols[t2].wet_weight - self.time_instant_sols[t1].wet_weight)

    def relative_growth_rate(self, t1=0, t2=-1):
        return (np.log(self.time_instant_sols[t2].wet_weight) - np.log(self.time_instant_sols[t1].wet_weight)) / \
            (self.t[t2] - self.t[t1])

    def compute_emissions(self, t1=0, t2=-1):
        # TODO: Add methane emissions if it is a RUM model
        if t2 > -1:
            t2 += 1
        elif t2 == -1:
            t2 = self.__len__()
        return -simpson(self.carbon_dioxide[t1:t2], self.t[t1:t2])

    def print_growth_report(self, t1=0, t2=-1):
        w1 = self.time_instant_sols[t1].wet_weight
        w2 = self.time_instant_sols[t2].wet_weight
        print(f"Initial Weight: {w1 / 1000} kg\n"
              f"Final Weight: {w2 / 1000} kg\n")
        print(f"DFI: {self.daily_feed_intake(t1, t2):.5} g\n"
              f"ADG: {self.average_daily_gain(t1, t2):.4} g\n"
              f"FCR: {self.feed_conversion_ratio(t1, t2):.4}\n"
              f"RGR: {self.relative_growth_rate(t1, t2) * 100:.4} %")
