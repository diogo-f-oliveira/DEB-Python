from collections.abc import MutableSequence
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

        self.organism = model.organism

        # Save state variables
        self.t = t
        self.E, self.V, self.E_H, self.E_R = state_vars

        # Apply transform to the organism
        model.transform(self.organism, t, state_vars)

        # Powers
        powers = model.compute_powers(t, state_vars, compute_dissipation_power=True)
        self.p_A, self.p_C, self.p_S, self.p_G, self.p_J, self.p_R, self.p_D = powers

        # Fluxes
        self.organic_fluxes = model.organic_fluxes(self.p_A, self.p_D, self.p_G, self.E_H)
        self.mineral_fluxes = model.mineral_fluxes(self.p_A, self.p_D, self.p_G, self.E_H)
        # Entropy
        self.entropy = model.entropy_generation(self.p_A, self.p_D, self.p_G, self.E_H)[0]
        # Physical Length
        self.calculate_real_variables()

    def calculate_real_variables(self):
        """Computes real variables such as physical length, etc... (WIP)"""
        self.physical_length = self.organism.compute_physical_length(self.V)
        self.physical_volume = self.organism.compute_physical_volume(self.V, self.E, self.E_R)
        self.wet_weight = self.organism.compute_wet_weight(self.V, self.E, self.E_R)
        self.dry_weight = self.organism.compute_dry_weight(self.V, self.E, self.E_R)


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

        self.E, self.V, self.E_H, self.E_R = ode_sol.y

        self._time_instant_sols = [TimeInstantSol(self.model, self.t[i], ode_sol.y[:, i]) for i in range(len(self.t))]

        self.time_of_birth = None
        self.time_of_weaning = None
        self.time_of_puberty = None
        self.calculate_stage_transitions()

    def calculate_stage_transitions(self):
        """Calculates the time step of life stage transitions."""
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
        # TODO: Interpolate when the time step does not exist.
        if isinstance(time_step, int):
            return self._time_instant_sols[time_step]
        elif isinstance(time_step, (float, np.float64)):
            if time_step in self._time_to_index:
                return self._time_instant_sols[self._time_to_index[time_step]]

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
        tsol = self._time_instant_sols[-1]
        if hasattr(tsol, item):
            if isinstance(getattr(tsol, item), np.ndarray):
                return np.concatenate([getattr(sol, item) for sol in self._time_instant_sols], axis=1)
            else:
                return np.array([getattr(sol, item) for sol in self._time_instant_sols])
        else:
            raise AttributeError

    @property
    def total_feed_intake(self):
        return -simpson(self.organic_fluxes[0, :], self.t) * self._time_instant_sols[-1].organism.comp.X.w

    @property
    def daily_feed_intake(self):
        return self.total_feed_intake / (self.t[-1] - self.t[0])

    @property
    def average_daily_gain(self):
        return (self._time_instant_sols[-1].wet_weight - self._time_instant_sols[0].wet_weight) / \
               (self.t[-1] - self.t[0])

    @property
    def feed_consumption_ratio(self):
        return self.total_feed_intake / (self._time_instant_sols[-1].wet_weight - self._time_instant_sols[0].wet_weight)

    @property
    def relative_growth_rate(self):
        return (np.log(self._time_instant_sols[-1].wet_weight) - np.log(self._time_instant_sols[0].wet_weight)) / \
               (self.t[-1] - self.t[0])
