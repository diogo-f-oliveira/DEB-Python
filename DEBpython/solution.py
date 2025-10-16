from collections.abc import MutableSequence
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from scipy.integrate import simpson

from DEBpython.state import State


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

    @property
    def state_values(self):
        return self.state.state_values

    @property
    def t(self):
        return self.state.t


class TimeIntervalSol(MutableSequence):
    """
    TimeIntervalSol class:

    Stores the complete solution to the integration of state equations, including state variables, powers, fluxes,
    entropy and real variables such as physical length, as well as time of stage transitions.
    """
    _TIME_TOL = 1e-10

    def __init__(self, model, ode_sol):
        """
        Creates an instance of TimeIntervalSol from a model containing a simulation of the organism.
        :param model: model class
        """
        self.model = model

        self.t = ode_sol.t
        # self._time_to_index = {self.t[i]: i for i in range(len(self.t))}

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

    @property
    def time_bounds(self):
        """Return  bounds of the simulation time."""
        return self.time_instant_sols[0].t, self.time_instant_sols[-1].t

    def _assert_in_domain(self, t):
        """Raise IndexError if ``t`` is outside time bounds of the simulation."""
        t = float(t)
        lo, hi = self.time_bounds
        if t < lo - self._TIME_TOL or t > hi + self._TIME_TOL:
            raise IndexError(f"Requested time {t} is outside simulation time [{lo}, {hi}].")
        return t

    def __getitem__(self, key):
        """Index-only access.

        - ``int`` → ``TimeInstantSol`` at that index (supports negative indices).
        - ``slice`` of **ints** → smaller ``TimeIntervalSol`` restricted to the existing grid.

        Time-based access is provided by explicit methods: ``at_time(t)``, ``window(t_start, t_stop)``,
        and ``resample(dt, ...)``.
        """
        # Single integer index
        if isinstance(key, (int, np.integer)):
            return self.time_instant_sols[key]

        # Slice of integers (no floats allowed here)
        if isinstance(key, slice):
            for part in (key.start, key.stop, key.step):
                if part is None:
                    continue
                if not isinstance(part, (int, np.integer)):
                    raise TypeError("Index slicing is index-based only. "
                                    "For time-based slicing or resampling, use .window() or .resample().")
            idx = np.arange(len(self.t))[key]
            times = self.t[idx]
            var_names = self.model.organism.state.STATE_VARS
            y = np.vstack([getattr(self, name)[idx] for name in var_names])
            ode_like = SimpleNamespace(t=np.asarray(times, dtype=float), y=y)
            return TimeIntervalSol(self.model, ode_like)

        raise TypeError("Use integers or slices of integers for indexing. "
                        "For time-based access, use .at_time(t), .window(...), or .resample(dt, ...).")

    def find_closest_time_step(self, time_step):
        """Return the index of the first time strictly greater than ``time_step``.

        If ``time_step`` is greater than or equal to the last grid time, returns ``len(t)``.
        (Callers should handle bounds before calling when needed.)
        """
        t = np.asarray(self.t)
        idx = np.searchsorted(t, time_step, side='right')
        return idx

    def index_for_time(self, t):
        """Return the index of a grid time matching ``t`` within ``tol``; otherwise ``None``.

        Uses ``np.searchsorted`` to check nearest neighbors for closeness.
        """
        t = float(t)
        arr = np.asarray(self.t)
        i = int(np.searchsorted(arr, t))
        candidates = []
        if i < len(arr):
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        for j in candidates:
            if abs(arr[j] - t) <= self._TIME_TOL:
                return j
        return None

    def at_time(self, t):
        """Return a ``TimeInstantSol`` at time ``t`` (linear interpolation if off-grid)."""
        self._assert_in_domain(t)

        j = self.index_for_time(t)
        if j is not None:
            return self.time_instant_sols[j]

        # Interpolate between neighbors
        arr = np.asarray(self.t)
        j_right = int(np.searchsorted(arr, t, side='right'))
        if j_right == 0:
            return self.time_instant_sols[0]
        if j_right >= len(arr):
            return self.time_instant_sols[-1]
        j_left = j_right - 1
        s1 = self.time_instant_sols[j_left]
        s2 = self.time_instant_sols[j_right]
        interp_state = self.linear_interpolation(
            t=t, t1=s1.t, state1=s1.state.state_values, t2=s2.t, state2=s2.state.state_values
        )
        return TimeInstantSol(self.model, t, interp_state)

    def window(self, t_start, t_stop, include_end=True):
        """Return a ``TimeIntervalSol`` for ``[t_start, t_stop]`` with interpolated endpoints.

        The interior points come from the existing grid strictly inside the window.
        """
        if t_start > t_stop:
            t_start, t_stop = t_stop, t_start
        self._assert_in_domain(t_start)
        self._assert_in_domain(t_stop)

        j_start = self.index_for_time(t_start)
        j_stop = self.index_for_time(t_stop)
        t_start_eff = float(self.t[j_start]) if j_start is not None else t_start
        t_stop_eff = float(self.t[j_stop]) if j_stop is not None else t_stop

        mask = (self.t > t_start_eff + self._TIME_TOL) & (self.t < t_stop_eff - self._TIME_TOL)
        times = [t_start_eff]
        times.extend(self.t[mask].astype(float).tolist())
        if include_end and (len(times) == 0 or not np.isclose(times[-1], t_stop_eff, atol=self._TIME_TOL)):
            times.append(t_stop_eff)
        return self._build_subinterval_from_times(np.asarray(times, dtype=float))

    def resample(self, dt, t_start=None, t_stop=None, include_end=True):
        """Uniformly resample the solution every ``dt`` time units in ``[t_start, t_stop]``.

        Endpoints are included when ``include_end`` is True (default). ``dt`` must be > 0.
        """
        if dt is None:
            raise ValueError("dt must be provided")
        dt = float(dt)
        if not (dt > 0):
            raise ValueError("dt must be positive")
        tol = self._TIME_TOL
        if t_start is None:
            t_start = float(self.t[0])
        if t_stop is None:
            t_stop = float(self.t[-1])
        t_start = float(t_start)
        t_stop = float(t_stop)
        if t_start > t_stop:
            t_start, t_stop = t_stop, t_start
        lo, hi = float(self.t[0]), float(self.t[-1])
        if t_start < lo - tol or t_stop > hi + tol:
            raise IndexError(f"Resample range [{t_start}, {t_stop}] is outside simulation time [{lo}, {hi}].")

        n = max(1, int(np.floor((t_stop - t_start) / dt)))
        times = t_start + dt * np.arange(n + 1, dtype=float)
        if include_end and (times.size == 0 or times[-1] < t_stop - 1e-12):
            times = np.append(times, t_stop)
        if not include_end and times.size > 0 and np.isclose(times[-1], t_stop, atol=tol):
            times = times[:-1]
        return self._build_subinterval_from_times(times)

    def _build_subinterval_from_times(self, times):
        """Create a new TimeIntervalSol by sampling the current solution at ``times``.

        For times not on the original grid, linear interpolation is used to build a
        TimeInstantSol so that all derived quantities (powers/fluxes) are recomputed at that time.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty 1D array")

        # Build list of TimeInstantSol objects at requested times (reuse existing or interpolate)
        instants = [self.at_time(float(t)) for t in times]

        # Assemble state matrix y with shape (n_state_vars, n_times)
        n_state = len(self.model.organism.state.STATE_VARS)
        y = np.empty((n_state, len(instants)), dtype=float)
        for j, sol in enumerate(instants):
            y[:, j] = sol.state.state_values

        ode_like = SimpleNamespace(t=np.asarray(times, dtype=float), y=y)
        return TimeIntervalSol(self.model, ode_like)

    # def __getitem__(self, time_step):
    #     # TODO: Check for time steps outside of simulation time
    #     # TODO: Case for slice of time steps
    #     if isinstance(time_step, int):
    #         return self.time_instant_sols[time_step]
    #     elif isinstance(time_step, (float, np.float64)):
    #         if time_step in self._time_to_index:
    #             return self.time_instant_sols[self._time_to_index[time_step]]
    #         # Interpolate if time_step doesn't exist
    #         else:
    #             ti2 = self.find_closest_time_step(time_step)
    #             # TODO: Check if ti2 is at the beginning of the simulation
    #             ti1 = ti2 - 1
    #             sol1 = self.time_instant_sols[ti1]
    #             sol2 = self.time_instant_sols[ti2]
    #             interpolated_state = self.linear_interpolation(t=time_step,
    #                                                            t1=sol1.t, state1=sol1.state.state_values,
    #                                                            t2=sol2.t, state2=sol2.state.state_values)
    #             # t1, t2 = sol1.t, sol2.t
    #             # st1 = np.array([sol1.E, sol1.V, sol1.E_H, sol1.E_R])
    #             # st2 = np.array([sol2.E, sol2.V, sol2.E_H, sol2.E_R])
    #             #
    #             # state = (st2 - st1) / (t2 - t1) * (time_step - t1) + st1
    #             return TimeInstantSol(self.model, time_step, interpolated_state)
    #
    # def find_closest_time_step(self, time_step):
    #     for i, t in enumerate(self.t):
    #         if t > time_step:
    #             t_i = i
    #             break
    #     return t_i
    #
    @staticmethod
    def find_closest_value(variable, value):
        for i, v in enumerate(variable):
            if v > value:
                time_step = i
                break
        return time_step

    @staticmethod
    def linear_interpolation(t, t1, state1, t2, state2):
        return (state2 - state1) / (t2 - t1) * (t - t1) + state1

    def __setitem__(self, key, value):
        # TODO: Add a time step in the correct place
        return

    def __delitem__(self, key):
        # TODO: Delete a time step
        return

    def __len__(self):
        return len(self.t)

    def insert(self, index: int, value) -> None:
        return

    # Auto-timeseries accessor: fetch any attribute defined on TimeInstantSol across all time steps.
    def __getattr__(self, name):
        """Return the time series for an attribute defined on ``TimeInstantSol``.

        Examples
        --------
        If each ``TimeInstantSol`` has a scalar ``wet_weight`` (float), ``interval.wet_weight``
        returns a 1D array of shape (T,), where T is the number of time steps.

        If an attribute is a vector (shape (K,)), the result is stacked with time on axis 0:
        shape (T, K).

        For higher-dimensional arrays, values are stacked along a new leading time axis:
        shape (T, ...).

        The computed series is cached on the instance under the same name to avoid recomputation.
        """
        # Only invoked if normal attribute lookup fails; avoid recursion by using __dict__ directly
        if not self.time_instant_sols:
            raise AttributeError(name)

        sample = self.time_instant_sols[0]
        if not hasattr(sample, name):
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}', "
                                 f"and it is not present on TimeInstantSol.")

        # Ignore callables (methods/properties that return callables)
        v0 = getattr(sample, name)
        if callable(v0):
            raise AttributeError(f"Attribute '{name}' on TimeInstantSol is callable; not a data series.")

        # Gather values from all time instants
        values = []
        for sol in self.time_instant_sols:
            if not hasattr(sol, name):
                raise AttributeError(f"Not all TimeInstantSol instances define attribute '{name}'. "
                                     f"Cannot build a consistent time series."                )
            values.append(getattr(sol, name))

        a0 = np.asarray(values[0])
        try:
            if a0.ndim == 0:
                # Scalar per time -> (T,)
                out = np.asarray(values)
            else:
                # Vector/array per time -> stack with time as the first axis -> (T, ...)
                out = np.stack([np.asarray(v) for v in values], axis=0)
        except Exception:
            # Inconsistent shapes -> fall back to object array to avoid silent shape errors
            out = np.array(values, dtype=object)

        # Cache result for subsequent accesses
        self.__dict__[name] = out
        return out

    def integrate(self, variable, t1=0, t2=-1):
        if t2 > -1:
            t2 += 1
        elif t2 == -1:
            t2 = self.__len__()
        return -simpson(variable[t1:t2], self.t[t1:t2])

    # TODO: Create a class based on TimeIntervalSol that implements the following methods
    # def total_feed_intake(self, t1=0, t2=-1):
    #     if t2 > -1:
    #         t2 += 1
    #     elif t2 == -1:
    #         t2 = self.__len__()
    #     return -simpson(self.feed_intake[t1:t2], self.t[t1:t2])
    #
    # def daily_feed_intake(self, t1=0, t2=-1):
    #     return self.total_feed_intake(t1, t2) / (self.t[t2] - self.t[t1])
    #
    # def average_daily_gain(self, t1=0, t2=-1):
    #     return (self.time_instant_sols[t2].wet_weight - self.time_instant_sols[t1].wet_weight) / \
    #         (self.t[t2] - self.t[t1])
    #
    # def feed_conversion_ratio(self, t1=0, t2=-1):
    #     return self.total_feed_intake(t1, t2) / \
    #         (self.time_instant_sols[t2].wet_weight - self.time_instant_sols[t1].wet_weight)
    #
    # def relative_growth_rate(self, t1=0, t2=-1):
    #     return (np.log(self.time_instant_sols[t2].wet_weight) - np.log(self.time_instant_sols[t1].wet_weight)) / \
    #         (self.t[t2] - self.t[t1])
    #
    # def compute_emissions(self, t1=0, t2=-1):
    #     # TODO: Add methane emissions if it is a RUM model
    #     if t2 > -1:
    #         t2 += 1
    #     elif t2 == -1:
    #         t2 = self.__len__()
    #     return -simpson(self.carbon_dioxide[t1:t2], self.t[t1:t2])
    #
    # def print_growth_report(self, t1=0, t2=-1):
    #     w1 = self.time_instant_sols[t1].wet_weight
    #     w2 = self.time_instant_sols[t2].wet_weight
    #     print(f"Initial Weight: {w1 / 1000} kg\n"
    #           f"Final Weight: {w2 / 1000} kg\n")
    #     print(f"DFI: {self.daily_feed_intake(t1, t2):.5} g\n"
    #           f"ADG: {self.average_daily_gain(t1, t2):.4} g\n"
    #           f"FCR: {self.feed_conversion_ratio(t1, t2):.4}\n"
    #           f"RGR: {self.relative_growth_rate(t1, t2) * 100:.4} %")
