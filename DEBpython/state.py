import numpy as np

from .composition import Compound


class State:
    STATE_VARS = {
        'E': float,
        'V': float,
        'E_H': float,
        'E_R': float,
    }
    ENV_VARS = {
        'T': float,
        'p_X': float,
        'food_comp': Compound.food,
    }

    def __init__(self):
        # Time
        self.t = 0.
        # State variables
        for var, default_type in self.STATE_VARS.items():
            setattr(self, var, default_type())

        # Environmental variables
        for var, default_type in self.ENV_VARS.items():
            setattr(self, var, default_type())

    def set_state_vars(self, state_values):
        valid, reason = self.check_state_validity(state_values)
        if not valid:
            raise ValueError(reason)
        for i, (var, val) in enumerate(zip(self.STATE_VARS, state_values)):
            setattr(self, var, val)

    def set_environment_state(self, env_values):
        valid, reason = self.check_environment_validity(env_values)
        if not valid:
            raise ValueError(reason)
        for i, (var, val) in enumerate(zip(self.ENV_VARS, env_values)):
            setattr(self, var, val)

    @property
    def state_values(self):
        return np.array(list(getattr(self, var) for var in self.STATE_VARS))

    @property
    def state_names(self):
        return list(self.STATE_VARS.keys())

    @property
    def env_state_values(self):
        return np.array(getattr(self, var) for var in self.ENV_VARS)

    def check_state_validity(self, state_values):
        if len(state_values) != len(self.STATE_VARS):
            valid = False
            reason = f'Input state_values does not have the correct number of states {len(self.STATE_VARS)}.'
            return valid, reason
        for i, (var, val) in enumerate(zip(self.STATE_VARS, state_values)):
            if not isinstance(val, self.STATE_VARS[var]):
                valid = False
                reason = (f'State value {val} is not of the defined type for state variable {var} '
                          f'({self.STATE_VARS[var]}).')
                return valid, reason
        return True, ''

    def check_environment_validity(self, env_values):
        if len(env_values) != len(self.ENV_VARS):
            valid = False
            reason = f'Input env_state_values does not have the correct number of variables {len(self.ENV_VARS)}.'
            return valid, reason
        for i, (var, val) in enumerate(zip(self.ENV_VARS, env_values)):
            if not isinstance(val, self.ENV_VARS[var]):
                valid = False
                reason = (f'Environment value {val} is not of the defined type for environment variable {var} '
                          f'({self.ENV_VARS[var]}).')
                return valid, reason
        return True, ''


class ABJState(State):
    STATE_VARS = {
        'E': float,
        'V': float,
        'E_H': float,
        'E_R': float,
        's_Hjb': float,
    }
    ENV_VARS = {
        'T': float,
        'p_X': float,
        'food_comp': Compound.food,
    }
