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
        self.t = 0
        # State variables
        for var, default_type in self.STATE_VARS.items():
            setattr(self, var, default_type())

        # Environmental variables
        for var, default_type in self.ENV_VARS.items():
            setattr(self, var, default_type())

    def set_state_vars(self, state_values):
        if len(state_values) != len(self.STATE_VARS):
            raise ValueError('Input state_values does not have the defined number of states len(self.STATE_VARS).')
        for i, (var, val) in enumerate(zip(self.STATE_VARS, state_values)):
            if not isinstance(val, self.STATE_VARS[var]):
                raise ValueError(f'State value {val} is not of the defined type for state variable {var} '
                                 f'({self.STATE_VARS[var]}).')
            setattr(self, var, val)

    def set_environment_state(self, env_state_values):
        if len(env_state_values) != len(self.ENV_VARS):
            raise ValueError('Input state_values does not have the defined number of states len(self.STATE_VARS).')
        for i, (var, val) in enumerate(zip(self.ENV_VARS, env_state_values)):
            if not isinstance(val, self.ENV_VARS[var]):
                raise ValueError(f'Environment state value {val} is not of the defined type for environment variable '
                                 f'{var} ({self.ENV_VARS[var]}).')
            setattr(self, var, val)

    @property
    def state(self):
        return (getattr(self, var) for var in self.STATE_VARS)

    @property
    def environment_state(self):
        return (getattr(self, var) for var in self.ENV_VARS)
