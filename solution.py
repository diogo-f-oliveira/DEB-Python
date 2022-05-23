class TimeInstantSol:
    """
        TimeIntervalSol class:

        Stores the complete state of the organism at a given time step, including state variables, powers,  fluxes,
        entropy and real variables such as physical length.
        """
    def __init__(self, model, t, state_vars):
        self.model_type = type(model).__name__

        self.organism = model.organism

        # Save state variables
        self.t = t
        self.E, self.V, self.E_H, self.E_R = state_vars

        # Powers
        self.calculate_powers(model)
        # Fluxes
        self.mineral_fluxes = model.mineral_fluxes(self.p_A, self.p_D, self.p_G)
        # Entropy
        self.entropy = model.entropy_generation(self.p_A, self.p_D, self.p_G)
        # Physical Length
        self.calculate_real_variables()

    def calculate_powers(self, model):
        """Computes all powers over every time step."""
        if self.model_type == 'STD':
            self.p_A = model.p_A(self.V, self.E_H, self.t)
            self.p_C = model.p_C(self.E, self.V)
            self.p_S = model.p_S(self.V)
            self.p_G = model.p_G(self.p_C, self.p_S)
            self.p_J = model.p_J(self.E_H)
            self.p_R = model.p_R(self.p_C, self.p_J)
            self.p_D = model.p_D(self.p_S, self.p_J, self.p_R, self.E_H)
        elif self.model_type == 'STX':
            self.p_A = model.p_A(self.V, self.E_H, self.t)
            self.p_C = model.p_C(self.E, self.V)
            self.p_S = model.p_S(self.V)
            self.p_G = model.p_G(self.p_C, self.p_S, self.V, self.E_H)
            self.p_J = model.p_J(self.E_H)
            self.p_R = model.p_R(self.p_C, self.p_J, self.p_S, self.p_G, self.E_H)
            self.p_D = model.p_D(self.p_S, self.p_J, self.p_R, self.E_H)

    def calculate_real_variables(self):
        """Computes real variables such as physical length, etc... (WIP)"""
        self.physical_length = self.organism.convert_to_physical_length(self.V)


class TimeIntervalSol(TimeInstantSol):
    """
    TimeIntervalSol class:

    Stores the complete solution to the integration of state equations, including state variables, powers, fluxes,
    entropy and real variables such as physical length, as well as time of stage transitions.
    """

    def __init__(self, model):

        # (model.ode_sol.y[0], model.ode_sol.y[1], model.ode_sol.y[2], model.ode_sol.y[3])
        super().__init__(model, model.ode_sol.t, model.ode_sol.y)

        self.time_of_birth = None
        self.time_of_weaning = None
        self.time_of_puberty = None
        self.calculate_stage_transitions()

    def calculate_stage_transitions(self):
        """Calculates the time step of life stage transitions."""
        for t, E_H in zip(self.t, self.E_H):
            if not self.time_of_birth and E_H > self.organism.E_Hb:
                self.time_of_birth = t
            elif not self.time_of_weaning and hasattr(self.organism, 'E_Hx'):
                if E_H > self.organism.E_Hx:
                    self.time_of_weaning = t
            elif not self.time_of_puberty and E_H > self.organism.E_Hp:
                self.time_of_puberty = t
