from scipy.integrate import solve_ivp
import numpy as np
from pet import Pet, Ruminant
import solution


class STD:
    """
    class STD:

        Standard DEB model.
        Takes as input a Pet class.
        Calculates all fluxes based on state variables: Reserve (E), Structure (V), Maturity (E_H) and Reproduction
        Buffer (E_R).
        Integrates all state variables over time according to an input function of scaled functional feeding response
        (f) over time.
    """

    MAX_STEP_SIZE = 48 / 24  # Maximum step size during integration of state equations

    def __init__(self, organism):
        """Takes as input a Pet class or a dictionary of parameters to create a Pet class."""

        # Create the Pet class from the dictionary of parameters
        if isinstance(organism, dict):
            organism = Pet(**organism)
        # Check that organism is a Pet class
        elif not isinstance(organism, Pet):
            raise Exception("Input must be of class Pet or a dictionary of parameters to create a Pet class.")

        self.filter_pet(organism)

        self.organism = organism
        self.ode_sol = None  # Output from ODE solver
        self.sol = None  # Full solution including powers, fluxes and entropy
        self.food_function = None  # Function of scaled functional feeding response (f) over time

    @staticmethod
    def filter_pet(organism):
        """Ensures that the parameters of the organism are valid."""
        # Check validity of parameters of Pet
        valid, reason = organism.check_validity()
        if not valid:
            raise Exception(f"Invalid Pet parameters. {reason}")

    def simulate(self, t_span, food_function=1, step_size='auto', initial_state='embryo', transformation=None):
        """
        Integrates state equations over time. The output from the solver is stored in self.ode_sol.

        :param food_function: Function of scaled functional feeding response (f) over time. Must be of signature
            f = food_function(time). If a numerical input between 0 and 1 is provided, a food_function for constant
            scaled functional response is created.
        :param t_span: (t0, tf). Interval of integration. The solver starts at t=t0 and integrates until it reaches
            t=tf.
        :param step_size: Step size of integration. If step_size='auto', the solver will decide the step size. Else
            input a numerical value for fixed step size.
        :param initial_state: Values of state variables at time t0. Format is (E, V, E_H, E_R). If initial_state='birth'
            the state variables are initialized with the values for birth (E_0, V_0, 0, 0), where E_0 and V_0 are
            defined in the Pet class.
        :param transformation: Function that changes parameters of Pet during simulation. Takes as input the organism,
        the time step t and the state variables at time t. Can be used for example to change food characteristics
        or temperature.
        """

        # Get initial state
        if initial_state == 'embryo':
            initial_state = (self.organism.E_0, self.organism.V_0, 0, 0)
        elif len(initial_state) != 4:
            raise Exception(f"Invalid input {initial_state} for initial state. The initial state must be a list or "
                            f"tuple of length 4 with format (E, V, E_H, E_R).")

        # Store the food function
        if isinstance(food_function, (float, int)):
            if food_function < 0 or food_function > 1:
                raise Exception("The scaled functional response f must be between 0 and 1.")
            else:
                self.food_function = lambda t: food_function
        elif callable(food_function):
            self.food_function = food_function
        else:
            raise Exception("Argument food_function must be a number between 0 and 1 or callable.")

        # Define the times at which the solver should store the computed solution.
        if step_size == 'auto':
            t_eval = None
        elif isinstance(step_size, (float, int)):
            t_eval = np.arange(*t_span, step_size)
        else:
            raise Exception(f"Invalid step size value: {step_size}. Please select 'auto' for automatic step size during"
                            f" integration or input a fixed step size.")

        # Transformations to Pet parameters during simulation
        if callable(transformation):
            self.transform = transformation
        elif transformation is not None:
            raise Exception("Transformation function must be callable.")

        # Integrate the state equations
        self.ode_sol = solve_ivp(self.state_changes, t_span, initial_state, t_eval=t_eval, max_step=self.MAX_STEP_SIZE)
        self.sol = solution.TimeIntervalSol(self, self.ode_sol)
        return self.sol

    def transform(self, organism, t, state_vars):
        return

    def fully_grown(self, f=1, E_R=0):
        """
        Returns a TimeInstantSol of the organism at full growth for a given food level and reproduction buffer
        :param f: scaled functional feeding response f
        :param E_R: reproduction buffer E_R
        :return: TimeInstantSol of the organism at full growth
        """
        # Create food function
        self.food_function = lambda t: f

        # State variables at full growth
        state_vars = (self.organism.E_m * (self.organism.ultimate_length(f)) ** 3,
                      (self.organism.ultimate_length(f)) ** 3,
                      self.organism.E_Hp,
                      E_R)
        return solution.TimeInstantSol(self, 0, state_vars)

    def state_changes(self, t, state_vars):
        """
        Computes the derivatives of the state variables according to the standard DEB model equations. Function used in
        the integration solver.
        :param t: time
        :param state_vars: tuple of state variables (E, V, E_H, E_R)
        :return: derivatives of the state variables (dE, dV, dE_H, dE_R)
        """
        # Unpacking state variables (Reserve (E), Structure (E), Maturity (E_H), Reproduction Buffer (E_R))
        E, V, E_H, E_R = state_vars

        # Apply transform to Pet parameters
        self.transform(self.organism, t, state_vars)

        # Compute powers
        p_A, p_C, p_S, p_G, p_J, p_R, p_D = self.compute_powers(t, state_vars)

        # Changes to state variables
        dE = p_A - p_C
        dV = p_G / self.organism.E_G

        # Maturity or Reproduction Buffer logic
        if E_H < self.organism.E_Hp:
            dE_H = p_R
            dE_R = 0
        else:
            dE_H = 0
            dE_R = self.organism.kap_R * p_R
        return dE, dV, dE_H, dE_R

    def compute_powers(self, t, state_vars, compute_dissipation_power=False):
        """
        Computes all powers, with the option of also calculating the dissipation power p_D
        :param t: time
        :param E: reserve
        :param V: structure
        :param E_H: maturity
        :param E_R: reproduction buffer
        :param compute_dissipation_power: whether to compute the dissipation power
        :return: tuple of powers (p_A, p_C, p_S, p_G, p_J, p_R, p_D)
        """
        # Unpacking state variables (Reserve (E), Structure (E), Maturity (E_H), Reproduction Buffer (E_R))
        E, V, E_H, E_R = state_vars

        # Computing powers
        p_A = self.p_A(V, E_H, t)  # Assimilation power
        p_C = self.p_C(E, V)  # Mobilization power
        p_S = self.p_S(V)  # Somatic maintenance power
        p_G = self.p_G(p_C, p_S)  # Growth power
        p_J = self.p_J(E_H)  # Maturity maintenance power
        p_R = self.p_R(p_C, p_J)  # Reproduction power

        # Dissipation power
        if compute_dissipation_power:
            p_D = self.p_D(p_S, p_J, p_R, E_H)
        else:
            p_D = 0

        return p_A, p_C, p_S, p_G, p_J, p_R, p_D

    def p_A(self, V, E_H, t):
        """
        Computes the assimilation power p_A.

        :param V: Scalar or array of Structure values
        :param E_H: Scalar or array of Maturity values
        :param t: Scalar or array of Time values
        :return: Scalar or array of assimilation power p_A values
        """
        if type(E_H) == np.ndarray:
            # Preallocate p_A
            p_A = np.zeros_like(E_H)
            for i, (structure, maturity, time) in enumerate(zip(V, E_H, t)):
                if maturity < self.organism.E_Hb:  # Embryo life stage
                    p_A[i] = 0
                else:
                    p_A[i] = self.organism.p_Am * self.food_function(time) * (structure ** (2 / 3))
            return p_A
        else:
            if E_H < self.organism.E_Hb:  # Embryo life stage
                return 0
            else:
                return self.organism.p_Am * self.food_function(t) * (V ** (2 / 3))

    def p_C(self, E, V):
        """
        Computes the mobilization power p_C.

        :param E: Scalar or array of Reserve values
        :param V: Scalar or array of Structure values
        :return: Scalar or array of mobilization power p_C values
        """
        return E * (self.organism.E_G * self.organism.v * (V ** (-1 / 3)) + self.organism.p_M) / \
               (self.organism.kappa * E / V + self.organism.E_G)

    def p_S(self, V):
        """
        Computes the somatic maintenance power p_S.

        :param V: Scalar or array of Structure values
        :return: Scalar or array of somatic maintenance power p_S values
        """
        return self.organism.p_M * V + self.organism.p_T * (V ** (2 / 3))

    def p_G(self, p_C, p_S):
        """
        Computes the growth power p_G.

        :param p_C: Scalar or array of mobilization power values
        :param p_S: Scalar or array of somatic maintenance power values
        :return: Scalar or array of growth power p_G values
        """
        return self.organism.kappa * p_C - p_S

    def p_J(self, E_H):
        """
        Computes the maturity maintenance power p_J

        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of maturity maintenance power p_J values
        """
        if type(E_H) == np.ndarray:
            p_J = np.zeros_like(E_H)
            for i, maturity in enumerate(E_H):
                if maturity < self.organism.E_Hp:
                    p_J[i] = self.organism.k_J * maturity
                else:  # Adult life stage
                    p_J[i] = self.organism.k_J * self.organism.E_Hp
            return p_J
        else:
            if E_H < self.organism.E_Hp:
                return self.organism.k_J * E_H
            else:  # Adult life stage
                return self.organism.k_J * self.organism.E_Hp

    def p_R(self, p_C, p_J):
        """
        Computes the reproduction power p_R

        :param p_C: Scalar or array of mobilization power values
        :param p_J: Scalar or array of maturity maintenance power values
        :return: Scalar or array of reproduction power p_R values
        """
        return (1 - self.organism.kappa) * p_C - p_J

    def p_D(self, p_S, p_J, p_R, E_H):
        """
        Computes the dissipation power p_D

        :param p_S: Scalar or array of somatic maintenance power values
        :param p_J: Scalar or array of maturity maintenance power values
        :param p_R: Scalar or array of reproduction power values
        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of dissipation power p_D values
        """
        if type(E_H) == np.ndarray:
            p_D = np.zeros_like(E_H)
            for i, (somatic_power, maturity_power, reproduction_power, maturity) in enumerate(zip(p_S, p_J, p_R, E_H)):
                if maturity < self.organism.E_Hp:
                    p_D[i] = somatic_power + maturity_power + reproduction_power
                else:
                    p_D[i] = somatic_power + maturity_power + (1 - self.organism.kap_R) * reproduction_power
            return p_D
        else:
            if E_H < self.organism.E_Hp:
                return p_S + p_J + p_R
            else:
                return p_S + p_J + (1 - self.organism.kap_R) * p_R

    def mineral_fluxes(self, p_A, p_D, p_G, E_H):
        """
        Computes the mineral fluxes from the assimilation power p_A, dissipation power p_D and growth power p_G.

        :param p_A: Scalar or array of assimilation power values
        :param p_D: Scalar or array of dissipation power values
        :param p_G: Scalar or array of growth power values
        :return: array of mineral fluxes values. Each row corresponds to the flux of CO2, H2O, O2 and N-Waste
            respectively.
        """
        if type(p_A) != np.ndarray:
            p_A = np.array([p_A])
            p_D = np.array([p_D])
            p_G = np.array([p_G])
        powers = np.array([p_A, p_D, p_G])
        return self.organism.eta_M @ powers

    def entropy_generation(self, p_A, p_D, p_G, E_H):
        """
        Computes the entropy from the assimilation power p_A, dissipation power p_D and growth power p_G.
        :param p_A: Scalar or array of assimilation power values
        :param p_D: Scalar or array of dissipation power values
        :param p_G: Scalar or array of growth power values
        :return: scalar or array of entropy values.
        """
        if type(p_A) != np.ndarray:
            p_A = np.array([p_A])
            p_D = np.array([p_D])
            p_G = np.array([p_G])
        powers = np.array([p_A, p_D, p_G])
        return (self.organism.comp.h_M @ np.linalg.inv(self.organism.comp.n_M) @ self.organism.comp.n_O
                - self.organism.comp.h_O) @ self.organism.gamma_O @ powers / self.organism.T / self.organism.comp.E.mu


class STX(STD):
    """
    class STX:

        DEB model STX for mammals.
        Considers fetal development that starts after a preparation time t0. Until maturity E_Hx, the animal feeds on
        milk, which can have a higher nutritional value modelled by the parameter f_milk. Afterwards the animal switches
        to solid food.
        Takes as input a Pet class that must have parameters t_0 and E_Hx.
        Calculates all fluxes based on state variables: Reserve (E), Structure (V), Maturity (E_H) and Reproduction
        Buffer (E_R).
        Integrates all state variables over time according to an input function of scaled functional feeding response
        (f) over time.
    """

    def filter_pet(self, organism):
        """Ensures that the parameters of Pet are valid and that the required parameters for model STX, t_0 and E_Hx,
        are defined."""
        # Checks validity of parameters of Pet
        super().filter_pet(organism)

        # Check that the Pet class has parameters t_0 and E_Hx defined
        if not hasattr(organism, 't_0') or not hasattr(organism, 'E_Hx'):
            raise Exception('The organism is not compatible with model STX: parameters t_0 and E_Hx are required.')
        elif organism.t_0 <= 0:
            raise Exception("The time until start of development can't be negative.")
        elif organism.E_Hx <= organism.E_Hb or organism.E_Hx >= organism.E_Hp:
            raise Exception("The weaning maturity level must be larger than the maturity at birth and smaller than "
                            "maturity at puberty.")
        # Set f_milk to 1 if it is not defined
        if not hasattr(organism, 'f_milk'):
            setattr(organism, 'f_milk', 1)
        elif organism.f_milk <= 0:
            raise Exception("The parameter f_milk must be positive.")
        # Set the energy density of the mother to the maximum energy density
        if not hasattr(organism, 'E_density_mother'):
            setattr(organism, 'E_density_mother', organism.E_m)
        # Set initial reserve E_0
        setattr(organism, 'E_0', organism.E_density_mother * organism.V_0)

    def state_changes(self, t, state_vars):
        """
        Computes the derivatives of the state variables according to the standard DEB model equations. Function used in
        the integration solver.
        :param t: time
        :param state_vars: tuple of state variables (E, V, E_H, E_R)
        :return: derivatives of the state variables (dE, dV, dE_H, dE_R)
        """

        # Unpacking state variables (Reserve (E), Structure (E), Maturity (E_H), Reproduction Buffer (E_R))
        E, V, E_H, E_R = state_vars

        # Apply transform to Pet parameters
        self.transform(self.organism, t, state_vars)

        # Compute powers
        p_A, p_C, p_S, p_G, p_J, p_R, p_D = self.compute_powers(t, state_vars)

        # Pet is a foetus
        if E_H < self.organism.E_Hb:
            if t < self.organism.t_0:  # Gestation doesn't start until t=t_0
                dE, dV, dE_H, dE_R = 0, 0, 0, 0
            else:
                dE = self.organism.v * self.organism.E_density_mother * (V ** (2 / 3))
                dV = p_G / self.organism.E_G
                dE_H = p_R
                dE_R = 0
        else:
            dE = p_A - p_C
            dV = p_G / self.organism.E_G
            # Maturity or Reproduction Buffer logic
            if E_H < self.organism.E_Hp:
                dE_H = p_R
                dE_R = 0
            else:
                dE_H = 0
                dE_R = self.organism.kap_R * p_R

        return dE, dV, dE_H, dE_R

    def compute_powers(self, t, state_vars, compute_dissipation_power=False):
        """
        Computes all powers, with the option of also calculating the dissipation power p_D
        :param t: time
        :param E: reserve
        :param V: structure
        :param E_H: maturity
        :param E_R: reproduction buffer
        :param compute_dissipation_power: whether to compute the dissipation power
        :return: tuple of powers (p_A, p_C, p_S, p_G, p_J, p_R, p_D)
        """
        # Unpacking state variables (Reserve (E), Structure (E), Maturity (E_H), Reproduction Buffer (E_R))
        E, V, E_H, E_R = state_vars

        # Computing powers
        p_A = self.p_A(V, E_H, t)  # Assimilation power
        p_C = self.p_C(E, V)  # Mobilization power
        p_S = self.p_S(V)  # Somatic maintenance power
        p_G = self.p_G(p_C, p_S, V, E_H)  # Growth power
        p_J = self.p_J(E_H)  # Maturity maintenance power
        p_R = self.p_R(p_C, p_J, p_S, p_G, E_H)  # Reproduction power

        # Dissipation power
        if compute_dissipation_power:
            p_D = self.p_D(p_S, p_J, p_R, E_H)
        else:
            p_D = 0

        return p_A, p_C, p_S, p_G, p_J, p_R, p_D

    def p_A(self, V, E_H, t):
        """
        Computes the assimilation power p_A.

        :param V: Scalar or array of Strucure values
        :param E_H: Scalar or array of Maturity values
        :param t: Scalar or array of Time values
        :return: Scalar or array of assimilation power p_A values
        """
        if type(E_H) == np.ndarray:
            p_A = np.zeros_like(E_H)
            for i, (structure, maturity, time) in enumerate(zip(V, E_H, t)):
                if maturity < self.organism.E_Hb:  # Pet is a foetus
                    p_A[i] = 0
                elif maturity < self.organism.E_Hx:  # Baby stage
                    p_A[i] = self.organism.p_Am * self.organism.f_milk * (structure ** (2 / 3))
                else:  # Adult
                    p_A[i] = self.organism.p_Am * self.food_function(time) * (structure ** (2 / 3))
            return p_A
        else:
            if E_H < self.organism.E_Hb:  # Pet is a foetus
                return 0
            elif E_H < self.organism.E_Hx:  # Baby stage
                return self.organism.p_Am * self.organism.f_milk * (V ** (2 / 3))
            else:  # Adult
                return self.organism.p_Am * self.food_function(t) * (V ** (2 / 3))

    def p_G(self, p_C, p_S, V, E_H):
        """
        Computes the growth power p_G.

        :param p_C: Scalar or array of mobilization power values
        :param p_S: Scalar or array of somatic maintenance power values
        :param V: Scalar or array of Structure values
        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of growth power p_G values
        """
        if type(E_H) == np.ndarray:
            p_G = np.zeros_like(E_H)
            for i, (maturity, mobil, soma_maint, structure) in enumerate(zip(E_H, p_C, p_S, V)):
                if maturity < self.organism.E_Hb:  # Pet is a foetus
                    p_G[i] = self.organism.E_G * self.organism.v * (structure ** (2 / 3))
                else:
                    p_G[i] = self.organism.kappa * mobil - soma_maint
            return p_G
        else:
            if E_H < self.organism.E_Hb:  # Pet is a foetus
                return self.organism.E_G * self.organism.v * (V ** (2 / 3))
            else:
                return self.organism.kappa * p_C - p_S

    def p_R(self, p_C, p_J, p_S, p_G, E_H):
        """
        Computes the reproduction power p_R

        :param p_C: Scalar or array of mobilization power values
        :param p_J: Scalar or array of maturity maintenance power values
        :param p_S: Scalar or array of somatic maintenance values
        :param p_G: Scalar or array of growth power values
        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of reproduction power p_R values
        """
        if type(E_H) == np.ndarray:
            p_R = np.zeros_like(E_H)
            for i, (maturity, mobil, mat_maint, soma_maint, growth) in enumerate(zip(E_H, p_C, p_J, p_S, p_G)):
                if maturity < self.organism.E_Hb:  # Pet is a foetus
                    p_R[i] = (1 - self.organism.kappa) * (soma_maint + growth) / self.organism.kappa - mat_maint
                else:
                    p_R[i] = (1 - self.organism.kappa) * mobil - mat_maint
            return p_R
        else:
            if E_H < self.organism.E_Hb:  # Pet is a foetus
                return (1 - self.organism.kappa) * (p_S + p_G) / self.organism.kappa - p_J
            else:
                return (1 - self.organism.kappa) * p_C - p_J


class RUM(STX):
    def __init__(self, organism):
        """Takes as input a Ruminant class or a dictionary of parameters to create a Pet class."""

        # Create the Pet class from the dictionary of parameters
        if isinstance(organism, dict):
            organism = Ruminant(**organism)
        # Check that organism is a Pet class
        elif not isinstance(organism, Ruminant):
            raise Exception("Input must be of class Ruminant or a dictionary of parameters to create a Ruminant class.")

        self.filter_pet(organism)

        self.organism = organism
        self.ode_sol = None  # Output from ODE solver
        self.sol = None  # Full solution including powers, fluxes and entropy
        self.food_function = None  # Function of scaled functional feeding response (f) over time

    def mineral_fluxes(self, p_A, p_D, p_G, E_H):
        """
        Computes the mineral fluxes using the basic organic powers. Until weaning, the organism does not ruminate and
        therefore the standard assimilation equation applies. Afterwards, both sub transformations occur. The mineral
        fluxes are in following format (CO2, H2O, O2, N-Waste, CH4).
        :param p_A: scalar of assimilation power
        :param p_D: scalar of dissipation power
        :param p_G: scalar of growth power
        :param E_H: scalar of maturity
        :return: (5,1) np.ndarray of mineral fluxes
        """
        if type(p_A) != np.ndarray:
            p_A = np.array([p_A])
            p_D = np.array([p_D])
            p_G = np.array([p_G])
        powers = np.array([p_A, p_D, p_G])

        if E_H < self.organism.E_Hx:  # Use standard assimilation reaction
            mineral_fluxes = self.organism.eta_M_CO2 @ powers
            mineral_fluxes = np.pad(mineral_fluxes, ((0, 1), (0, 0)))
        else:  # Consider production of methane during rumination
            eta_M = self.organism.eta_M
            mineral_fluxes = eta_M @ powers
        return mineral_fluxes

    def entropy_generation(self, p_A, p_D, p_G, E_H):
        """
        Computes the entropy generated using the basic organic powers. Until weaning, the organism does not ruminate and
        therefore the standard assimilation equation applies. Afterwards, both sub transformations occur.
        :param p_A: scalar of assimilation power
        :param p_D: scalar of dissipation power
        :param p_G: scalar of growth power
        :param E_H: scalar of maturity
        :return: scalar of entropy
        """
        if type(p_A) != np.ndarray:
            p_A = np.array([p_A])
            p_D = np.array([p_D])
            p_G = np.array([p_G])
        powers = np.array([p_A, p_D, p_G])
        if E_H < self.organism.E_Hx:  # Use standard assimilation reaction
            return -(self.organism.comp.h_M[:-1] @ self.organism.gamma_M_CO2 + self.organism.comp.h_O @
                     self.organism.gamma_O) @ powers / self.organism.T / self.organism.comp.E.mu
        else:  # Consider production of methane during rumination
            return -(self.organism.comp.h_M @ self.organism.gamma_M + self.organism.comp.h_O @ self.organism.gamma_O) \
                   @ powers / self.organism.T / self.organism.comp.E.mu
