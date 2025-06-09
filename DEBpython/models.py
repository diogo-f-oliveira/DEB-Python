from .environment import Environment
from .pet import Pet, Ruminant
from .solution import TimeIntervalSol, TimeInstantSol

from scipy.integrate import solve_ivp
import numpy as np


# TODO: Base Model Class

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

    MAX_STEP_SIZE = 1  # Maximum step size during integration of state equations, in days

    def __init__(self, organism: Pet, env: Environment):
        """Takes as input a Pet class or a dictionary of parameters to create a Pet class."""

        # Create the Pet class from the dictionary of parameters
        if isinstance(organism, dict):
            organism = Pet(**organism)
        # Check that organism is a Pet class
        elif not isinstance(organism, Pet):
            raise Exception("Input must be of class Pet or a dictionary of parameters used to create a Pet class.")
        self.filter_pet(organism)
        self.organism = organism
        self.state = organism.state

        # Add env
        self.env = env

        self.ode_sol = None  # Output from ODE solver
        self.sol = None  # Full solution including powers, fluxes and entropy

    @staticmethod
    def filter_pet(organism):
        """Ensures that the parameters of the organism are valid."""
        # Check validity of parameters of Pet
        valid, reason = organism.check_validity()
        if not valid:
            raise Exception(f"Invalid Pet parameters. {reason}")

    def simulate(self, t_span, step_size='auto', initial_state='embryo'):
        # TODO: Add events argument for solve_ivp
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

        # Define the times at which the solver should store the computed solution.
        if step_size == 'auto':
            t_eval = None
        elif isinstance(step_size, (float, int)):
            t_eval = np.arange(*t_span, step_size)
        else:
            raise Exception(f"Invalid step size value: {step_size}. Please select 'auto' for automatic step size during"
                            f" integration or input a fixed step size.")

        # Get initial state
        if initial_state == 'embryo':
            initial_state = (self.organism.E_0, self.organism.V_0, 0, 0)
        elif initial_state == 'birth':
            initial_state = self.get_state_at_maturity(self.organism.E_Hb)
            initial_state[2] *= (1 + 1e-6)  # To ensure simulation starts at the next stage
        elif initial_state == 'puberty':
            initial_state = self.get_state_at_maturity(self.organism.E_Hp)
            initial_state[2] *= (1 + 1e-6)  # To ensure simulation starts at the next stage
        elif len(initial_state) != 4:
            raise Exception(f"Invalid input {initial_state} for initial state. The initial state must be a list or "
                            f"tuple of length 4 with format (E, V, E_H, E_R) or a specified keyword.")

        # Integrate the state equations
        self.ode_sol = solve_ivp(self.state_changes, t_span, initial_state, t_eval=t_eval, max_step=self.MAX_STEP_SIZE)
        self.sol = TimeIntervalSol(self, self.ode_sol)
        return self.sol

    # TODO: Update function to work with Environment and State paradigm
    def get_state_at_maturity(self, E_H):
        event = lambda t, states: states[2] - E_H
        event.terminal = True
        embryo_state = (self.organism.E_0, self.organism.V_0, 0, 0)
        self.food_function = lambda t: 1
        ode_sol = solve_ivp(self.state_changes, (0, 1e6), embryo_state, max_step=self.MAX_STEP_SIZE, events=event)
        if ode_sol.status != 1:
            raise Exception(f"Simulation couldn't reach maturity level {E_H}.")

        return ode_sol.y[:, -1]

    # TODO: Update function to work with Environment and State paradigm
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
        state_vars = (self.organism.E_m * (self.organism.L_inf(f)) ** 3,
                      (self.organism.L_inf(f)) ** 3,
                      self.organism.E_Hp,
                      E_R)
        return TimeInstantSol(self, 0, state_vars)

    def state_changes(self, t, state_vars):
        """
        Computes the derivatives of the state variables according to the standard DEB model equations. Function used in
        the integration solver.
        :param t: time
        :param state_vars: tuple of state variables (E, V, E_H, E_R)
        :return: derivatives of the state variables (dE, dV, dE_H, dE_R)
        """
        # Setting state variables in State class
        self.state.t = t
        self.state.set_state_vars(state_vars)

        # Updating the environment to time step t
        self.env.update()

        # Compute powers
        p_A, p_C, p_S, p_G, p_J, p_R = self.compute_powers()

        # Changes to state variables
        dE = p_A - p_C
        dV = p_G / self.organism.E_G

        # Maturity or Reproduction Buffer logic
        if self.state.E_H < self.organism.E_Hp:
            dE_H = p_R
            dE_R = 0
        else:
            dE_H = 0
            dE_R = self.organism.kap_R * p_R
        return dE, dV, dE_H, dE_R

    def compute_powers(self, compute_dissipation_power=False):
        """
        Computes all powers, with the option of also calculating the dissipation power p_D.
        :param compute_dissipation_power: If True, computes and returns the dissipation power p_D
        :return: tuple of powers (p_A, p_C, p_S, p_G, p_J, p_R) or (p_A, p_C, p_S, p_G, p_J, p_R, p_D) if
        compute_dissipation_power is True.
        """

        # Computing powers
        p_A = self.p_A()  # Assimilation power
        p_S = self.p_S()  # Somatic maintenance power
        p_C = self.p_C(p_S)  # Mobilization power
        p_G = self.p_G(p_C, p_S)  # Growth power
        p_J = self.p_J()  # Maturity maintenance power
        p_R = self.p_R(p_C, p_J)  # Reproduction power

        # Dissipation power
        if compute_dissipation_power:
            p_D = self.p_D(p_S, p_J, p_R)
            return p_A, p_C, p_S, p_G, p_J, p_R, p_D
        else:
            return p_A, p_C, p_S, p_G, p_J, p_R

    def p_A(self):
        """Computes the assimilation power"""
        if self.state.E_H < self.organism.E_Hb:
            return 0
        else:
            return self.state.p_X * self.organism.kap_X

    def p_C(self, p_S):
        """
        Computes the mobilization power p_C.

        :param p_S: Scalar of somatic maintenance power p_S value
        :return: Scalar of mobilization power p_C value
        """
        E, V = self.state.E, self.state.V
        return (E / V) * (self.organism.E_G * self.organism.v * (V ** (2 / 3)) + p_S) / \
            (self.organism.kap * (E / V) + self.organism.E_G)

    def p_S(self):
        """
        Computes the somatic maintenance power p_S.

        :return: Scalar of somatic maintenance power p_S values
        """

        V = self.state.V
        return self.organism.p_M * V + self.organism.p_T * (V ** (2 / 3))

    def p_G(self, p_C, p_S):
        """
        Computes the growth power p_G.

        :param p_C: Scalar or array of mobilization power values
        :param p_S: Scalar or array of somatic maintenance power values
        :return: Scalar or array of growth power p_G values
        """
        return self.organism.kap * p_C - p_S

    def p_J(self):
        if self.state.E_H < self.organism.E_Hp:
            return self.organism.k_J * self.state.E_H
        else:  # Adult life stage
            return self.organism.k_J * self.organism.E_Hp

    def p_R(self, p_C, p_J):
        """
        Computes the reproduction power p_R

        :param p_C: Scalar or array of mobilization power values
        :param p_J: Scalar or array of maturity maintenance power values
        :return: Scalar or array of reproduction power p_R values
        """
        return (1 - self.organism.kap) * p_C - p_J

    def p_D(self, p_S, p_J, p_R):
        """
        Computes the dissipation power p_D

        :param p_S: Scalar or array of somatic maintenance power values
        :param p_J: Scalar or array of maturity maintenance power values
        :param p_R: Scalar or array of reproduction power values
        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of dissipation power p_D values
        """
        if self.state.E_H < self.organism.E_Hp:
            return p_S + p_J + p_R
        else:
            return p_S + p_J + (1 - self.organism.kap_R) * p_R

    def organic_fluxes(self, p_A, p_D, p_G):
        """
        Computes the organic fluxes from the assimilation power p_A, dissipation power p_D and growth power p_G.
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
        return self.organism.eta_O @ powers

    def mineral_fluxes(self, p_A, p_D, p_G):
        """
        Computes the mineral fluxes from the assimilation power p_A, dissipation power p_D and growth power p_G.
        :param p_A: Scalar or array of assimilation power values
        :param p_D: Scalar or array of dissipation power values
        :param p_G: Scalar or array of growth power values
        :return: array of mineral fluxes values. Each row corresponds to the flux of CO2, H2O, O2 and N-Waste
            respectively in mol/d.
        """
        if type(p_A) != np.ndarray:
            p_A = np.array([p_A])
            p_D = np.array([p_D])
            p_G = np.array([p_G])
        powers = np.array([p_A, p_D, p_G])
        return self.organism.eta_M @ powers

    def entropy_generation(self, p_A, p_D, p_G):
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
                - self.organism.comp.h_O) @ self.organism.gamma_O @ powers / self.state.T / self.organism.comp.E.mu


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
            raise Exception('The organism is not compatible with model STX: parameters t_0 and E_Hx must be defined.')
        elif organism.t_0 <= 0:
            raise Exception("The time until start of development can't be negative.")
        elif organism.E_Hx <= organism.E_Hb or organism.E_Hx >= organism.E_Hp:
            raise Exception("The maturity at weaning E_Hx must be larger than the maturity at birth and smaller "
                            "than maturity at puberty.")

        # Set the energy density of the mother to the maximum energy density
        if not hasattr(organism, 'E_density_mother'):
            setattr(organism, 'E_density_mother', organism.E_m)
        # Set initial reserve E_0
        setattr(organism, 'E_0', organism.E_density_mother * organism.V_0)

    def simulate(self, t_span, food_function=1, step_size='auto', initial_state='embryo', transformation=None):
        if initial_state == 'weaning':
            initial_state = self.get_state_at_maturity(self.organism.E_Hx)
            initial_state[2] *= (1 + 1e-6)  # To ensure simulation starts at the next stage
        return super().simulate(t_span=t_span, step_size=step_size, initial_state=initial_state, )

    def state_changes(self, t, state_vars):
        """
        Computes the derivatives of the state variables according to the standard DEB model equations. Function used in
        the integration solver.
        :param t: time
        :param state_vars: tuple of state variables (E, V, E_H, E_R)
        :return: derivatives of the state variables (dE, dV, dE_H, dE_R)
        """

        # Setting state variables in State class
        self.state.t = t
        self.state.set_state_vars(state_vars)

        # Updating the environment to time step t
        self.env.update()

        # Compute powers
        p_A, p_C, p_S, p_G, p_J, p_R = self.compute_powers()

        # Pet is a foetus
        if self.state.E_H < self.organism.E_Hb:
            if t < self.organism.t_0:  # Gestation doesn't start until t=t_0
                dE, dV, dE_H, dE_R = 0, 0, 0, 0
            else:
                dE = self.organism.v * self.organism.E_density_mother * (self.state.V ** (2 / 3))
                dV = p_G / self.organism.E_G
                dE_H = p_R
                dE_R = 0
        else:
            dE = p_A - p_C
            dV = p_G / self.organism.E_G
            # Maturity or Reproduction Buffer logic
            if self.state.E_H < self.organism.E_Hp:
                dE_H = p_R
                dE_R = 0
            else:
                dE_H = 0
                dE_R = self.organism.kap_R * p_R

        return dE, dV, dE_H, dE_R

    def compute_powers(self, compute_dissipation_power=False):
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

        # Computing powers
        p_A = self.p_A()  # Assimilation power
        p_S = self.p_S()  # Somatic maintenance power
        p_C = self.p_C(p_S)  # Mobilization power
        p_G = self.p_G(p_C, p_S)  # Growth power
        p_J = self.p_J()  # Maturity maintenance power
        p_R = self.p_R(p_C, p_J, p_S, p_G)  # Reproduction power

        # Dissipation power
        if compute_dissipation_power:
            p_D = self.p_D(p_S, p_J, p_R)
            return p_A, p_C, p_S, p_G, p_J, p_R, p_D
        else:
            return p_A, p_C, p_S, p_G, p_J, p_R

    def p_A(self):
        """
        Computes the assimilation power p_A.

        :param V: Scalar or array of Strucure values
        :param E_H: Scalar or array of Maturity values
        :param t: Scalar or array of Time values
        :return: Scalar or array of assimilation power p_A values
        """
        if self.state.E_H < self.organism.E_Hb:  # Pet is a foetus
            return 0
        elif self.state.E_H < self.organism.E_Hx:  # Baby stage
            return self.organism.p_Am * self.organism.f_milk * (self.state.V ** (2 / 3))
        else:  # Adult
            return self.state.p_X * self.organism.kap_X

    def p_G(self, p_C, p_S):
        """
        Computes the growth power p_G.

        :param p_C: Scalar or array of mobilization power values
        :param p_S: Scalar or array of somatic maintenance power values
        :param V: Scalar or array of Structure values
        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of growth power p_G values
        """

        if self.state.E_H < self.organism.E_Hb:  # Pet is a foetus
            return self.organism.E_G * self.organism.v * (self.state.V ** (2 / 3))
        else:
            return self.organism.kap * p_C - p_S

    def p_R(self, p_C, p_J, p_S, p_G):
        """
        Computes the reproduction power p_R

        :param p_C: Scalar or array of mobilization power values
        :param p_J: Scalar or array of maturity maintenance power values
        :param p_S: Scalar or array of somatic maintenance values
        :param p_G: Scalar or array of growth power values
        :param E_H: Scalar or array of Maturity values
        :return: Scalar or array of reproduction power p_R values
        """

        if self.state.E_H < self.organism.E_Hb:  # Pet is a foetus
            return (1 - self.organism.kap) * (p_S + p_G) / self.organism.kap - p_J
        else:
            return (1 - self.organism.kap) * p_C - p_J


class RUM(STX):
    def __init__(self, organism: Ruminant, env: Environment):
        """Takes as input a Ruminant class or a dictionary of parameters to create a Pet class."""

        # Create the Pet class from the dictionary of parameters
        if isinstance(organism, dict):
            organism = Ruminant(**organism)
        # Check that organism is a Pet class
        elif not isinstance(organism, Ruminant):
            raise Exception("Input must be of class Ruminant or a dictionary of parameters to create a Ruminant class.")
        self.filter_pet(organism)
        self.organism = organism
        self.state = organism.state

        # Add env
        self.env = env

        self.ode_sol = None  # Output from ODE solver
        self.sol = None  # Full solution including powers, fluxes and entropy

    def mineral_fluxes(self, p_A, p_D, p_G):
        """
        Computes the mineral fluxes using the basic organic powers. Until weaning, the organism does not ruminate and
        therefore the standard assimilation equation applies. Afterwards, both sub transformations occur. The mineral
        fluxes are in following format (CO2, H2O, O2, N-Waste, CH4). Units are mol/d.
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

        if self.state.E_H < self.organism.E_Hx:  # Use standard assimilation reaction
            mineral_fluxes = self.organism.eta_M_CO2 @ powers
            mineral_fluxes = np.pad(mineral_fluxes, ((0, 1), (0, 0)))
        else:  # Consider production of methane during rumination
            eta_M = self.organism.eta_M
            mineral_fluxes = eta_M @ powers
        return mineral_fluxes

    def entropy_generation(self, p_A, p_D, p_G):
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
        if self.state.E_H < self.organism.E_Hx:  # Use standard assimilation reaction
            return -(self.organism.comp.h_M[:-1] @ self.organism.gamma_M_CO2 + self.organism.comp.h_O @
                     self.organism.gamma_O) @ powers / self.organism.T / self.organism.comp.E.mu
        else:  # Consider production of methane during rumination
            return -(self.organism.comp.h_M @ self.organism.gamma_M + self.organism.comp.h_O @ self.organism.gamma_O) \
                @ powers / self.organism.T / self.organism.comp.E.mu


class ABJ(STD):
    def __init__(self, organism: Pet, env: Environment):
        super(ABJ, self).__init__(organism, env)
        self.L_b = None
        self.L_j = None

    def filter_pet(self, organism):
        super(ABJ, self).filter_pet(organism)

        # Check that the Pet class has parameters E_Hj defined
        if not hasattr(organism, 'E_Hx'):
            raise Exception('The organism is not compatible with model ABJ: parameter and E_Hj must be defined.')
        elif organism.E_Hj <= organism.E_Hb or organism.E_Hj >= organism.E_Hp:
            raise Exception("The maturity at metamorphosis E_Hj must be larger than the maturity at birth and smaller "
                            "than maturity at puberty.")

    def p_A(self):
        if self.state.E_H < self.organism.E_Hb:
            return 0
        elif self.state.E_H < self.organism.E_Hj:
            return self.state.p_X * self.organism.kap_X * (self.state.V ** (2 / 3))
        else:
            return self.state.p_X * self.organism.kap_X

    def p_C(self, p_S):
        E, V = self.state.E, self.state.V
        return (E / V) * (self.organism.E_G * self.organism.v * (V ** (2 / 3)) + p_S) / \
            (self.organism.kap + self.organism.E_G * V / E)

    @property
    def s_M(self):
        return
