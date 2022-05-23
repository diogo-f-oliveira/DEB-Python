import pet
import models
from visualizer import Plotter

if __name__ == '__main__':
    # Create Pet from parameters. Standard composition is assumed. Parameters t_0 and E_Hx are required in the STX model
    mammal = pet.Pet(
        p_Am=2501.3,  # Surface-specific maximum assimilation rate (J/d.cm^2)
        kappa=0.976264,  # Allocation fraction to soma (-)
        v=0.107224,  # Energy conductance (cm/d)
        p_M=42.2556,  # Volume-specific somatic maintenance rate (J/d.cm^3)
        E_G=8261.79,  # Specific cost for Structure (J/cm^3)
        k_J=0.0002,  # Maturity maintenance rate constant (d^-1)
        kap_R=0.95,  # Reproduction efficiency (-)
        E_Hb=2071229.972,  # Maturity at birth (J)
        E_Hp=30724119.81,  # Maturity at puberty (J)
        T=298.15,  # Temperature (K)
        t_0=109.4715964,  # Time until start of development (d)
        E_Hx=15139260.45,  # Maturity at weaning (J)
    )

    print(mammal)

    # Create STX model from Pet class
    model = models.STX(mammal)

    # Get the full state of the organism at maximum growth
    fully_grown_sol = model.fully_grown()
    print(f"Entropy generation at maximum growth: {fully_grown_sol.entropy}")

    # Simulate the organism from birth with constant food density equal to 1. The returned solution contains the full
    # state of the organism, including powers, fluxes and entropy at all simulated time steps
    sol = model.simulate(food_function=1, t_span=(0, 10000), step_size='auto', initial_state='birth')

    # Visualize the simulation
    viz = Plotter(sol)

    viz.plot_state_vars()
    viz.plot_powers()
    viz.plot_mineral_fluxes()
    viz.plot_real_variables()
    viz.plot_entropy_generation()
