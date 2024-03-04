from .pet import Pet, Ruminant
import numpy as np


def length_curve(pet: Pet, t, L_i, t0=0.0, f=1.0):
    r_B = pet.r_B(f=f)
    L_inf = pet.L_inf(f=f)
    return L_inf - (L_inf - L_i) * np.exp(-(t - t0) * r_B)


def weight_curve(pet: Pet, t, W_i, t0=0.0, f=1.0):
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    L = length_curve(pet, t, L_i, t0=t0, f=f)
    return W_conv_factor * np.power(L, 3)


# Powers
def assimilation_power(pet: Pet, t, L_i, t0=0.0, f=1.0):
    L = length_curve(pet, t, L_i, t0=t0, f=f)
    return f * pet.p_Am * np.power(L, 2)


def somatic_maintenance_power(pet: Pet, t, L_i, t0=0.0, f=1.0):
    L = length_curve(pet, t, L_i, t0=t0, f=f)
    return pet.p_M * np.power(L, 3) + pet.p_T * np.power(L, 2)


def growth_power(pet: Pet, t, L_i, t0=0.0, f=1.0):
    p_A = assimilation_power(pet, t, L_i, t0=t0, f=f)
    p_S = somatic_maintenance_power(pet, t, L_i, t0=t0, f=f)
    r_B = pet.r_B(f=f)
    return 3 * r_B / pet.k_M * (pet.kap * p_A - p_S)


def dissipation_power(pet: Pet, t, L_i, t0=0.0, f=1.0):
    p_A = assimilation_power(pet, t, L_i, t0=t0, f=f)
    p_S = somatic_maintenance_power(pet, t, L_i, t0=t0, f=f)
    r_B = pet.r_B(f=f)
    return 3 * r_B / pet.k_M * ((1 - pet.kap) * p_A + (1 + f / pet.kap / pet.g) * p_S)


def mobilization_power(pet: Pet, t, L_i, t0=0.0, f=1.0):
    p_A = assimilation_power(pet, t, L_i, t0=t0, f=f)
    p_S = somatic_maintenance_power(pet, t, L_i, t0=t0, f=f)
    return (pet.E_G * p_A + f * pet.E_m * p_S) / (pet.kap * f * pet.E_m + pet.E_G)


def feed_intake_curve(pet: Pet, t, W_i, t0=0.0, f=1.0):
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    return pet.comp.X.w / pet.comp.X.mu / pet.kap_X * p_A


def cumulative_feed_intake_curve(pet: Pet, t, W_i, t0=0.0, f=1.0):
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    L_inf = pet.L_inf(f=f)
    L = length_curve(pet, t, L_i, t0=t0, f=f)
    coef = f * pet.comp.X.w / pet.comp.X.mu / pet.kap_X * pet.p_Am
    return coef * ((L_inf ** 2) * (t - t0) - 0.5 / pet.r_B(f=f) * ((L_inf + L) ** 2 - (L_inf + L_i) ** 2))


def dynamic_feed_conversion_ratio(pet: Pet, t, W_i, t0=0.0, f=1.0):
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)

    L = length_curve(pet, t, L_i, t0=t0, f=f)
    L_inf = pet.L_inf(f=f)
    r_B = pet.r_B(f=f)

    feed_coef = f * pet.comp.X.w / pet.comp.X.mu / pet.kap_X * pet.p_Am
    weight_coef = 1 / 3 / r_B / W_conv_factor

    return feed_coef * weight_coef / (L_inf - L)


def faeces_production_curve(pet: Pet, t, W_i, t0=0.0, f=1.0):
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    return pet.comp.P.w * pet.kap_P / pet.comp.P.mu / pet.kap_X * p_A


def methane_emissions_curve(pet: Ruminant, t, W_i, t0=0.0, f=1.0):
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    return pet.comp.M.w * pet.eta_M[4, 0] * p_A


def co2_emissions_curve(pet: Pet, t, W_i, t0=0.0, f=1.0, assimilation=True, growth=True, dissipation=True):
    select_fluxes = np.diag(np.array([assimilation, growth, dissipation], dtype=float))
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    p_G = growth_power(pet, t, L_i, t0=t0, f=f)
    p_D = dissipation_power(pet, t, L_i, t0=t0, f=f)
    p = np.array([p_A, p_G, p_D])
    return pet.comp.C.w * pet.eta_M[0, :] @ select_fluxes @ p


def oxygen_consumption_curve(pet: Pet, t, W_i, t0=0.0, f=1.0, assimilation=True, growth=True, dissipation=True):
    select_fluxes = np.diag(np.array([assimilation, growth, dissipation], dtype=float))
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    p_G = growth_power(pet, t, L_i, t0=t0, f=f)
    p_D = dissipation_power(pet, t, L_i, t0=t0, f=f)
    p = np.array([p_A, p_G, p_D])
    return pet.comp.O.w * pet.eta_M[2, :] @ select_fluxes @ p


def water_production_curve(pet: Pet, t, W_i, t0=0.0, f=1.0, assimilation=True, growth=True, dissipation=True):
    select_fluxes = np.diag(np.array([assimilation, growth, dissipation], dtype=float))
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    p_G = growth_power(pet, t, L_i, t0=t0, f=f)
    p_D = dissipation_power(pet, t, L_i, t0=t0, f=f)
    p = np.array([p_A, p_G, p_D])
    return pet.comp.H.w * pet.eta_M[1, :] @ select_fluxes @ p


def n_waste_production_curve(pet: Pet, t, W_i, t0=0.0, f=1.0, assimilation=True, growth=True, dissipation=True):
    select_fluxes = np.diag(np.array([assimilation, growth, dissipation], dtype=float))
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    p_G = growth_power(pet, t, L_i, t0=t0, f=f)
    p_D = dissipation_power(pet, t, L_i, t0=t0, f=f)
    p = np.array([p_A, p_G, p_D])
    return pet.comp.N.w * pet.eta_M[3, :] @ select_fluxes @ p


def heat_generation(pet: Pet, t, W_i, t0=0.0, f=1.0, assimilation=True, growth=True, dissipation=True):
    select_fluxes = np.diag(np.array([assimilation, growth, dissipation], dtype=float))
    W_conv_factor = 1 + f * pet.omega
    L_i = np.power(W_i / W_conv_factor, 1 / 3)
    p_A = assimilation_power(pet, t, L_i=L_i, t0=t0, f=f)
    p_G = growth_power(pet, t, L_i, t0=t0, f=f)
    p_D = dissipation_power(pet, t, L_i, t0=t0, f=f)
    p = np.array([p_A, p_G, p_D])
    return -(pet.comp.mu_O @ pet.eta_O + pet.comp.mu_M @ pet.eta_M) @ select_fluxes @ p
