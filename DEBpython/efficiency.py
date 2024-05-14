import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .pet import Pet, Ruminant
from .formulae import weight_curve, cumulative_feed_intake_curve


def average_daily_gain(weight_init, weight_final, duration):
    return (weight_final - weight_init) / duration


def feed_conversion_ratio(mode='total', weight_init=None, weight_final=None, total_feed_intake=None, dfi=None,
                          adg=None):
    if mode == 'total':
        return total_feed_intake / (weight_final - weight_init)
    elif mode == 'daily':
        return dfi / adg


def feed_conversion_ratio_deb(pet: Pet, weight_init, duration, weight_final=None, units='kg'):
    if units == 'kg':
        weight_init *= 1000
    if weight_final is None:
        weight_final = weight_curve(pet, t=duration, W_i=weight_init)
    tfi = cumulative_feed_intake_curve(pet, t=duration, W_i=weight_init)
    return feed_conversion_ratio(weight_init=weight_init, weight_final=weight_final, total_feed_intake=tfi)


def compute_residual_feed_intake(data, weight_init, weight_final, total_feed_intake, duration):
    if isinstance(weight_init, str):
        weight_init = data[weight_init]
    if isinstance(weight_final, str):
        weight_final = data[weight_final]
    if isinstance(total_feed_intake, str):
        total_feed_intake = data[total_feed_intake]
    if isinstance(data, pd.DataFrame):
        rfi = pd.DataFrame(index=data.index)
    else:
        rfi = pd.DataFrame()

    rfi['metabolic_weight'] = np.power((weight_init + weight_final) / 2, 0.75)
    rfi['adg'] = (weight_final - weight_init) / duration
    rfi['dfi'] = total_feed_intake / duration

    edfi_reg = LinearRegression().fit(rfi[['adg', 'metabolic_weight']], rfi['dfi'])
    rfi['edfi'] = edfi_reg.predict(rfi[['adg', 'metabolic_weight']])
    rfi['rfi'] = rfi['dfi'] - rfi['edfi']

    return rfi


def methane_yield(mode='total', total_methane_emissions=None, total_feed_intake=None, dme=None, dfi=None):
    """
    Computes the methane yield in g CH4 /kg of dry matter intake.
    @param mode: If 'total', computes the methane yield using the totals for the whole period. If 'daily', computes the
    methane yield using average values for the duration.
    @param total_methane_emissions: Total methane emissions for the duration, in g CH4
    @param total_feed_intake: Total dry matter intake for the duration, in kg DM
    @param dme: Average daily methane emissions, in g CH4
    @param dfi: Average daily dry matter intake, in kg DM
    @return: MY, the methane yield in g CH4 /kg DM
    """
    if mode == 'total':
        return total_methane_emissions / total_feed_intake
    elif mode == 'daily':
        return dme / dfi


def methane_yield_deb(rum: Ruminant):
    """
    Computes the methane yield in g CH4 /kg of dry matter intake.
    @param rum: a Ruminant object
    @return: MY, the methane yield in g CH4 /kg DM
    """
    return rum.comp.M.w * rum.eta_M[4, 0] * rum.comp.X.mu / rum.comp.X.w / rum.kap_X / 1000


def methane_intensity(mode='total', weight_init=None, weight_final=None, total_methane_emissions=None,
                      dme=None, adg=None):
    """
    Computes the methane intensity in g CH4 /kg of weight gain.
    @param mode: If 'total', computes the methane intensity using the totals for the whole period. If 'daily', computes
    the methane intensity using average values for the duration.
    @param weight_init: Initial weight, in kg
    @param weight_final: Final weight, in kg
    @param total_methane_emissions: Total methane emissions for the duration, in g CH4
    @param dme: Average daily methane emissions, in g CH4
    @param adg: Average daily weight gain, in kg
    @return: MI, the methane intensity in g CH4 /kg weight gain
    """
    if mode == 'total':
        return total_methane_emissions / (weight_final - weight_init)
    elif mode == 'daily':
        return dme / adg

