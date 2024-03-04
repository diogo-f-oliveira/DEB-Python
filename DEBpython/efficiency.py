import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .pet import Pet
from .formulae import weight_curve, cumulative_feed_intake_curve


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
