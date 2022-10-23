#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import os

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
from datetime import datetime, timezone, timedelta

def subsample_indexes(c, time_steps, percentage, evenly=False):
    bs, l, d = c.shape
    n_to_subsample = int(l * percentage)
    if evenly:
        subsampled_idx = np.arange(l)[::int((l-1)/(n_to_subsample-1))] if n_to_subsample !=0 else np.arange(1)[0:1]
    else:
        subsampled_idx = sorted(np.random.choice(np.arange(l), n_to_subsample, replace=False))
    new_c = c[:,subsampled_idx]
    new_time_steps = time_steps[subsampled_idx]
    return new_c, new_time_steps


def to_timestamp(dt_str, tz_str='UTC+7:00'):
    t, tz=dt_str, tz_str
    tz=tz.split(':')[0].split('+')
    # str转换成datetime
    t=datetime.strptime(t, '%Y-%m-%d %H:%M')
    # datetime添加时区
    utc_dt=t.replace(tzinfo=timezone(timedelta(hours=int(tz[1]))))
    # datetime转换成timestamp
    return utc_dt.timestamp()


class SampleDataWrapper:
    def __init__(self, observe_t, observe_y, observe_u, predict_t, predict_y, predict_u):
        self.observe_y = observe_y
        self.observe_u = observe_u
        self.predict_y = predict_y
        self.predict_u = predict_u
        self.observe_t = observe_t
        self.predict_t = predict_t


def rrse_metric(truth, prediction):
    """
    evaluation function
    RRSE error
    """
    length = min(truth.shape[0], prediction.shape[0])
    truth, prediction = truth[-length:], prediction[-length:]
    if len(truth.shape) == 1:
        indices = np.logical_and(~np.isnan(truth), ~np.isnan(prediction))
        truth, prediction = truth[indices], prediction[indices]
        se = np.sum((truth - prediction) ** 2, axis=0)  # summed squared error per channel
        rse = se / np.sum((truth - np.mean(truth, axis=0)) ** 2)  # relative squared error
        rrse = np.mean(np.sqrt(rse))  # square root, followed by mean over channels
        return 100 * rrse  # in percentage
    results = [rrse_metric(truth[:, i], prediction[:, i]) for i in range(truth.shape[-1])]
    return np.mean(results)
