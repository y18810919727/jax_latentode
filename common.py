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


