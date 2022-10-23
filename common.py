#!/usr/bin/python
# -*- coding:utf8 -*-
import time

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


class SampleDataWrapper:
    def __init__(self, observe_t, observe_y, observe_u, predict_t, predict_y, predict_u):
        self.observe_y = observe_y
        self.observe_u = observe_u
        self.predict_y = predict_y
        self.predict_u = predict_u
        self.observe_t = observe_t
        self.predict_t = predict_t

