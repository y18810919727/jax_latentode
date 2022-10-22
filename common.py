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
