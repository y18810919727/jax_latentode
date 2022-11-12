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
import torchcde
import torchsde

import numpy as np
import torch
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


def onceexp(data,a):#y一次指数平滑法
    x=[]
    t=data[0]
    x.append(t)
    for i in range(len(data)-1):
        t=t+a*(data[i+1]-t)
        x.append(t)
    return np.stack(x)


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


def sdeint_fn_batched(model, y0, ts, sdeint_fn, method='euler', dt=1e-2, adaptive=True, rtol=1e-6, atol=1e-6,  names=None):

    if names is None:
        names = {'drift': 'f', 'diffusion': 'g'}

    if ts.dim() == 1:
        aug_ys = sdeint_fn(
            sde=model,
            y0=y0,
            ts=ts,
            method=method,
            dt=dt,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            names=names
        )
        return aug_ys

    batch_size, h_dim = y0.shape
    class BatchedSDE:
        def __init__(self, model, names, dT):
            self.model = model
            self.noise_type = model.noise_type
            self.ori_drift = getattr(model, names['drift'])
            self.ori_diffusion = getattr(model, names['diffusion'])
            self.dT = dT

            # copy attributes in model to self
            for attr in dir(model):
                if not attr.startswith('__'):
                    setattr(self, attr, getattr(model, attr))
            setattr(self, names['drift'], self.f_reparameterization)
            setattr(self, names['diffusion'], self.g_reparameterization)

        def f_reparameterization(self, t, y):
            return self.ori_drift(t, y) * self.dT

        def g_reparameterization(self, t, y):
            return self.ori_diffusion(t, y) * self.dT

    dT = ts[1:] - ts[:-1]
    ys = [y0]
    for dt in dT:
        batched_sde = BatchedSDE(model, names, dt)

        bm = torchsde.BrownianInterval(
            t0=0,
            t1=1,
            size=(batch_size, h_dim),
            device=y0.device,
            levy_area_approximation='space-time'
        )  # We need space-time Levy area to use the SRK solver
        aug_ys = sdeint_fn(
            sde=batched_sde,
            y0=y0,
            ts=torch.Tensor([0.0, 1.0]).to(y0.device),
            bm=bm,
            method=method,
            dt=dt,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            names=names
        )
        ys.append(aug_ys[-1])
    return torch.stack(ys, dim=0)


class Interpolation:
    def __init__(self, ts, x):
        ts = ts[:, 0]
        x = x.transpose(0, 1)
        # ts = ts.transpose(0, 1)
        # x = torch.cat([ts, x], dim=-1)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x, ts)
        # coeffs = torchcde.natural_cubic_spline_coeffs(x, t=ts)
        self.X = torchcde.CubicSpline(coeffs, t=ts)

    def __call__(self, t):
        return self.X.evaluate(t)


def RMSE_torch(y_gt, y_pred):
    if len(y_gt.shape) == 3:
        return torch.mean(
            torch.stack(
                [RMSE_torch(y_gt[:, i], y_pred[:, i]) for i in range(y_gt.shape[1])]
            )
        )
    elif len(y_gt.shape) == 2:
        se = torch.sum((y_gt - y_pred) ** 2, dim=0)
        rse = torch.sqrt(se / y_gt.shape[0])
        return torch.mean(rse)
    else:
        raise AttributeError


def RRSE_torch(y_gt, y_pred):
    assert y_gt.shape == y_pred.shape
    if len(y_gt.shape) == 3:
        return torch.mean(
            torch.stack(
                [RRSE_torch(y_gt[:, i], y_pred[:, i]) for i in range(y_gt.shape[1])]
            )
        )

    elif len(y_gt.shape) == 2:
        # each shape (n_seq, n_outputs)
        se = torch.sum((y_gt - y_pred) ** 2, dim=0)
        rse = se / (torch.sum(
            (y_gt - torch.mean(y_gt, dim=0)) ** 2, dim=0
        ) + 1e-6)
        return torch.mean(torch.sqrt(rse))
    else:
        raise AttributeError


def RMSE_jnp(y_gt, y_pred):
    if len(y_gt.shape) == 3:
        return jnp.mean(
            jnp.stack(
                [RMSE_jnp(y_gt[i, :], y_pred[i, :]) for i in range(y_gt.shape[0])]
            )
        )
    elif len(y_gt.shape) == 2:
        se = jnp.sum((y_gt - y_pred) ** 2, axis=0)
        rse = jnp.sqrt(se / y_gt.shape[0])
        return jnp.mean(rse)
    else:
        raise AttributeError


def RRSE_jnp(y_gt, y_pred):
    assert y_gt.shape == y_pred.shape
    if len(y_gt.shape) == 3:
        return jnp.mean(
            jnp.stack(
                [RRSE_jnp(y_gt[i, :], y_pred[i, :]) for i in range(y_gt.shape[0])]
            )
        )

    elif len(y_gt.shape) == 2:
        # each shape (n_seq, n_outputs)
        se = jnp.sum((y_gt - y_pred) ** 2, axis=0)
        rse = se / (jnp.sum(
            (y_gt - jnp.mean(y_gt, axis=0)) ** 2, axis=0
        ) + 1e-6)
        return jnp.mean(jnp.sqrt(rse))
    else:
        raise AttributeError


class TimeRecorder:
    def __init__(self):
        self.infos = {}

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] = time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}:{:.2f}s'.format(info, t) for info, t in self.infos.items()])

if __name__ == '__main__':
    # a = np.random.randn(3, 3)
    # b = np.random.randn(3, 3)
    # y_gt = jnp.array(a)
    # y_pred = jnp.array(b)
    #
    # y_gt_t = torch.from_numpy(a)
    # y_pred_t = torch.from_numpy(b)
    #
    # print(RMSE_jnp(y_gt, y_pred))
    # print(RMSE_torch(y_gt_t, y_pred_t))
    # X = torch.randn((3, 3, 3))
    # t = torch.linspace(0,1,3).reshape((3,1,1)).repeat((1,3,1))
    # print(X.shape, t.shape)
    # inter = Intepolation(X, t)
    # pos = torch.mean(t, dim=1)
    # print(inter(pos).shape)
    x = torch.rand((30, 16, 3))
    T = torch.linspace(0, 1, 30)
    inter = Interpolation(x, T)
    print(inter(0.5).shape)


