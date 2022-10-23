#!/usr/bin/python
# -*- coding:utf8 -*-
import time

from common import *
from func import Func

class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int

    def __init__(
            self, *, data_size, hidden_size, latent_size, width_size, depth, key, us_num, **kwargs
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jrandom.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size + us_num,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

    # Encoder of the VAE
    def _latent(self, ts, ys, us, key):
        data = jnp.concatenate([ts[:, None], ys, us], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size:]
        std = jnp.exp(logstd)
        latent = mean + jrandom.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, us, latent):
        dt0 = 0.4  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        linear_interp = diffrax.LinearInterpolation(ts=ts, ys=us)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
            args=(linear_interp,),
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        # double check: 只取前两维作为ys，求loss
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys[:, [0, 1]]) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean ** 2 + std ** 2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Run both encoder and decoder during training.
    def train(self, ts, ys, us, *, key):
        latent, mean, std = self._latent(ts, ys, us, key)
        pred_ys = self._sample(ts, us, latent)
        return self._loss(ys, pred_ys, mean, std)

    # Run just the decoder during inference.
    def sample(self, sdata: SampleDataWrapper, *, key):
        latent, mean, std = self._latent(sdata.observe_t, sdata.observe_y, sdata.observe_u, key)
        return self._sample(sdata.predict_t, sdata.predict_u, latent)