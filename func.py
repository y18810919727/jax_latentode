#!/usr/bin/python
# -*- coding:utf8 -*-


from common import *

class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([y, args[0].evaluate(t)], axis=0))