import numpy as np

# !/usr/bin/python
# -*- coding:utf8 -*-

from common import *
from dataset import get_data
from func import Func
from latent_ode import LatentODE
from dataset import dataloader
from sklearn.metrics import mean_squared_error
import torch
from numpy import *

from dataset import select_dataset

def main(
    dataset_size=10000,
    batch_size=256,
    lr=1e-2,
    steps=250,
    test_steps=1000,
    save_every=50,
    hidden_size=16,
    latent_size=16,
    width_size=16,
    depth=2,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jrandom.split(key, 5)

    jnp_train, jnp_test = select_dataset(dataset_name='cstr', ct_time=True, sp=0.5)

    ts_history, ts_forward, external_input_history, external_input_forward, \
    observation_history, observation_forward = jnp_train

    ts_history_test, ts_forward_test, external_input_history_test, external_input_forward_test, \
    observation_history_test, observation_forward_test = jnp_test
    # 拼接全length的jnp array
    ts = torch.cat([torch.tensor(ts_history.tolist()),
                    torch.tensor(ts_forward.tolist())], dim=1)
    ts = jnp.array(ts.numpy().tolist())
    us = torch.cat([torch.tensor(external_input_history.tolist()),
                    torch.tensor(external_input_forward.tolist())], dim=1)
    us = jnp.array(us.numpy().tolist())
    ys = torch.cat([torch.tensor(observation_history.tolist()),
                    torch.tensor(observation_forward.tolist())], dim=1)
    ys = jnp.array(ys.numpy().tolist())

    # ts, ys, us = get_data(dataset_size, key=data_key)


    model = LatentODE(
        data_size=ys.shape[-1] + us.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        us_num=us.shape[-1],
    )

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, us_i, key_i):
        batch_size, _ = ts_i.shape
        key_i = jrandom.split(key_i, batch_size)
        loss = jax.vmap(model.train)(ts_i, ys_i, us_i, key=key_i)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, us_i, key_i):
        value, grads = loss(model, ts_i, ys_i, us_i, key_i)
        key_i = jrandom.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Plot results
    num_plots = 1 + (steps - 1) // save_every
    if ((steps - 1) % save_every) != 0:
        num_plots += 1
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 8, 8))
    axs[0].set_ylabel("x")
    axs = iter(axs)
    for step, (ts_i, ys_i, us_i) in zip(
            range(steps), dataloader((ts, ys, us), batch_size, key=loader_key)
    ):
        start = time.time()
        value, model, opt_state, train_key = make_step(
            model, opt_state, ts_i, ys_i, us_i, train_key
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

        if (step % save_every) == 0 or step == steps - 1:
            # ax = next(axs)
            # Sample over a longer time interval than we trained on. The model will be
            # sufficiently good that it will correctly extrapolate!

            acc_rmse = []
            for test_step, (ts_history_test_i, ts_forward_test_i, external_input_history_test_i,
                            external_input_forward_test_i, observation_history_test_i,
                            observation_forward_test_i) in zip(
                    range(test_steps), dataloader((ts_history_test, ts_forward_test, external_input_history_test,
                                                   external_input_forward_test, observation_history_test,
                                                   observation_forward_test), batch_size, key=loader_key)
            ):

                test_data = SampleDataWrapper(ts_history_test_i, ts_forward_test_i, external_input_history_test_i,
                            external_input_forward_test_i, observation_history_test_i,
                            observation_forward_test_i)

                sample_y = model.sample(test_data, key=sample_key)
                sample_t = np.asarray(test_data.predict_t)
                sample_y = np.asarray(sample_y)
                acc_rmse.append(np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y)))
            # print(f"RMSE: {np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y))}")
            print(f"RMSE: {mean(acc_rmse)}")
            # ax.plot(sample_t, sample_y[:, 0])
            # ax.plot(sample_t, sample_y[:, 1])
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_xlabel("t")

    # plt.savefig("latent_ode.png")
    # plt.show()


if __name__ == "__main__":
    main()
