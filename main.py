import numpy as np

# !/usr/bin/python
# -*- coding:utf8 -*-

from common import *
from dataset import get_data
from func import Func
from latent_ode import LatentODE
from dataset import dataloader
from sklearn.metrics import mean_squared_error


from dataset import select_dataset

def main(
    dataset_size=10000,
    batch_size=256,
    lr=1e-2,
    steps=250,
    save_every=50,
    hidden_size=16,
    latent_size=16,
    width_size=16,
    depth=2,
    seed=5678,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jrandom.split(key, 5)


    # ts, ys = get_data(dataset_size, key=data_key)
    # ys_train, ys_val, ys_test = select_dataset(dataset_name='cstr', ct_time=True, sp=0.5)
    # print(len(ys_train))
    ts, ys, us = get_data(dataset_size, key=data_key)


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
            ax = next(axs)
            # Sample over a longer time interval than we trained on. The model will be
            # sufficiently good that it will correctly extrapolate!
            test_data = SampleDataWrapper(ts_i[0], ys_i[0], us_i[0], ts_i[0] + 2, ys_i[0], us_i[0])

            sample_y = model.sample(test_data, key=sample_key)
            sample_t = np.asarray(test_data.predict_t)
            sample_y = np.asarray(sample_y)
            print(f"RMSE: {np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y))}")

            ax.plot(sample_t, sample_y[:, 0])
            ax.plot(sample_t, sample_y[:, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("t")

    plt.savefig("latent_ode.png")
    plt.show()


if __name__ == "__main__":
    main()
