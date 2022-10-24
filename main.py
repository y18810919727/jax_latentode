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
from aim import Run

def main(
    dataset_size=10000,
    batch_size=512,    # 256,
    lr=1e-2,
    steps=250,
    test_steps=1000,
    save_every=10,
    hidden_size=16,
    latent_size=16,
    width_size=16,
    depth=2,
    seed=5678,
    epoch_num=800,
    max_epochs_stop=50,
    min_epochs=200
):
    # aim run
    run = Run(
        experiment='latent_ode',
        repo='latent_ode_tmp/.aim',
    )

    args = {
        'dataset_size': dataset_size,
        'batch_size': batch_size,
        'lr': lr,
        'steps': steps,
        'test_steps': test_steps,
        'save_every': save_every,
        'hidden_size': hidden_size,
        'latent_size': latent_size,
        'width_size': width_size,
        'depth': depth,
        'seed': seed,
        'epoch_num': epoch_num,
        'max_epochs_stop': max_epochs_stop,
        'min_epochs': min_epochs,
    }

    run["hparams"] = args

    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jrandom.split(key, 5)

    train_all, test_all = select_dataset(dataset_name='cstr', ct_time=True, sp=0.5)

    ts_train, us_train, ys_train = [torch.cat([train_all[i], train_all[i+1]], dim=1) for i in range(0, 6, 2)]

    ts_history_test, ts_forward_test, external_input_history_test, external_input_forward_test, \
    observation_history_test, observation_forward_test = test_all
    # # 拼接全length的jnp array
    # ts = torch.cat([torch.tensor(ts_history.tolist()),
    #                 torch.tensor(ts_forward.tolist())], dim=1)
    # ts = jnp.array(ts.numpy().tolist())
    # us = torch.cat([torch.tensor(external_input_history.tolist()),
    #                 torch.tensor(external_input_forward.tolist())], dim=1)
    # us = jnp.array(us.numpy().tolist())
    # ys = torch.cat([torch.tensor(observation_history.tolist()),
    #                 torch.tensor(observation_forward.tolist())], dim=1)
    # ys = jnp.array(ys.numpy().tolist())

    # ts, ys, us = get_data(dataset_size, key=data_key)

    model = LatentODE(
        out_size = ys_train.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        us_num=us_train.shape[-1],
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

    @eqx.filter_jit
    def model_eval(model, ts_history_test_i, observation_history_test_i, external_input_history_test_i, ts_forward_test_i,
             observation_forward_test_i, external_input_forward_test_i, key_i):
        batch_size, _ = ts_history_test_i.shape
        key_i = jrandom.split(key_i, batch_size)
        pred_y = jax.vmap(model.sample)(ts_history_test_i, observation_history_test_i, external_input_history_test_i,
                 ts_forward_test_i, external_input_forward_test_i, key=key_i)
        # rmse_sum = jnp.sum(jnp.sqrt(jnp.mean((pred_y - observation_forward_test_i) ** 2, axis=1)))
        rmse_sum = RMSE_jnp(observation_forward_test_i, pred_y) * batch_size
        return pred_y, rmse_sum, batch_size

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Plot results
    num_plots = 1 + (steps - 1) // save_every
    if ((steps - 1) % save_every) != 0:
        num_plots += 1
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 8, 8))
    axs[0].set_ylabel("x")
    axs = iter(axs)
    iter_train_dataloader = dataloader((ts_train, ys_train, us_train), batch_size, key=loader_key)
    iter_test_dataloader = dataloader((ts_history_test, ts_forward_test, external_input_history_test,
                                                   external_input_forward_test, observation_history_test,
                                                   observation_forward_test), batch_size, key=loader_key)

    best_test = 1e12
    best_dev_epoch = -1

    for epoch in range(epoch_num):
        acc_loss = 0
        acc_step = 0
        for step, (ts_i, ys_i, us_i) in enumerate(iter_train_dataloader):
            if ts_i is None:
                break
            start = time.time()
            value, model, opt_state, train_key = make_step(
                model, opt_state, ts_i, ys_i, us_i, train_key
            )
            end = time.time()
            print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
            acc_step += 1
            acc_loss += float(value)

        run.track(float(acc_loss / acc_step), name='epoch_loss', epoch=epoch, context={"latent_ode": "train"})
        print('epoch = {} with train_loss = {:.4f}'.format(epoch, float(acc_loss / acc_step)))

        if (epoch % save_every) == 0 or epoch == epoch_num - 1:
            # ax = next(axs)
            # Sample over a longer time interval than we trained on. The model will be
            # sufficiently good that it will correctly extrapolate!

            sum_rmse = 0
            sum_bs = 0
            for (ts_history_test_i, ts_forward_test_i, external_input_history_test_i,
                            external_input_forward_test_i, observation_history_test_i,
                            observation_forward_test_i) in iter_test_dataloader:
                if ts_history_test_i is None:
                    break
                pred_y, rmse, batch_size = model_eval(model, ts_history_test_i, observation_history_test_i, external_input_history_test_i,
                           ts_forward_test_i, observation_forward_test_i, external_input_forward_test_i, sample_key)
                # acc_rmse.append(np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y)))
                sum_rmse = sum_rmse + rmse
                sum_bs = sum_bs + batch_size

            # print(f"RMSE: {np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y))}")
            print(f"RMSE: {sum_rmse/sum_bs}")
            epoch_test = sum_rmse / sum_bs
            if best_test > epoch_test:
                best_test = epoch_test
                best_dev_epoch = epoch
                # ckpt = dict()
                # ckpt['model'] = model
                # ckpt['epoch'] = epoch + 1
                print('best_test update at epoch = {}'.format(epoch))
            if epoch - best_dev_epoch > max_epochs_stop and epoch > min_epochs:
                print('Early stopping at epoch = {}'.format(epoch))


    # plt.savefig("latent_ode.png")
    # plt.show()


if __name__ == "__main__":
    main()
