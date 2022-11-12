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
# from aim import Run
import logging
import argparse


def main(
        dataset_size=10000,
        batch_size=512,  # 256,
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
        min_epochs=200,
        dataset_name="cstr"):

    logger.info(f"data: {args.data}")

    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jrandom.split(key, 5)

    train_all, test_all = select_dataset(dataset_name, ct_time=True, sp=args.sp, evenly=args.evenly)

    ts_train, us_train, ys_train = [torch.cat([train_all[i], train_all[i + 1]], dim=1) for i in range(0, 6, 2)]

    ts_history_test, ts_forward_test, external_input_history_test, external_input_forward_test, \
    observation_history_test, observation_forward_test = test_all

    model = LatentODE(
        out_size=ys_train.shape[-1],
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
    def model_eval(model, ts_history_test_i, observation_history_test_i, external_input_history_test_i,
                   ts_forward_test_i,
                   observation_forward_test_i, external_input_forward_test_i, key_i):
        batch_size, _ = ts_history_test_i.shape
        key_i = jrandom.split(key_i, batch_size)
        pred_y = jax.vmap(model.sample)(ts_history_test_i, observation_history_test_i, external_input_history_test_i,
                                        ts_forward_test_i, external_input_forward_test_i, key=key_i)
        # rmse_sum = jnp.sum(jnp.sqrt(jnp.mean((pred_y - observation_forward_test_i) ** 2, axis=1)))
        # TODO：这里需要换维度[batch_size, length, dim] -> [length, batch_size, dim]
        rmse_sum = RMSE_jnp(observation_forward_test_i,
                            pred_y[:, -observation_forward_test_i.shape[1]:, :]) * batch_size
        # TODO:加入rrse评价指标计算
        rrse_sum = RRSE_jnp(observation_forward_test_i,
                            pred_y[:, -observation_forward_test_i.shape[1]:, :]) * batch_size

        return pred_y, rmse_sum, rrse_sum, batch_size

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
            logger.info(f"epoch: {epoch}, step: {step}, loss: {value}, computation time: {end - start}")
            acc_step += 1
            acc_loss += float(value)

        logger.info('epoch = {} with train_loss = {:.4f}'.format(epoch, float(acc_loss / acc_step)))

        if (epoch % save_every) == 0 or epoch == epoch_num - 1:
            # ax = next(axs)
            # Sample over a longer time interval than we trained on. The model will be
            # sufficiently good that it will correctly extrapolate!

            sum_rmse = 0
            sum_rrse = 0
            sum_bs = 0
            for (ts_history_test_i, ts_forward_test_i, external_input_history_test_i,
                 external_input_forward_test_i, observation_history_test_i,
                 observation_forward_test_i) in iter_test_dataloader:
                if ts_history_test_i is None:
                    break
                pred_y, rmse, rrse, batch_size = model_eval(model, ts_history_test_i, observation_history_test_i,
                                                      external_input_history_test_i,
                                                      ts_forward_test_i, observation_forward_test_i,
                                                      external_input_forward_test_i, sample_key)
                # acc_rmse.append(np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y)))
                sum_rmse = sum_rmse + rmse
                sum_rrse = sum_rrse + rrse
                sum_bs = sum_bs + batch_size

            # print(f"RMSE: {np.sqrt(mean_squared_error(sample_y[:, [0, 1]], test_data.predict_y))}")
            logger.info(f"\n RMSE: {sum_rmse / sum_bs}")
            logger.info(f"\n RRSE: {sum_rrse / sum_bs}")
            epoch_test = sum_rmse / sum_bs
            if best_test > epoch_test:
                best_test = epoch_test
                best_dev_epoch = epoch
                # ckpt = dict()
                # ckpt['model'] = model
                # ckpt['epoch'] = epoch + 1
                logger.info('best_test update at epoch = {}'.format(epoch))
            if epoch - best_dev_epoch > max_epochs_stop and epoch > min_epochs:
                logger.info('Early stopping at epoch = {}'.format(epoch))
                break

    # plt.savefig("latent_ode.png")
    # plt.show()


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-dir', type=str, required=True)

    parser.add_argument('--data', type=str, default='cstr', choices=['cstr', 'winding', 'thickener'])
    parser.add_argument('--sp', type=float, default=0.5, help='sp rate.')
    parser.add_argument('--evenly', type=str2bool, default=False)
    parser.add_argument('--train-epochs', type=int, default=800, help='Number of epochs for training.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--max_epochs_stop', type=int, default=50, help='Number of max epochs for training stop')
    parser.add_argument('--min_epochs', type=int, default=200, help='Number of min epochs for training')

    parser.add_argument('--dt', type=float, default=1e-2)

    args = parser.parse_args()

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    time_now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    test_log = logging.FileHandler(f'logs/latent_ode_{args.data}_sp_{str(args.sp)}_evenly_{str(args.evenly)}_{str(time_now)}.log', 'w', encoding='utf-8')
    test_log.setLevel(logging.DEBUG)

    stdout_log = logging.StreamHandler()
    stdout_log.setLevel(logging.DEBUG)

    formator = logging.Formatter(fmt="%(asctime)s %(filename)s %(levelname)s %(message)s",
                                 datefmt="%Y/%m/%d %X")
    stdout_log.setFormatter(formator)
    test_log.setFormatter(formator)

    logger.addHandler(stdout_log)
    logger.addHandler(test_log)

    ckpt_dir = os.path.join(args.train_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    main(batch_size=args.batch_size, dataset_name=args.data)
