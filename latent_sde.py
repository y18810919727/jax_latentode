#!/usr/bin/python
# -*- coding:utf8 -*-

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latent SDE fit to a single time series with uncertainty quantification."""
import argparse
from common import RMSE_torch, sdeint_fn_batched, TimeRecorder, RRSE_torch
from dataset import select_dataset

import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union
from interpolation import KernelInterpolation

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim

import torchsde

import time

# w/ underscore -> numpy; w/o underscore -> torch.
Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val

def trans(x, device):
    return x.to(device).transpose(0, 1)

def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentSDE(torchsde.SDEIto):

    def __init__(self, h_size, y_size, u_size, theta=1.0, mu=0.0, sigma=0.5, ):
        super(LatentSDE, self).__init__(noise_type="diagonal")

        self.y_size, self.u_size, self. h_size = y_size, u_size, h_size

        # Prior drift.
        # self.register_buffer("theta", torch.tensor([[theta]]))
        # self.register_buffer("mu", torch.tensor([[mu]]))
        # self.register_buffer("sigma", torch.tensor([[sigma]]))
        adaptive_theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True)
        self.register_buffer("theta", adaptive_theta)

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(h_size + u_size, 2 * h_size),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * h_size, h_size)
        )
        # self.register_buffer("sigma", torch.tensor([[sigma]]))
        self.sigma = torch.nn.Sequential(
            torch.nn.Linear(h_size + u_size, 2 * h_size),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * h_size, h_size),
            nn.Softplus()
        )

        # Posterior drift.
        # self.register_buffer("logvar", torch.tensor([[logvar]]))
        # self.logvar = torch.nn.Sequential(
        #     torch.nn.Linear(1, 1),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(1, 1)
        # )
        # self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_mean", nn.Parameter(torch.zeros((1, h_size)), requires_grad=True))
        self.register_buffer("py0_logvar", nn.Parameter(torch.zeros((1, h_size)), requires_grad=True))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(h_size+y_size+u_size, 2*h_size),
            nn.Tanh(),
            nn.Linear(2*h_size, h_size)
        )
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

        self.h2y = nn.Sequential(
            nn.Linear(h_size, 2*h_size),
            nn.Tanh(),
            nn.Linear(2*h_size, 2*y_size)
        )

    def update_u(self, ts, u):
        if args.inter == 'gp':
            from interpolation import KernelInterpolation as Interpolation
        else:
            from common import Interpolation
        self.u_inter = Interpolation(ts, u)

    def update_y(self, ts, y):
        if args.inter == 'gp':
            from interpolation import KernelInterpolation as Interpolation
        else:
            from common import Interpolation
        self.y_inter = Interpolation(ts, y)

    def f(self, t, h):  # Approximate posterior drift.
        # if t.dim() == 0:
        #     t = torch.full_like(y, fill_value=t)

        return self.net(torch.cat([h, self.u_inter(t), self.y_inter(t)], dim=-1))
        # Positional encoding in transformers for time-inhomogeneous posterior.
        # return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))

    def g(self, t, h):  # Shared diffusion.
        return self.sigma(torch.cat([h, self.u_inter(t)], dim=-1))
        # return self.sigma.repeat(y.size(0), 1)

    def h(self, t, h):  # Prior drift
        target = self.mu(torch.cat([h, self.u_inter(t)], dim=-1))
        return self.theta * (target - h)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, :self.h_size]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        h = y[:, :self.h_size]
        g = self.g(t, h)
        g_logqp = torch.zeros_like(y[:, self.h_size:])
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, us, ys, batch_size, eps=None):
        eps = torch.randn(batch_size, self.h_size).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        self.update_u(ts, us)
        self.update_y(ts, ys)
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts[:, 0],
            method=args.method,
            dt=args.dt,
            adaptive=args.adaptive,
            rtol=args.rtol,
            atol=args.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        # aug_ys = sdeint_fn_batched(self, aug_y0, ts, sdeint_fn, method=args.method, dt=args.dt, adaptive=args.adaptive,
        #                   rtol=args.rtol, atol=args.atol, names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        ys, logqp_path = aug_ys[:, :, :self.h_size], aug_ys[-1, :, -1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, us, batch_size, y0=None, eps=None, bm=None):
        self.update_u(ts, us)
        eps = torch.randn(batch_size, self.h_size).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std if y0 is None else y0
        return sdeint_fn(self, y0, ts[:, 0], bm=bm, method=args.method, dt=args.dt, names={'drift': 'h'})
        # return sdeint_fn_batched(self, y0, ts, sdeint_fn, method=args.method, dt=args.dt, names={'drift': 'h'})

    def sample_q(self, ts, us, ys, batch_size, y0=None, eps=None, bm=None):
        self.update_u(ts, us)
        self.update_y(ts, ys)
        eps = torch.randn(batch_size, self.h_size).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std if y0 is None else y0
        return sdeint_fn(self, y0, ts[:, 0], bm=bm, method=args.method, dt=args.dt)
        # return sdeint_fn_batched(self, y0, ts, sdeint_fn, method=args.method, dt=args.dt)



    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)


def make_self_dataset(data_name, ct_time, sp, batch_size):
    train_all, test_all = select_dataset(data_name, ct_time, sp, evenly=args.evenly, batch_size=-1)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, arrays):
            self.arrays = arrays

        def __getitem__(self, index):
            return tuple(data[index] for data in self.arrays)

        def __len__(self):
            return self.arrays[0].shape[0]

    train_data_loader = torch.utils.data.DataLoader(Dataset(train_all), batch_size=batch_size, shuffle=True, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(Dataset(test_all), batch_size=batch_size, shuffle=True, num_workers=8)

    return train_data_loader, test_data_loader


def main():
    # Dataset.
    # ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_ = make_data()

    logger.info(f"data: {args.data}")

    train_data_loader, test_data_loader = make_self_dataset(args.data, args.ct_time, args.sp, args.batch_size)

    # Plotting parameters.
    # From https://colorbrewer2.org/.
    if args.data == "cstr":
        args.u_size, args.y_size = 1, 2
    elif args.data == "winding":
        args.u_size, args.y_size = 5, 2
    elif args.data == 'thickener':
        args.u_size, args.y_size = 4, 1

    # Model.
    model = LatentSDE(args.h_size, args.y_size, args.u_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = LinearScheduler(iters=args.kl_anneal_iters)

    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()

    best_test = 1e12
    best_dev_epoch = -1

    # if args.show_prior:
    #     with torch.no_grad():
    #         zs = model.sample_p(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
    #         ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
    #         zs_ = np.sort(zs_, axis=1)
    #
    #         img_dir = os.path.join(args.train_dir, 'prior.png')
    #         plt.subplot(frameon=False)
    #         for alpha, percentile in zip(alphas, percentiles):
    #             idx = int((1 - percentile) / 2. * vis_batch_size)
    #             zs_bot_ = zs_[:, idx]
    #             zs_top_ = zs_[:, -idx]
    #             plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)
    #
    #         # `zorder` determines who's on top; the larger the more at the top.
    #         plt.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)  # Data.
    #         plt.ylim(ylims)
    #         plt.xlabel('$t$')
    #         plt.ylabel('$Y_t$')
    #         plt.tight_layout()
    #         plt.savefig(img_dir, dpi=args.dpi)
    #         plt.close()
    #         logging.info(f'Saved prior figure at: {img_dir}')

    for global_step in tqdm.tqdm(range(args.train_epochs)):
        # Plot and save.
        if global_step % args.pause_iters == 0 and args.eval:
            img_path = os.path.join(args.train_dir, f'global_step_{global_step}.png')

            with torch.no_grad():
                # for i, (ts, ys) in enumerate(test_data_loader):
                rmse_sum = 0
                rrse_sum = 0
                sum_bs = 0
                for i, (t1, t2, u1, u2, y1, y2) in enumerate(test_data_loader):
                    logger.info(f"eval {len(test_data_loader)}-{i}")
                    batch_size, length, _ = y1.shape
                    t1, t2, u1, u2, y1, y2 = [trans(x, device) for x in [t1, t2, u1, u2, y1, y2]]

                    ys = model.sample_q(t1, u1, y1, batch_size)
                    zs = model.sample_p(t2, u2, batch_size=batch_size, y0=ys[-1])
                    pred_ys = model.h2y(zs)

                    rmse_sum += RMSE_torch(y2, pred_ys[..., :model.y_size]) * batch_size
                    rrse_sum += RRSE_torch(y2, pred_ys[..., :model.y_size]) * batch_size
                    sum_bs = sum_bs + batch_size

            # print(f"RMSE: {rmse_sum / sum_bs}")
            logger.info(
                f'\n RMSE: {rmse_sum / sum_bs}'
            )

            logger.info(
                f'\n RRSE: {rrse_sum / sum_bs}'
            )

            # ???RMSE??????????????????
            epoch_test = rmse_sum / sum_bs
            if best_test > epoch_test:
                best_test = epoch_test
                best_dev_epoch = global_step
                logger.info('best_test update at epoch = {}'.format(global_step))
                if args.save_ckpt:
                    torch.save(
                        {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'kl_scheduler': kl_scheduler},
                        os.path.join(ckpt_dir, f'global_step_{global_step}.ckpt')
                    )

            if global_step - best_dev_epoch > args.max_epochs_stop and global_step > args.min_epochs:
                logger.info('Early stopping at epoch = {}'.format(global_step))
                break

            # if args.save_ckpt:
            #     torch.save(
            #         {'model': model.state_dict(),
            #          'optimizer': optimizer.state_dict(),
            #          'scheduler': scheduler.state_dict(),
            #          'kl_scheduler': kl_scheduler},
            #         os.path.join(ckpt_dir, f'global_step_{global_step}.ckpt')
            #     )

        # Train.
        optimizer.zero_grad()
        for i, (t1, t2, u1, u2, y1, y2) in enumerate(train_data_loader):
            tr = TimeRecorder()

            with tr('data'):
                batch_size, length, _ = y1.shape
                t, u, y = torch.cat([t1, t2], dim=1), torch.cat([u1, u2], dim=1), torch.cat([y1, y2], dim=1)
                t, u, y = [trans(x, device) for x in [t, u, y]]
            with tr('forward'):
                zs, kl = model(t, u, y, batch_size=batch_size)
                # likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}[args.likelihood]
                # likelihood = likelihood_constructor(loc=zs, scale=args.scale)
                ys_mean, ys_logstd = torch.split(model.h2y(zs), model.y_size, dim=-1)
                likelihood = distributions.Normal(loc=ys_mean, scale=torch.nn.functional.softplus(ys_logstd))
                logpy = likelihood.log_prob(y).sum(dim=(0,2)).mean(dim=0)
                loss = -logpy + kl * kl_scheduler.val

            with tr('loss and train'):
                loss.backward()

                optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            logpy_metric.step(logpy)
            kl_metric.step(kl)
            loss_metric.step(loss)

            logger.info(
                f'\n epoch: {global_step}-{i}, '
                f'logpy: {logpy_metric.val:.3f}, '
                f'kl: {kl_metric.val:.3f}, '
                f'loss: {loss_metric.val:.3f}, '
                f'time: {tr.__str__()}'
            )

if __name__ == '__main__':
    # The argparse format supports both `--boolean-argument` and `--boolean-argument True`.
    # Trick from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--debug', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--save-ckpt', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--eval', type=str2bool, default=True, const=True, nargs="?")

    parser.add_argument('--data', type=str, default='cstr', choices=['cstr', 'winding', 'thickener'])
    parser.add_argument('--inter', type=str, default='cubic', choices=['gp', 'cubic'])
    parser.add_argument('--ct_time', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--evenly', type=str2bool, default=False)
    parser.add_argument('--sp', type=float, default=0.5, help='sp rate.')
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--u_size', type=int, default=1, help='Size of input')
    parser.add_argument('--y_size', type=int, default=1, help='Size of y.')
    parser.add_argument('--h_size', type=int, default=32, help='Size of hidden state in SDE.')
    parser.add_argument('--train-epochs', type=int, default=800, help='Number of epochs for training.')
    parser.add_argument('--pause-iters', type=int, default=5, help='Number of iterations before pausing.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace'], default='normal')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale parameter of Normal and Laplace.')
    parser.add_argument('--max_epochs_stop', type=int, default=50, help='Number of max epochs for training stop')
    parser.add_argument('--min_epochs', type=int, default=200, help='Number of min epochs for training')

    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)

    parser.add_argument('--show-prior', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-samples', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-percentiles', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-arrows', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-mean', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--hide-ticks', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    manual_seed(args.seed)

    # if args.debug:

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    time_now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    test_log = logging.FileHandler(f'logs/latent_sde_{args.data}_sp_{str(args.sp)}_evenly_{str(args.evenly)}_{str(time_now)}.log', 'w', encoding='utf-8')
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

    sdeint_fn = torchsde.sdeint_adjoint if args.adjoint else torchsde.sdeint

    main()