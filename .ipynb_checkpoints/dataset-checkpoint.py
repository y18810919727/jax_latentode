#!/usr/bin/python
# -*- coding:utf8 -*-
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import *


def get_data(dataset_size, *, key):
    ykey, tkey1, tkey2 = jrandom.split(key, 3)

    y0 = jrandom.normal(ykey, (dataset_size, 2))

    t0 = 0
    t1 = 2 + jrandom.uniform(tkey1, (dataset_size,))
    ts = jrandom.uniform(tkey2, (dataset_size, 20)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func(t, y, args):
        return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def select_dataset(dataset_name):
    if dataset_name == 'cstr':
        objects = pd.read_csv('data/cstr/data_url.csv')
        base = 'data/cstr'
        if not os.path.exists(base):
            os.mkdir(base)
        # _ = detect_download(objects, base)
        _ = detect_download(objects,
                            base,
                            'http://oss-cn-beijing.aliyuncs.com',
                            'io-system-data',
                            access_key['AccessKey ID'][0],
                            access_key['AccessKey Secret'][0]
                            )
        train_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.train_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)
        val_dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.val_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)



class CstrDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # ??????column????????????????????? ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df['0'], dtype=np.float32)
        data_out = np.array(data_df[['1', '2']], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return np.expand_dims(data_in, axis=1), data_out


class WindingDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # ??????column????????????????????? ['in','out1', 'out2']
        self.df = df
        self.used_columns = ['0', '1', '2', '3', '4', '5', '6']
        self.length = length
        for j in range(0, df.shape[0] - length + 1, step):
            begin_pos.append(j)
        self.begin_pos = begin_pos
        self.df = self.normalize(self.df)

    def normalize(self, df):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    def __len__(self):
        return len(self.begin_pos)

    def __getitem__(self, item):
        pos = self.begin_pos[item]
        data_df = self.df.iloc[pos:pos + self.length]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        data_in = np.array(data_df[['0', '1', '2', '3', '4']], dtype=np.float32)
        data_out = np.array(data_df[['5', '6']], dtype=np.float32)

        # return np.expand_dims(data_in, axis=1), np.expand_dims(data_out, axis=1)
        return data_in, data_out


class WesternDataset(Dataset):
    def __init__(self, df_list, length=1000, step=5, dilation=2):
        """

        Args:
            df_list:
            length:
            step: ??????segment???????????????????????????
            dilation: ???????????????????????????(1 min)?????????dilation????????????????????????
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        # ??????column????????????????????? ['c_in','c_out', 'v_out', 'v_in', 'pressure']
        self.used_columns = ['4', '11', '14', '16', '17']
        self.length = length
        self.dilation = dilation

        for df in df_list:
            df_split_all = df_split_all + self.split_df(df[self.used_columns])
        for i, df in enumerate(df_split_all):
            for j in range(0, df.shape[0] - length * dilation + 1, step):
                begin_pos_pair.append((i, j))
        self.begin_pos_pair = begin_pos_pair
        self.df_split_all = df_split_all
        self.df_split_all = self.normalize(self.df_split_all)

    def normalize(self, df_all_list):
        df_all = df_all_list[0].append(df_all_list[1:], ignore_index=True)
        mean = df_all.mean()
        std = df_all.std()
        return [(df - mean) / std for df in df_all_list]

    def split_df(self, df):
        """
        ????????????????????????split???
        Args:
            df:
        Returns: list -> [df1,df2,...]
        """
        df_list = []
        split_indexes = list(
            df[df.isnull().T.any()].index
        )
        split_indexes = [-1] + split_indexes + [df.shape[0]]
        for i in range(len(split_indexes) - 1):
            if split_indexes[i + 1] - split_indexes[i] - 1 < self.length:
                continue

            new_df = df.iloc[split_indexes[i] + 1:split_indexes[i + 1]]
            assert new_df.isnull().sum().sum() == 0
            df_list.append(new_df)
        return df_list

    def __len__(self):
        return len(self.begin_pos_pair)

    def __getitem__(self, item):
        df_index, pos = self.begin_pos_pair[item]
        data_array = np.array(self.df_split_all[df_index].iloc[pos:pos + self.length * self.dilation], dtype=np.float32)
        data_array = data_array[np.arange(self.length) * self.dilation]
        # c_in = data_array[:, 0]
        # c_out = data_array[:, 1]
        c_in, c_out, v_out, v_in, pressure = [np.squeeze(x, axis=1) for x in np.hsplit(data_array, 5)]

        v_in = v_in * 0.05
        v_out = v_out * 0.05

        external_input = np.stack(
            [
                c_in * c_in * c_in * v_in - c_out * c_out * c_out * v_out,
                c_in * c_in * v_in - c_out * c_out * v_out,
                c_in * v_in - c_out * v_out,
                v_in - v_out,
                v_in,
                v_out,
                c_in,
                c_out
            ],
            axis=1)
        observation = pressure
        return external_input, np.expand_dims(observation, axis=1)


class CTSample:
    def __init__(self, sp: float, base_tp=0.1, evenly=False):
        self.sp = np.clip(sp, 0.01, 1.0)
        self.base_tp = base_tp
        self.evenly = evenly

    def batch_collate_fn(self, external_input, observation):

        # external_input, observation = [torch.from_numpy(np.stack(x)) for x in zip(*batch)]
        bs, l, _ = external_input.shape
        time_steps = torch.arange(external_input.size(1)) * self.base_tp
        data = torch.cat([external_input, observation], dim=-1)
        new_data, tp = subsample_indexes(data, time_steps, self.sp, evenly=self.evenly)
        external_input, observation = new_data[..., :external_input.shape[-1]], new_data[..., -observation.shape[-1]:]

        # region [ati, t_{i} - t_{i-1}]
        # tp = torch.cat([tp[..., 0:1], tp], dim=-1)
        # dt = tp[..., 1:] - tp[..., :-1]
        # endregion

        # region [ati, t_{i+1} - t_{i}]
        tp = torch.cat([tp, tp[..., -1:]], dim=-1)
        dt = tp[..., 1:] - tp[..., :-1]
        # endregion

        def add_tp(x, tp):
            return torch.cat([
                x,
                tp.repeat(bs, 1).unsqueeze(dim=-1)
            ], dim=-1)

        external_input = add_tp(external_input, dt)
        # observation = add_tp(observation, dt)
        return external_input, observation
