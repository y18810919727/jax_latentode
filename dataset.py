#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import *
from common import subsample_indexes, to_timestamp


def get_data(dataset_size, *, key):
    TIME_SAMPLE_NUM = 20
    ykey, tkey1, tkey2 = jrandom.split(key, 3)

    y0 = jrandom.normal(ykey, (dataset_size, 2))
    u0 = jnp.zeros((dataset_size, 3))

    t0 = 0
    t1 = 2 + jrandom.uniform(tkey1, (dataset_size,))
    ts = jrandom.uniform(tkey2, (dataset_size, TIME_SAMPLE_NUM)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func_u(t, y, args):
        return t

    def func(t, y, args):
        return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve_y(ts, y0):
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

    def solve_u(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func_u),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve_y)(ts, y0)
    us = jax.vmap(solve_u)(ts, u0)

    return ts, ys, us


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
            # TODO: 在这里再变为jnp
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def select_dataset(dataset_name, ct_time, sp):
    if dataset_name == 'cstr':
        train_path = 'data/cstr/cstr_train.csv'
        val_path = 'data/cstr/cstr_val.csv'
        test_path = 'data/cstr/cstr_test.csv'
        history_length = 60
        forward_length = 180
        dataset_window = 2
        input_dim = 1
        output_dim = 2

        train_dataset = CstrDataset(pd.read_csv(train_path), history_length + forward_length, step=dataset_window)
        # val_dataset = CstrDataset(pd.read_csv(val_path), history_length + forward_length, step=dataset_window)
        test_dataset = CstrDataset(pd.read_csv(test_path), history_length + forward_length, step=dataset_window)
        collate_fn = None if not ct_time else \
            CTSample(sp=sp, history_length=history_length, forward_length=forward_length).batch_collate_fn

    elif dataset_name == 'winding':
        train_path = 'data/winding/winding_train.csv'
        val_path = 'data/winding/winding_val.csv'
        test_path = 'data/winding/winding_test.csv'
        history_length = 60
        forward_length = 180
        dataset_window = 2
        input_dim = 5
        output_dim = 2

        train_dataset = WindingDataset(pd.read_csv(train_path), history_length + forward_length, step=dataset_window)
        # val_dataset = WindingDataset(pd.read_csv(val_path), history_length + forward_length, step=dataset_window)
        test_dataset = WindingDataset(pd.read_csv(test_path), history_length + forward_length, step=dataset_window)
        collate_fn = None if not ct_time else \
            CTSample(sp=sp, history_length=history_length, forward_length=forward_length).batch_collate_fn

    elif dataset_name == 'thickener':
        data_dir = 'data/west'
        data_csvs = [pd.read_csv(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
        dataset_split = [0.6, 0.2, 0.2]
        train_size, val_size, test_size = [int(len(data_csvs) * ratio) for ratio in dataset_split]
        history_length = 120
        forward_length = 120
        dataset_window = 5
        dilation = 2
        input_dim = 8
        output_dim = 1

        train_dataset = WesternDataset(data_csvs[:train_size], history_length + forward_length,
                                       step=dataset_window, dilation=dilation)
        # val_dataset = WesternDataset(data_csvs[train_size:train_size + val_size], history_length + forward_length,
        #                                step=dataset_window, dilation=dilation)
        test_dataset = WesternDataset(data_csvs[-test_size:], history_length + forward_length,
                                    step=dataset_window, dilation=dilation)

        collate_fn = None if not ct_time else \
            CTSample(sp=sp, history_length=history_length, forward_length=forward_length).batch_collate_fn

    else:
        raise NotImplementedError

    def get_jnp_list(data_loader):
        for i, data in enumerate(data_loader):
            external_input_history, external_input_forward, observation_history, observation_forward = data
        external_input_history, ts_history = external_input_history.split([input_dim, 1], dim=-1)
        external_input_forward, ts_forward = external_input_forward.split([input_dim, 1], dim=-1)
        # external_input_history = torch.cat([external_input_history, tp_history], dim=-1)
        # external_input_forward = torch.cat([external_input_forward, tp_forward], dim=-1)

        ts_history = torch.clip(ts_history, 1e-6, 1e6)
        ts_forward = torch.clip(ts_forward, 1e-6, 1e6)
        ts_history = torch.cumsum(ts_history, dim=1)
        ts_forward = torch.cumsum(ts_forward, dim=1)
        ts_history = jnp.array(ts_history.squeeze(dim=-1).numpy().tolist())
        ts_forward = jnp.array(ts_forward.squeeze(dim=-1).numpy().tolist())
        external_input_history = jnp.array(external_input_history.numpy().tolist())
        external_input_forward = jnp.array(external_input_forward.numpy().tolist())
        observation_history = jnp.array(observation_history.numpy().tolist())
        observation_forward = jnp.array(observation_forward.numpy().tolist())

        return [ts_history, ts_forward, external_input_history, external_input_forward,
                            observation_history, observation_forward]

    train_loader = DataLoader(train_dataset, batch_size=100000,
                              shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=100000,
                             shuffle=True, num_workers=8, collate_fn=collate_fn)

    jnp_train = get_jnp_list(train_loader)
    jnp_test = get_jnp_list(test_loader)

    return jnp_train, jnp_test


class CstrDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['in','out1', 'out2']
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

    def get_ys(self):
        data_in = np.array(self.df['0'], dtype=np.float32)
        data_out = np.array(self.df[['1', '2']], dtype=np.float32)

        return [np.expand_dims(data_in, axis=1), data_out]


class WindingDataset(Dataset):
    def __init__(self, df, length=1000, step=5):
        df_split_all = []
        begin_pos = []

        # 每个column对应的数据含义 ['in','out1', 'out2']
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
            step: 数据segment切割窗口的移动步长
            dilation: 浓密机数据采样频率(1 min)过高，dilation表示数据稀释间距
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        df_split_all = []
        begin_pos_pair = []

        # 每个column对应的数据含义 ['c_in','c_out', 'v_out', 'v_in', 'pressure']
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
        # df_time = df_all.loc[:, ['Timestamp']]
        # df_all = df_all.drop(['Timestamp'], axis=1)
        mean = df_all.mean()
        std = df_all.std()
        # df_all = pd.concat([df_time, df_all], axis=1)
        df_all_list = [(df - mean) / std for df in df_all_list]
        # df_all_list_new = []
        # for df in df_all_list:
        #     df_time = df.loc[:, ['Timestamp']]
        #     # df_time转时间戳
        #     for i in range(len(df_time['Timestamp'])):
        #         df_time['Timestamp'][i] = to_timestamp(df_time['Timestamp'][i])
        #
        #     df = df.drop(['Timestamp'], axis=1)
        #     df = (df - mean) / std
        #     df = pd.concat([df_time, df], axis=1)
        #     df_all_list_new.append(df)

        return df_all_list

    def split_df(self, df):
        """
        将存在空值的位置split开
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
                c_out,
            ],
            axis=1)
        observation = pressure

        return external_input, np.expand_dims(observation, axis=1)


class CTSample:
    def __init__(self, sp: float, base_tp=0.1, evenly=False, history_length=500, forward_length=500):
        self.sp = np.clip(sp, 0.01, 1.0)
        self.base_tp = base_tp
        self.evenly = evenly
        self.history_length = history_length
        self.forward_length = forward_length

    def batch_collate_fn(self, batch):

        external_input, observation = [torch.from_numpy(np.stack(x)) for x in zip(*batch)]
        # external_input_origin = external_input
        # observation_origin = observation


        def data_rasample(external_input, observation):
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

            return external_input, observation

        # ts_history = ts[:, :history_length, :]
        # ts_forward = ts[:, forward_length:, :]
        external_input_history, observation_history = data_rasample(
            external_input[:, :self.history_length, :], observation[:, :self.history_length, :]
        )

        external_input_forward, observation_forward = data_rasample(
            external_input[:, self.history_length:, :], observation[:, self.history_length:, :]
        )
        # observation = add_tp(observation, dt)
        # return external_input, observation, external_input_origin, observation_origin
        return external_input_history, external_input_forward, observation_history, observation_forward

