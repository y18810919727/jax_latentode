import pandas as pd
from common import *
from dataset import get_data
from dataset import select_dataset
from dataset import CstrDataset, WesternDataset
import time


def main():
    # data_dir = 'data/west'
    # data_csvs = [pd.read_csv(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
    # dataset_split = [0.6, 0.2, 0.2]
    # train_size, val_size, test_size = [int(len(data_csvs) * ratio) for ratio in dataset_split]
    # history_length = 120
    # forward_length = 120
    # dataset_window = 5
    # dilation = 2
    #
    # train_dataset = WesternDataset(data_csvs[:train_size], history_length + forward_length,
    #                                step=dataset_window, dilation=dilation)

    dataset_name = 'thickener'
    ct_time = True
    sp = 0.5
    train_loader = select_dataset(dataset_name, ct_time, sp)

    # df = train_dataset.df_split_all
    print(train_loader)



if __name__ == "__main__":
    main()
