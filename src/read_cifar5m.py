import numpy as np
from torch.utils.data import Dataset

mean = np.array([0.4555, 0.4362, 0.3415])[None, None, None, :]
std = np.array([.2284, .2167, .2165])[None, None, None, :]


class BasicDataset(Dataset):
    def __init__(self, data_x, data_y):
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.data_x.shape[0]


def get_dataset(data_path=None):
    if data_path is None:
        data_path = '/n/holystore01/LABS/barak_lab/Everyone/cifar-5m'
    print(f'reading cifar 5m data from {data_path}')
    class_files = []
    class_labels = []
    min_len = np.inf
    for i in range(10):
        file_name = f'{data_path}/class{i}.npy'
        curr_data = np.load(file_name)
        class_files += [curr_data]
        class_labels += [i * np.ones(curr_data.shape[0])]
        min_len = np.min(min_len, curr_data.shape[0])

    for i in range(10):
        class_files[i] = class_files[i][:min_len]

    x = np.array((np.concatenate(class_files) / 255. - mean) / std)
    y = np.array((np.concatenate(class_labels)))

    return BasicDataset(x, y)