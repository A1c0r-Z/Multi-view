import struct
import os
import numpy as np
import pandas as pd
import torch
from conf import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


def read_bin(file_name):
    """
    function: read a bin file, return the tuple of the content in file
    """
    frmt = ""
    length = 0
    if 'hvecs32' in file_name:
        frmt = "I"
        length = 4
    elif 'hvecs' in file_name:
        frmt = "H"
        length = 2
    elif 'fvec' in file_name:
        frmt = "f"
        frmt_ = "I"
        length = 4
        with open(file_name, "rb") as f:
            vec_length = struct.unpack(frmt_, f.read(length))[0]
            f.close()
    else:
        print("Error:", file_name)
        exit()
    
    with open(file_name, "rb") as f:
        f_content = f.read()
        if 'fvec' in file_name:
            frmt_ = "I"
            content = struct.unpack((frmt_ + frmt * vec_length) * int(len(f_content)/((1 + vec_length) * length)), f_content)
        else:
            content = struct.unpack(frmt * int(len(f_content)/length), f_content)
        f.close()
    return content


def read_(file_name, num):
    """
    function: read a file, return the data in vector form
    """
    data = read_bin(file_name)
    length = data[0]
    assert len(data)/(1+length) == num
    lst = []
    for i in range(num):
        vec = np.array(data[(length+1)*i+1:(length+1)*(i+1)])
        assert len(vec) == length
        lst.append(vec)
    return lst
    

def get_name(lst):
    """
    function: get train and test data name
    """
    other = []
    train = []
    test = []
    for i in lst:
        if 'txt' in i:
            other.append(i)
        elif 'train' in i:
            train.append(i)
        elif 'test' in i:
            test.append(i)
        elif 'ipynb' in i:
            continue
        else:
            print('Error:', i)
            exit()
    train.sort()
    test.sort()
    return other, train, test


def get_data(name, num):
    """
    function: get data in DataFrame form
    """
    data = pd.DataFrame()
    for n in name:
        data[n] = read_(path+'//'+n, num)
    return data


def get_tensor(train_data):
    length = []
    for i in range(len(train_data.iloc[0, :])):
        length.append(len(train_data.iloc[0, :][i]))
    max_length = max(length)
    
    view = (torch.cat([torch.tensor(train_data.values[:, 0][i]).unsqueeze(0) for i in range(len(train_data.values[:, 0]))], dim=0)).unsqueeze(1)
    if view.shape[2] < max_length:
        view = torch.cat([view, torch.zeros(view.shape[0], view.shape[1], max_length-view.shape[2])], dim=2)
    for j in range(len(train_data.values[0, :])-1):
        a = (torch.cat([torch.tensor(train_data.values[:, j+1][i]).unsqueeze(0) for i in range(len(train_data.values[:, 0]))], dim=0)).unsqueeze(1)
        if a.shape[2] < max_length:
            a = torch.cat([a, torch.zeros(a.shape[0], a.shape[1], max_length-a.shape[2])], dim=2)
        view = torch.cat([view, a], dim=1)
    return view, length


def get_data_loader():
    name = os.listdir(path)
    other_name, train_name, test_name = get_name(name)

    train_data = get_data(train_name, train_num)
    for i in train_data.columns:
        if 'annot' in i:
            train_target = train_data[i]
            del train_data[i]
            break
    train_data, train_length = get_tensor(train_data)
    train_target = torch.cat([torch.tensor(train_target.values[:][i]).unsqueeze(0) for i in range(len(train_target.values[:]))], dim=0)

    test_data = get_data(test_name, test_num)
    for i in test_data.columns:
        if 'annot' in i:
            test_target = test_data[i]
            del test_data[i]
            break
    test_data, test_length = get_tensor(test_data)
    test_target = torch.cat([torch.tensor(test_target.values[:][i]).unsqueeze(0) for i in range(len(test_target.values[:]))], dim=0)

    assert train_length == test_length
    assert max(train_length) == d_vec
    # data_tensor = torch.cat([train_data, test_data], dim=0)
    # target_tensor = torch.cat([train_target, test_target], dim=0)

    train_dataset = MyDataset(train_data.to(torch.float32), train_target.to(torch.float32))
    test_dataset = MyDataset(test_data.to(torch.float32), test_target.to(torch.float32))

    train_loader = DataLoader(dataset=train_dataset,  # 传入的数据集, 必须参数
                              batch_size=batch_size,        # 输出的batch大小
                              shuffle=True,        # 数据是否打乱
                              num_workers=0)      # 进程数, 0表示只有主进程

    test_loader = DataLoader(dataset=test_dataset,  # 传入的数据集, 必须参数
                             batch_size=batch_size,        # 输出的batch大小
                             shuffle=False,        # 数据是否打乱
                             num_workers=0)      # 进程数, 0表示只有主进程

    return train_loader, test_loader
