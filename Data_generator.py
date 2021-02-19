import numpy as np
import random
import os


def get_index(list_name0, list_name1, file_name, file_path, data_path):
    index0 = []
    index1 = []
    for name0 in list_name0:
        name0_path = os.path.join(file_path, name0, file_name)
        index0_set = np.load(name0_path).tolist()
        index0_file_set = [os.path.join(data_path, name0, x) for x in index0_set]
        index0 = index0 + index0_file_set
    for name1 in list_name1:
        name1_path = os.path.join(file_path, name1, file_name)
        index1_set = np.load(name1_path).tolist()
        index1_file_set = [os.path.join(data_path, name1, x) for x in index1_set]
        index1 = index1 + index1_file_set
    return index0, index1


def upsample_index(index0, index1):
    random.shuffle(index0)
    random.shuffle(index1)
    s0, s1 = len(index0), len(index1)
    index0_up,  index1_up= [], []
    if s0>=s1:
        t = np.ceil(s0 / s1)
        for i in range(int(t)):
            index1_up = index1_up + index1
        s = len(index1_up)
        index0_up = index0 + index0[:s - s0]

    else:
        t = np.ceil(s1 / s0)
        for i in range(int(t)):
            index0_up = index0_up + index0
        s = len(index0_up)
        index1_up = index1 + index1[:s - s1]
    n = len(index0_up)
    assert len(index0_up) == len(index1_up)
    return index0_up, index1_up, n


def batch_index_gen(list_name0, list_name1, file_name, file_path, data_path,  batch_size):
    [index0, index1] = get_index(list_name0, list_name1, file_name, file_path, data_path)
    [index0_up, index1_up, n] = upsample_index(index0, index1)
    range_time = n // batch_size
    print(range_time)
    for i in range(range_time):
        index1_batch = index1_up[i*batch_size: batch_size*(i + 1)]
        index0_batch = index0_up[i*batch_size: batch_size *(i + 1)]
        batch_index = index0_batch + index1_batch
        label_list = np.zeros(2*batch_size)
        label_list[batch_size:] = 1
        label_list = label_list.tolist()
        data = list(zip(batch_index, label_list))
        random.shuffle(data)
        batch_index[:], label_list[:] = zip(*data)
        yield (batch_index, label_list)



