import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import cross_entropy
from skimage import transform
from Data_generator import get_index


class ModelTester:
    def __init__(self,
                 list_name0=None,
                 list_name1=None,
                 index_file=None,
                 index_file_path=None,
                 data_path=None,
                 image_size=[]):

        self.data_path = data_path
        self.list_name0 = list_name0
        self.list_name1 = list_name1
        self.index_file = index_file
        self.index_file_path = index_file_path
        self.image_size = image_size
        [self.index0, self.index1] = get_index(self.list_name0, self.list_name1,
                                               index_file, index_file_path, data_path)

    def set_model(self, model):
        self.trained_model = model

    def get_label(self, index0, index1):
        s0 = len(index0)
        s1 = len(index1)
        label = np.zeros(s0 + s1)
        label[s0:] = 1
        return label

    def predicion(self, indexs):
        logits = []
        for index in indexs:
            data = np.load(index)
            data_resized = transform.resize(data, (input_size[0],input_size[1], input_size[2]))
            input_data = (data_resized - np.mean(data_resized)) / np.std(data_resized)
            input_data = np.expand_dims(input_data, axis=[0, 4])
            output = self.trained_model.predict(input_data)
            logits.append(output[0][1])
            print(logits)
        return logits

    def get_result(self):
        logits0 = self.predicion(self.index0)
        logits1 = self.predicion(self.index1)
        logits = logits0 + logits1
        label = self.get_label(self.index0, self.index1)
        assert len(logits) == len(label)
        return [logits, label]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    index_file_path = r'/home/wyd/PycharmProjects/nodule-yuan/PGGO/index'
    data_path = r'/home/wyd/PycharmProjects/nodule-yuan/PGGO'

    list_name0 = ['AIS']
    list_name1 = ['MIA', 'IAC']
    input_size = [48, 48, 8, 1]
    index_files = ['train_index.npy', 'val_index.npy', 'test_index.npy']

    model_path = r'valloss0.591_loss0.544_classify_epoch098.hdf5'
    train_model = tf.keras.models.load_model(model_path, custom_objects={'cross_entropy': cross_entropy})

    result_test = []
    for index_file in index_files:

        tester = ModelTester(
                 list_name0=list_name0,
                 list_name1=list_name1,
                 index_file=index_file,
                 index_file_path=index_file_path,
                 data_path=data_path,
                 image_size=input_size)

        tester.set_model(train_model)
        result_test.append(tester.get_result())

    from Result_analysis import classificaton_evaluation
    np.save('result1.npy', result_test[2])
    classificaton_evaluation(result_test)






