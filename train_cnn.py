import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,TensorBoard,ModelCheckpoint
from model import CNN, CNN_radiologist, cross_entropy
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
import DataAugmentor
from Data_generator import batch_index_gen
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from skimage import transform

def arg_data(im_list, stretch_x, stretch_y, shear, rotate_z_angle, shift_x, shift_y):
    # the shape of  data and roi is [ H, W, Slice]
    random_params = {'stretch_x': stretch_x, 'stretch_y': stretch_y, 'shear': shear,
                         'rotate_z_angle': rotate_z_angle, 'shift_x': shift_x, 'shift_y': shift_y}
    param_generator = DataAugmentor.AugmentParametersGenerator()
    aug_generator = DataAugmentor.DataAugmentor3D()
    param_generator.RandomParameters(random_params)
    aug_generator.SetParameter(param_generator.GetRandomParametersDict())
    arg_data_list = []
    for im in im_list:
        arg_data = aug_generator.Execute(im)
        arg_data_list.append(arg_data)
    return arg_data_list


def batch_data_gen(input_size, list_name0, list_name1, file_name, index_file_path,data_path, batch_size=16):
    while True:
        index_gen = batch_index_gen(list_name0, list_name1, file_name, index_file_path, data_path, batch_size)
        for batch_index, label_list in index_gen:
            data_block = []
            for index in batch_index:
                data = np.load(index)
                data_resized = transform.resize(data, (input_size[0], input_size[1] , input_size[2]))
                data_aug = arg_data([data_resized], 0.2, 0.2, 0.2, 12, 10, 10)[0]
                data_aug = (data_aug - np.mean(data_aug)) / np.std(data_aug)
                data_block.append(data_aug)

            input_label = to_categorical(np.asarray(label_list), num_classes=2)
            input_data = np.asarray(data_block).astype(np.float32)
            input_data = np.squeeze(input_data)
            input_data = np.expand_dims(input_data, axis=4)

            # for i in range(input_data.shape[0]):
            #     plt.imshow(input_data[i, :, :, 4, 0], cmap='gray')
            #     plt.show()
            yield (input_data, input_label)


def trainModel(input_size, EPOCHS, Batch_size, STEPS_PER_EPOCH, val_steps):
    trainGenerator = batch_data_gen(input_size, list_name0, list_name1, 'train_index.npy', index_file_path, data_path, batch_size=Batch_size)
    validationGenerator = batch_data_gen(input_size, list_name0, list_name1, 'val_index.npy', index_file_path, data_path, batch_size=Batch_size)
    callbacks = [
        TensorBoard(),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min'),
        EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.001),
        ModelCheckpoint(filepath='RESULT1/valloss{val_loss:.3f}_loss{loss:.3f}_classify_epoch{epoch:03d}.hdf5',
                        monitor='val_loss',
                        mode='min', period=1)
    ]
    model = CNN(input_size)
    model.compile(optimizer=Adam(lr=1e-3), loss=cross_entropy, metrics=['accuracy', tf.keras.metrics.AUC()])
    history = model.fit_generator(trainGenerator, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                                  validation_data=validationGenerator,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)
    print('train finish!')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    index_file_path = r'/home/wyd/PycharmProjects/nodule-yuan/PGGO/index'
    data_path = r'/home/wyd/PycharmProjects/nodule-yuan/PGGO'

    list_name0 = ['AIS', 'MIA']
    list_name1 = ['IAC']

    # list_name0 = ['AIS']
    # list_name1 = ['MIA', 'IAC']

    # list_name0 = ['AIS']
    # list_name1 = ['IAC']

    BATCH_SIZE = 16
    EPOCHS = 200
    input_size = [48, 48, 8, 1]

    os.makedirs('RESULT1')
    trainModel(input_size, EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH=21, val_steps=5)

