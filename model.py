import tensorflow as tf
from tensorflow.keras.layers import Input,Conv3D,BatchNormalization, PReLU
from tensorflow.keras.layers import multiply,Reshape
from tensorflow.keras.layers import GlobalAveragePooling3D, Dense, Concatenate, Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

def conv_block(input, out_dim, name, acitvation=True, kernel_size= (3, 3, 1), strides=1):
    bias_initial = Constant(value=0.1)
    x =Conv3D(filters=out_dim, kernel_size=kernel_size, strides=strides, bias_initializer = bias_initial, padding='same', name= name + '_conv')(input)
    x = BatchNormalization(name=name + '_bn')(x)

    if acitvation:
        # output = Mish()(x)
        output = PReLU(shared_axes=[1, 2, 3])(x)
    else:
        output = x
    return output


def SE_block(input, ratio=8):
    x1 = GlobalAveragePooling3D()(input)
    x2 = Dense(x1.shape[-1]//ratio, activation='relu')(x1)
    x3 = Dense(x1.shape[-1], activation='sigmoid')(x2)
    x4 = Reshape((1, 1, 1, x1.shape[-1]))(x3)
    result = multiply([input, x4])
    return result


def res_bottleneck_layer_SE(input, out_dim, name, stride=1):
    conv = Conv3D(filters=out_dim, kernel_size=1, strides=stride, padding='same', name = name + '_3')(input)
    residual = BatchNormalization()(conv)

    x = conv_block(input, int(out_dim/4), kernel_size=1,name = name + '_1')
    x = conv_block(x, int(out_dim/4), kernel_size=3,strides = stride, name = name + '_2')
    x = conv_block(x, out_dim, acitvation=False, kernel_size=1, name = name + '_3')
    x = SE_block(x)
    x = x + residual
    output = PReLU(shared_axes=[1, 2, 3])(x)
    return output


def CNN(input_size, filters=[8, 16, 32, 64]):
    input = Input(shape=input_size, name = 'input')
    input_feature = conv_block(input, filters[0], name= 'input_conv')

    conv1_1 = res_bottleneck_layer_SE(input_feature, out_dim=filters[0], name='block0-1', stride=2)
    conv1_2 = res_bottleneck_layer_SE(conv1_1, out_dim=filters[0], name='block0-2', stride=1)

    conv2_1 = res_bottleneck_layer_SE(conv1_2, out_dim=filters[0], name='block1-1', stride=1)
    conv2_2 = res_bottleneck_layer_SE(conv2_1, out_dim=filters[0], name='block1-2', stride=1)


    conv3_1 = res_bottleneck_layer_SE(conv2_2, out_dim=filters[1], name='block2-1', stride=2)
    conv3_2 = res_bottleneck_layer_SE(conv3_1, out_dim=filters[1], name='block2-2', stride=1)

    #
    conv4_1 = res_bottleneck_layer_SE(conv3_2, out_dim=filters[2], name='block3-1', stride=1)
    conv4_2 = res_bottleneck_layer_SE(conv4_1, out_dim=filters[2], name='block3-2', stride=1)

    avg_pool = GlobalAveragePooling3D()(conv4_2)
    fc1 = Dense(32, name='fc1')(avg_pool)
    fc1 = PReLU()(fc1)
    prob = Dense(2, activation='sigmoid', name='prob')(fc1)
    model = Model(inputs=[input], outputs=[prob])
    return model


def CNN_radiologist(input_size1, input_size2, filters=[8, 16, 32, 64]):
    input1 = Input(shape=input_size1, name = 'input1')
    input2 = Input(shape=input_size2, name = 'input2')

    input_feature = conv_block(input1, filters[0], name= 'input_conv')

    conv1_1 = res_bottleneck_layer_SE(input_feature, out_dim=filters[0], name='block0-1', stride=2)
    conv1_2 = res_bottleneck_layer_SE(conv1_1, out_dim=filters[0], name='block0-2', stride=1)

    conv2_1 = res_bottleneck_layer_SE(conv1_2, out_dim=filters[0], name='block1-1', stride=1)
    conv2_2 = res_bottleneck_layer_SE(conv2_1, out_dim=filters[0], name='block1-2', stride=1)


    conv3_1 = res_bottleneck_layer_SE(conv2_2, out_dim=filters[1], name='block2-1', stride=2)
    conv3_2 = res_bottleneck_layer_SE(conv3_1, out_dim=filters[1], name='block2-2', stride=1)

    conv4_1 = res_bottleneck_layer_SE(conv3_2, out_dim=filters[1], name='block3-1', stride=1)
    conv4_2 = res_bottleneck_layer_SE(conv4_1, out_dim=filters[1], name='block3-2', stride=1)

    avg_pool = GlobalAveragePooling3D()(conv4_2)
    avg_pool = Concatenate(axis=-1)([avg_pool, input2])
    fc1 = Dense(32, name='fc1')(avg_pool)
    fc1 = PReLU()(fc1)
    fc1 = Concatenate(axis=-1)([fc1, input2])
    prob = Dense(2, activation='sigmoid', name='prob')(fc1)
    model = Model(inputs=[input1, input2], outputs=[prob])
    return model


def cross_entropy(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)