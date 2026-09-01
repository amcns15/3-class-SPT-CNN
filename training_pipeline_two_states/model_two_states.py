import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import  Dropout, Dense, MaxPool3D, Input, GlobalAveragePooling3D, BatchNormalization

# class Conv2Plus1D(keras.layers.Layer):  # from online tensorflow article
#     def __init__(self, filters,kernel_size, padding = "same"):
#     # apply convolution spacially and then temporally
#         super().__init__()
#         self.seq = keras.Sequential([
#             # Spatial decomposition
#             layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
#             layers.BatchNormalization(),
#             layers.ReLU(),
#             # Temporal decomposition
#             layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding),
#             layers.BatchNormalization(),
#             layers.ReLU()
#         ])

class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="same"):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x, training=None):
        return self.seq(x, training=training)

    def call(self, x):
        return self.seq(x)


def model_simple(frame = 5, height=44, width=16, channels = 1):
    model = keras.Sequential([
        keras.Input(shape=(frame, height, width, channels)),

    Conv2Plus1D(
        filters=8, 
        kernel_size=(3, 5, 5),
        padding='same'
    ),

    layers.MaxPool3D(
        pool_size=(1, 2, 2), 
        strides=(1, 2, 2), 
        padding='same'
    ),

    Conv2Plus1D(
        filters=16, 
        kernel_size=(3, 3, 3),
        padding='same'
    ),

    layers.MaxPool3D(
        pool_size=(1, 2, 2), 
        strides=(1, 2, 2), 
        padding='same'
    ),

    Conv2Plus1D(
        filters=32, 
        kernel_size=(1, 3, 3),
        padding='same'
    ),

    layers.MaxPool3D(
        pool_size=(1, 2, 2), 
        strides=(1, 2, 2), 
        padding='same'
    ),

    layers.GlobalAveragePooling3D(),

    layers.Dense(32, activation='swish'),
    layers.Dropout(0.3),

    # layers.Dense(16, activation='relu'),
    # layers.Dropout(0.5),

    layers.Dense(1, activation = "sigmoid")
    ])


    return model


    # model.add(Input(shape=(frame, height, width, channels)))

    # model.add(c)
    # model.add(MaxPool3D(pool_size=(1, 2, 2)))
    
    # model.add(Conv2Plus1D(filters=16, kernel_size=(3, 3, 3), padding='same'))
    # model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # model.add(Conv2Plus1D(filters=32, kernel_size=(3, 3, 3), padding='same'))
    # model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # model.add(GlobalAveragePooling3D())
    # model.add(Dropout(0.5))


    # model.add(Dense(1, activation='sigmoid'))