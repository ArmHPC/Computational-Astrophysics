import os

import numpy as np
from PIL import Image

import tensorflow as tf
# import tensorflow_transform as tft
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, \
    concatenate
from tensorflow.keras import regularizers

input_shape = (140, 20, 1)
num_classes = 5
reg = None
reg_l1 = regularizers.l1()
reg_l2 = regularizers.l2()
ks = 16
kshape = (3, 3)
drop_size = 0.15
act = "relu"

i = Input(input_shape)

# 32----------------------------------------------------------------------------------
t11 = Conv2D(ks, kshape, kernel_regularizer=reg, padding="same", activation=act)(i)
t11 = BatchNormalization()(t11)
t1i = concatenate((i, t11))

t12 = Conv2D(ks * 2, kshape, kernel_regularizer=reg, padding="same", activation=act)(t1i)
t12 = BatchNormalization()(t12)
t2i = concatenate((i, t11, t12))

t13 = Conv2D(ks * 4, kshape, kernel_regularizer=reg, padding="same", activation=act)(t2i)
t13 = BatchNormalization()(t13)
t13 = MaxPooling2D(2, 2)(t13)
t13 = Dropout(drop_size * 3)(t13)
# 16----------------------------------------------------------------------------------
t21 = Conv2D(ks * 2, (3, 3), kernel_regularizer=reg, padding="same", activation=act)(t13)
t21 = BatchNormalization()(t21)
t2i2 = concatenate((t13, t21))
t22 = Conv2D(ks * 4, (3, 3), kernel_regularizer=reg, padding="same", activation=act)(t2i2)
t22 = BatchNormalization()(t22)
t2i3 = concatenate((t13, t21, t22))
t23 = Conv2D(ks * 8, (3, 3), kernel_regularizer=reg, padding="same", activation=act)(t2i3)
t23 = BatchNormalization()(t23)
t23 = MaxPooling2D(2, 2)(t23)
t23 = Dropout(drop_size * 3)(t23)
# 8--------------------------------------------------------------------------------------
t31 = Conv2D(ks * 2, (3, 3), kernel_regularizer=reg, padding="same", activation=act)(t23)
t31 = BatchNormalization()(t31)
t3i2 = concatenate((t23, t31))
t32 = Conv2D(ks * 4, (3, 3), kernel_regularizer=reg, padding="same", activation=act)(t3i2)
t32 = BatchNormalization()(t32)
t3i3 = concatenate((t23, t31, t32))
t33 = Conv2D(ks * 8, (3, 3), kernel_regularizer=reg, padding="same", activation=act)(t3i3)
t33 = BatchNormalization()(t33)
t33 = MaxPooling2D(2, 2)(t33)
t33 = Dropout(drop_size * 3)(t33)
# 4------------------------------------------------------------------------------------
output = Flatten()(t33)
# output = Dropout(drop_size*4)
output = Dense(16 * ks, kernel_regularizer=reg, activation=act)(output)
output = BatchNormalization()(output)
output = Dropout(3 * drop_size)(output)
output = Dense(16 * ks, kernel_regularizer=reg, activation=act)(output)
output = BatchNormalization()(output)
output = Dropout(3 * drop_size)(output)
output = Dense(16 * ks, kernel_regularizer=reg, activation=act)(output)
output = BatchNormalization()(output)
output = Dropout(3 * drop_size)(output)

output = Dense(num_classes, activation='softmax')(output)
model2 = Model(i, output)

model2.compile(optimizer="adam",
               loss="categorical_crossentropy",
               metrics=["accuracy"])

checkpoint_filepath = './checkpoint/checkpoint'
model2.load_weights(checkpoint_filepath)

img_directory = './data/extractedContours_2/'

contours = tf.keras.utils.image_dataset_from_directory(
    img_directory,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode='grayscale',
    batch_size=128,
    image_size=input_shape[:-1],
    # shuffle=True,
    subset=None,
)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)


def normalize(arr):
    print(tf.math.reduce_min(arr))
    print(tf.math.reduce_max(arr))
    return (arr - tf.math.reduce_min(arr)) / (tf.math.reduce_max(arr) - tf.math.reduce_min(arr))


normalized_ds = contours.map(lambda x: (normalization_layer(x)))
image_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
