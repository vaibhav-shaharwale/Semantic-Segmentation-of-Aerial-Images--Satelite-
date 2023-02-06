import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def down_sampling_block(inputs, filters, kernel_size=(3,3), padding='same', strides=1):
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x

def up_sampling_block(inputs, filters, kernel_size=(3,3), padding='same', strides=1):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape, num_classes=1):
    inputs = keras.layers.Input(shape=input_shape)
    filters = 64
    block1 = down_sampling_block(inputs, filters)
    filters *= 2
    block2 = down_sampling_block(block1, filters)
    filters *= 2
    block3 = down_sampling_block(block2, filters)
    filters *= 2
    block4 = down_sampling_block(block3, filters)
    filters *= 2
    block5 = down_sampling_block(block4, filters)
    block6 = up_sampling_block(block5, filters // 2)
    block6 = layers.concatenate([block6, block3], axis=-1)      # concatenating on last axis i.e. channel axis
    block7 = up_sampling_block(block6, filters // 4)
    block7 = layers.concatenate([block7, block2], axis=-1)
    block8 = up_sampling_block(block7, filters // 8)
    block8 = layers.concatenate([block8, block1], axis=-1)
    block9 = up_sampling_block(block8, filters // 16)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(block9)
    
    model = keras.Model(inputs, outputs, name="u_net")

    return model