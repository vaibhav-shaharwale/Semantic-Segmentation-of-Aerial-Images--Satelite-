from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
#from keras.layers.convolutional import Conv2D
from keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)     # Not in original unet network
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)     # Not in original unet network
    x = Activation('relu')(x)

    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2,2))(x)

    return x, p

def decoder_block(input, skip_feature, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_feature])
    x = conv_block(x, num_filters)

    return x


def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)       #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation ="softmax")(d4)

    model = Model(inputs, outputs, name="u_net")

    return model
