from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict


def conv_block(input_data, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(input_data, num_filters, dropout_prob=0.5):
    x = conv_block(input_data, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    p = layers.Dropout(dropout_prob)(p)
    return x, p


def decoder_block(input_data, skip_features, num_filters, dropout_prob=0.5):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_data)
    x = layers.Concatenate()([x, skip_features])
    x = layers.Dropout(dropout_prob)(x)
    x = conv_block(x, num_filters)
    return x


def get_unet(input_data_shape, output_channels=1, **kwargs) -> keras.Model:
    print("output_channels = ", output_channels)

    inputs = keras.Input(input_data_shape)

    # TODO: regularization
    # s1, p1 = encoder_block(inputs, 16)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    #
    b1 = conv_block(p4, 1024)
    # b1 = conv_block(p1, 32)
    #
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    # d4 = decoder_block(b1, s1, 16)

    outputs = layers.Conv2D(output_channels, 1, padding="same", activation="sigmoid")(d4)

    model = keras.Model(inputs, outputs, name="U-Net")
    model.summary()
    return model
