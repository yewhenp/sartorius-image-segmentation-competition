from tensorflow import keras
from tensorflow.keras import layers


def conv_block(input_data, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(input_data, num_filters):
    x = conv_block(input_data, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input_data, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_data)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def get_model(input_data_shape) -> keras.Model:
    inputs = keras.Input(input_data_shape)

    s1, p1 = encoder_block(inputs, 16)
    # s2, p2 = encoder_block(p1, 128)
    # s3, p3 = encoder_block(p2, 256)
    # s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p1, 16)

    # d1 = decoder_block(b1, s4, 512)
    # d2 = decoder_block(d1, s3, 256)
    # d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(b1, s1, 16)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = keras.Model(inputs, outputs, name="U-Net")
    return model
