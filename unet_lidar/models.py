from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from keras_unet.models import custom_unet

from tensorflow.keras import backend as K


def conv_f(x, maps):
    conv = Conv2D(maps, (3, 3), activation='relu', padding='same')(x)
    # drop = Dropout(0.5)(conv)
    return conv


def pool_f(x):
    pool = MaxPooling2D(pool_size=(2, 2))(x)
    return pool


def up_f(x_up, x):
    up = concatenate([UpSampling2D(size=(2, 2))(x_up), x], axis=3)
    return up


def encoder_f(x, maps):
    conv = conv_f(x, maps)
    conv = conv_f(conv, maps)
    pool = pool_f(conv)
    return pool, conv


def decoder_f(x, x_bypass, maps):
    up = up_f(x, x_bypass)
    conv = conv_f(up, maps)
    # conv = conv_f(conv, maps)
    return conv


def unet4(scale=32, is_linear=False):
    inputs = Input((None, None, 3))

    e1, e1_bypass = encoder_f(inputs, scale)
    e2, e2_bypass = encoder_f(e1, scale * 2)
    e3, e3_bypass = encoder_f(e2, scale * 4)
    e4, e4_bypass = encoder_f(e3, scale * 8)
    _, e5 = encoder_f(e4, scale * 16)

    d1 = decoder_f(e5, e4_bypass, scale * 8)
    d2 = decoder_f(d1, e3_bypass, scale * 4)
    d3 = decoder_f(d2, e2_bypass, scale * 2)
    d4 = decoder_f(d3, e1_bypass, scale)

    if is_linear:
        outputs = Conv2D(1, (1, 1), activation='linear')(d4)
    else:
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def unet_model(is_train=False, type='unet4', is_linear=False):
    if type == 'unet4':
        model = unet4(scale=32, is_linear=is_linear)
    elif type == 'custom_unet':
        model = custom_unet(
            input_shape=(None, None, 3),
            use_batch_norm=False,
            num_classes=1,
            filters=32,
            dropout=0.2,
            output_activation='linear' if is_linear else 'sigmoid'
        )
    else:
        raise Exception('Unknown model type')

    if is_train:
        if is_linear:
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
        else:
            model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error')

        model.summary()

    return model

