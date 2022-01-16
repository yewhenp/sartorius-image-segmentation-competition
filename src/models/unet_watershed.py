from skimage.segmentation import watershed
from skimage.measure import label
import numpy as np
from typing import Dict

# START PROJECT IMPORTS
from .unet import get_unet
# END PROJECT_IMPORTS


def calc_watershed_transform(data, threshold=0.5):
    data = 255 * data
    pix_threshold = 255 * threshold

    mask = data[:, :, 0]
    energy = np.mean(data, axis=2)
    energy[energy < pix_threshold] = 0

    # energy = energy.astype("int")
    mask[mask < pix_threshold] = 0
    mask[mask > 0] = 1

    mrkr = label(energy)
    return watershed(-energy, mrkr, mask=mask, watershed_line=True)


class UnetWatershed:
    def __init__(self, input_data_shape):
        self.model = get_unet(input_data_shape, 6)

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x, epochs, validation_data, shuffle, callbacks):
        self.model.fit(x=x, epochs=epochs, validation_data=validation_data, shuffle=shuffle, callbacks=callbacks)

    def predict(self, data):
        prediction = self.model.predict(data)
        rez = [calc_watershed_transform(prediction[ind]) for ind in range(prediction.shape[0])]
        return np.asarray(rez)


def get_unet_watershed(input_data_shape, **kwargs):
    return UnetWatershed(input_data_shape)
