from tensorflow import keras
from .unet import get_unet
from typing import Tuple

class ModelNames:
    U_NET = "U-NET"

models_getter = {
    ModelNames.U_NET: get_unet
}

def get_model(name: str, input_shape: Tuple) -> keras.Model:
    return models_getter[name](input_shape)