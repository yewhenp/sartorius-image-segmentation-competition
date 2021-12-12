from tensorflow import keras
from .unet import get_unet
from typing import Tuple, Dict

from ..constants import ConfigKeys as ck


class ModelNames:
    U_NET = "U-NET"


models_getter = {
    ModelNames.U_NET: get_unet
}


def get_model(cnf: Dict, input_shape: Tuple) -> keras.Model:
    return models_getter[cnf[ck.MODEL_NAME]](input_shape)