from tensorflow import keras
from typing import Tuple, Dict

# START PROJECT IMPORTS
from ..constants import ConfigKeys as ck
from .unet import get_unet
# END PROJECT_IMPORTS


class ModelNames:
    U_NET = "U-NET"


models_getter = {
    ModelNames.U_NET: get_unet
}


def get_model(cnf: Dict, input_shape: Tuple) -> keras.Model:
    return models_getter[cnf[ck.MODEL_NAME]](input_shape)