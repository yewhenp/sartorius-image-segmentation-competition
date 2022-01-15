from tensorflow import keras
from typing import Tuple, Dict

# START PROJECT IMPORTS
from ..constants import ConfigKeys as ck
from .unet import get_unet
from .unet_watershed import get_unet_watershed
# END PROJECT_IMPORTS


class ModelNames:
    U_NET = "U-NET"
    U_NET_WATERSHED = "U-NET-WATERSHED"


models_getter = {
    ModelNames.U_NET: get_unet,
    ModelNames.U_NET_WATERSHED: get_unet_watershed,
}


def get_model(cnf: Dict, input_shape: Tuple) -> keras.Model:
    return models_getter[cnf[ck.MODEL_NAME]](input_shape)
