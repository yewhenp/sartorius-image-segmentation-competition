from tensorflow import keras
from typing import Tuple, Dict

# START PROJECT IMPORTS
from ..constants import ConfigKeys as ck
from .unet import get_unet
from .unet_watershed import get_unet_watershed
from .mask_r_cnn import get_mask_rcnn
# END PROJECT_IMPORTS


class ModelNames:
    U_NET = "U-NET"
    U_NET_WATERSHED = "U-NET-WATERSHED"
    MASK_R_CNN = "MASK-R-CNN"


models_getter = {
    ModelNames.U_NET: get_unet,
    ModelNames.U_NET_WATERSHED: get_unet_watershed,
    ModelNames.MASK_R_CNN: get_mask_rcnn
}


def get_model(cnf: Dict, input_shape: Tuple) -> keras.Model:
    return models_getter[cnf[ck.MODEL_NAME]](input_shape, config=cnf)
