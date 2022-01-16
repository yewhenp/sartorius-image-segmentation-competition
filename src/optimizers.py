from lib2to3.pgen2.token import OP
from typing import Dict
from tensorflow.keras.optimizers import Adam, SGD
from .constants import ck

OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

def get_optimizer(conf: Dict):
    return OPTIMIZERS[conf[ck.OPTIMIZER]](**conf[ck.OPTIMIZER_PARAMETERS])