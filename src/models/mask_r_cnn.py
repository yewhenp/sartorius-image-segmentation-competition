from git import HEAD

from typing import Dict
import os

# START PROJECT IMPORTS
from ..constants import ROOT_DIR, ck, WIDTH, ROOT_DIR, HEIGHT
from .mrcnn.model import log
from .mrcnn.config import Config
from .mrcnn import utils
from .mrcnn import model as modellib
# END PROJECT_IMPORTS

class ShapesConfig(Config):
    def __init__(self, config: Dict):
        super().__init__()

        self.IMAGES_PER_GPU = config[ck.BATCH_SIZE]
        self.LEARNING_MOMENTUM = config[ck.OPTIMIZER_PARAMETERS].get("learning_momentum", self.LEARNING_MOMENTUM)
        self.LEARNING_RATE = config[ck.OPTIMIZER_PARAMETERS].get("learning_rate", self.LEARNING_RATE)


    NAME = "sartorius_mrcnn"
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1

    IMAGE_MIN_DIM = HEIGHT
    IMAGE_MAX_DIM = WIDTH

    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 200
    # STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


def get_mask_rcnn(input_data_shape, **kwargs):
    cnf = kwargs["config"]
    config = ShapesConfig(cnf)
    MODEL_DIR = os.path.join(cnf[ck.WEIGHTS_DIR], cnf[ck.EXPERIMENT_NAME])
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    init_with = cnf.get("mrcnn_init_with", "coco")
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    
    model.sgd_learning_rate = config.LEARNING_RATE
    
    return model

