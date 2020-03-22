import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCH = 'FConv'
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.FACTORY = 'objet'
_C.DATASET.PATH_TO_JSON = "./resources/ps_meta.json"
_C.DATASET.BATCH_SIZE = 8
_C.DATASET.NUM_WORKERS = 1


# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.001


# ---------------------------------------------------------------------------- #
# Logger
# ---------------------------------------------------------------------------- #
_C.LOGGER = CN()
_C.LOGGER.TB_PERIOD = 10
_C.LOGGER.TB_OUTPUT_DIR = 'logs/tensorboard/'



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
