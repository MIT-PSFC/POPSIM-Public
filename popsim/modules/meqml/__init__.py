import os
from importlib.resources import files

MODULE_DIR = files("popsim.modules.meqml")
TCV_CHECKPOINTS = os.path.join(MODULE_DIR, "checkpoints/TCV")
