import os
DATA_ROOT_PATH = os.path.abspath(os.path.join(__file__, "../data/"))
Standard_DATA_PATH = os.path.abspath(os.path.join(__file__, "../data/Standard Bern Barcelona/"))
FEATURE_PATH = os.path.abspath(os.path.join(__file__, "../data/new_feature/"))
CHECKPOINT_PATH = os.path.abspath(os.path.join(__file__, "../checkpoints"))
USE_GPU = False
threshold = 0.5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
