import warnings
import numpy as np
warnings.filterwarnings("ignore")

from data_builder import build_tags
from utils import utils

config = utils.get_config("exp318")
config["mode"] = "training"

print("Building tags...")
tags_train, tags_val = build_tags.get_tags(config)
print(f"# Training Tiles:   {len(np.unique(tags_train.files))}")
print(f"# Validation Tiles: {len(np.unique(tags_val.files))}")
print("Done.")