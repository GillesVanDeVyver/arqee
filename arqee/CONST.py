# file to define constants
import os
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

DOWNLOAD_LINKS = {
    "mobilenetv2_regional_quality":("gillesvdv/mobilenetv2_regional_quality","mobilenetv2_regional_quality.zip"),
    "sample_data":"https://api.github.com/repos/GillesVanDeVyver/us_cardiac_sample_data/contents/sample_data",
    "nnunet_hunt4_alax": ("gillesvdv/nnunet_hunt4_alax","nnunet_hunt4_alax.zip"),
    "nnunet_hunt4_a2c_a4c" : ("gillesvdv/nnunet_hunt4_a2c_a4c","nnunet_hunt4_a2c_a4c.zip")
}

DEFAULT_SLOPE_INTERCEPTS_PIXEL_BASED_METHODS = {
    "gcnr": np.array([0.03123222, 0.71816888]),
    "cnr": np.array([0.21145037, 0.48586693]),
    "cr": np.array([0.03724453, 1.04729782]),
    "intensity": np.array([5.11751306, 150.53709063])
}

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')