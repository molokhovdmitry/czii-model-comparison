from types import SimpleNamespace
from custom_train_config import cfg
import numpy as np

# Create a copy of our custom config
cfg = SimpleNamespace(**cfg.__dict__)

# Model specific settings
cfg.name = 'cfg_effnetb3_custom'
cfg.model = "mdl_efficientnet"
cfg.backbone = 'efficientnet-b3'
cfg.backbone_args = dict(spatial_dims=3,    
                         in_channels=cfg.in_channels,
                         out_channels=cfg.n_classes,
                         backbone=cfg.backbone,
                         pretrained=cfg.pretrained)
cfg.class_weights = np.array([64,64,64,64,64,64,1])
cfg.lvl_weights = np.array([0,0,0,1])

# Set path to Kaggle czii-cryo-et-object-identification dataset
# Make sure to update this path to where your dataset is located
cfg.data_folder = '/home/dmitry/vkr/czii-cryo-et-object-identification/train/static/ExperimentRuns/'

# Output directory
cfg.output_dir = "output"

# Export
basic_cfg = cfg 