from types import SimpleNamespace
from custom_train_config import cfg

# Create a copy of our custom config
cfg = SimpleNamespace(**cfg.__dict__)

# Model specific settings
cfg.name = 'cfg_resnet34_custom'
cfg.model = "mdl_resnet"
cfg.backbone = "resnet34"
cfg.pool_head = "avg"
cfg.depth = 4
cfg.start_channel = 8

# Set path to Kaggle czii-cryo-et-object-identification dataset
# Make sure to update this path to where your dataset is located
cfg.data_folder = '/home/dmitry/vkr/czii-cryo-et-object-identification/train/static/ExperimentRuns/'

# Output directory
cfg.output_dir = "output"

# Export
basic_cfg = cfg 