from types import SimpleNamespace
import pandas as pd
from monai import transforms as mt
from common_config import basic_cfg

# Create a copy of the basic config
cfg = SimpleNamespace(**basic_cfg.__dict__)

# Modify config for custom tomogram selection
cfg.name = 'custom_train_config'
cfg.model = "mdl_1"
cfg.dataset = "ds_1"

# Custom data filtering
# Setup to use our own data filtering logic instead of using folds
cfg.custom_data_split = True

# Function to filter data for training and validation
def get_custom_data_split(df):
    # Training tomograms: TS_5_4, TS_69_2, TS_6_4, TS_6_6, TS_73_6, TS_86_3
    train_tomograms = ['TS_5_4', 'TS_69_2', 'TS_6_4', 'TS_6_6', 'TS_73_6', 'TS_86_3']
    # Validation tomogram: TS_99_9
    val_tomogram = 'TS_99_9'
    
    # Filter dataframes based on specified tomograms
    train_df = df[df['experiment'].isin(train_tomograms)].copy()
    val_df = df[df['experiment'] == val_tomogram].copy()
    
    # If val_df is empty, use a small portion of train_df for validation
    if len(val_df) == 0:
        print("Warning: Validation tomogram TS_99_9 not found in dataset. Using a portion of training data for validation.")
        val_df = train_df.sample(frac=0.1, random_state=42)
    
    print(f"Training data size: {len(train_df)}")
    print(f"Validation data size: {len(val_df)}")
    
    return train_df, val_df

# Set this function in the config
cfg.get_custom_data_split = get_custom_data_split

# Training parameters
cfg.batch_size = 8
cfg.batch_size_val = 8
cfg.epochs = 10
cfg.lr = 1e-3
cfg.optimizer = "Adam"
cfg.schedule = "cosine"
cfg.fold = 0  # Not actually used with our custom split, but kept for consistency

# Mixed precision settings
cfg.mixed_precision = False
cfg.bf16 = True 