import argparse
import os
import sys

# Add the repository paths
sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")
sys.path.append("metrics")

from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="cfg_resnet34_custom", help="config filename")
    parser.add_argument("--fold", type=int, default=0, help="fold")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="pretrained weights")
    parser.add_argument("--bf16", type=lambda x: (str(x).lower() == 'true'), default=None, help="use bf16")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    
    parser_args, _ = parser.parse_known_args()
    
    print(f"Training with config: {parser_args.config}")
    
    # Import config
    config_module = __import__(parser_args.config)
    cfg = config_module.basic_cfg
    
    # Override settings from command line if provided
    if parser_args.fold is not None:
        cfg.fold = parser_args.fold
    if parser_args.pretrained_weights is not None:
        cfg.pretrained_weights = parser_args.pretrained_weights
    if parser_args.bf16 is not None:
        cfg.bf16 = parser_args.bf16
        
    # Use CPU if specified
    if parser_args.cpu:
        cfg.distributed = False
        
    # Create output directory
    os.makedirs(f"{cfg.output_dir}", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/fold{cfg.fold}", exist_ok=True)
    
    # Train model
    train(cfg) 