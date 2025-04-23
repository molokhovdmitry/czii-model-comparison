#!/usr/bin/env python3
import os
import glob
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Generate masks for all motl files in all experiments")
    parser.add_argument("-r", "--radius", type=int, default=8, 
                        help="Sphere radius for mask generation")
    parser.add_argument("-s", "--shape", type=str, default="184,630,630", 
                        help="Shape of the output mask (z,y,x)")
    parser.add_argument("-v", "--value", type=int, default=1, 
                        help="Value to assign to voxels in the spheres")
    parser.add_argument("-d", "--data_dir", type=str, 
                        default=os.path.join(os.environ.get("DATA_DIR", "./czii_data")),
                        help="Path to the czii_data directory")
    args = parser.parse_args()
    
    # Parse shape from command line
    shape = [int(x) for x in args.shape.split(",")]
    if len(shape) != 3:
        raise ValueError("Shape must have exactly 3 dimensions (z,y,x)")
    
    # Get path to motl2sph_mask.py script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motl2sph_mask.py")
    
    # Get all experiment directories
    exp_dirs = glob.glob(os.path.join(args.data_dir, "TS_*"))
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"Processing experiment: {exp_name}")
        
        # Find clean_motls directory
        clean_motls_dir = os.path.join(exp_dir, "clean_motls")
        if not os.path.exists(clean_motls_dir):
            print(f"  Warning: clean_motls directory not found in {exp_name}, skipping")
            continue
        
        # Create masks directory if it doesn't exist
        masks_dir = os.path.join(exp_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Get all protein directories
        protein_dirs = [d for d in os.listdir(clean_motls_dir) 
                       if os.path.isdir(os.path.join(clean_motls_dir, d))]
        
        for protein in protein_dirs:
            protein_dir = os.path.join(clean_motls_dir, protein)
            print(f"  Processing protein: {protein}")
            
            # Find all motl files (both .csv and .em formats are supported)
            motl_files = []
            motl_files.extend(glob.glob(os.path.join(protein_dir, "*.csv")))
            motl_files.extend(glob.glob(os.path.join(protein_dir, "*.em")))
            motl_files.extend(glob.glob(os.path.join(protein_dir, "*.txt")))
            
            for motl_file in motl_files:
                motl_filename = os.path.basename(motl_file)
                base_name = os.path.splitext(motl_filename)[0]
                output_file = os.path.join(masks_dir, f"{protein}_{base_name}.mrc")
                
                # Skip if output file already exists
                if os.path.exists(output_file):
                    print(f"    Skipping {motl_filename}, mask already exists")
                    continue
                
                print(f"    Creating mask for {motl_filename}")
                cmd = [
                    "python", script_path,
                    "-r", str(args.radius),
                    "-motl", motl_file,
                    "-o", output_file,
                    "-shape", str(shape[2]), str(shape[1]), str(shape[0]),
                    "-value", str(args.value)
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    print(f"    Created mask: {output_file}")
                except subprocess.CalledProcessError as e:
                    print(f"    Error creating mask for {motl_filename}: {e}")

if __name__ == "__main__":
    main() 