#!/usr/bin/env python3
import os
import glob
import csv
from pathlib import Path

def main():
    # Base directory for the CZII data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "czii_data")
    output_file = os.path.join(base_dir, "czii_metadata.csv")
    
    # Define protein classes we're working with
    proteins = [
        "apo-ferritin",
        "beta-amylase",
        "beta-galactosidase",
        "ribosome",
        "thyroglobulin",
        "virus-like-particle"
    ]
    
    # Create header for CSV
    header = ["tomo_name", "tomo", "lamella_file"]
    
    # Add mask columns for each protein
    for protein in proteins:
        header.append(f"{protein}_mask")
    
    # Add motl file path columns
    for protein in proteins:
        header.append(f"path_to_motl_clean_{protein}")
    
    # Get all experiment directories
    exp_dirs = sorted(glob.glob(os.path.join(data_dir, "TS_*")))
    
    # Create the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for exp_dir in exp_dirs:
            exp_name = os.path.basename(exp_dir)
            print(f"Processing {exp_name}")
            
            # Find tomogram file
            tomo_files = glob.glob(os.path.join(exp_dir, "*.rec"))
            if not tomo_files:
                print(f"Warning: No tomogram (.rec) file found in {exp_name}, skipping")
                continue
            tomo_file = tomo_files[0]
            
            # Find lamella mask file
            lamella_files = glob.glob(os.path.join(exp_dir, f"{exp_name}_lamellamask.em"))
            if not lamella_files:
                print(f"Warning: No lamella mask file found in {exp_name}, skipping")
                continue
            lamella_file = lamella_files[0]
            
            # Start building the row
            row = [exp_name, tomo_file, lamella_file]
            
            # Add mask files for each protein
            for protein in proteins:
                mask_file = os.path.join(exp_dir, "masks", f"{protein}_TM_cnnIF4_cnnIF8_cnnIF32_motl_{protein}.mrc")
                if os.path.exists(mask_file):
                    row.append(mask_file)
                else:
                    # If the specific mask doesn't exist, use the lamella mask as fallback
                    print(f"Warning: Mask for {protein} not found in {exp_name}, using lamella mask instead")
                    row.append(lamella_file)
            
            # Add motl files paths
            for protein in proteins:
                motl_file = os.path.join(exp_dir, "clean_motls", protein, f"TM_cnnIF4_cnnIF8_cnnIF32_motl_{protein}.csv")
                if os.path.exists(motl_file):
                    row.append(motl_file)
                else:
                    # If motl file doesn't exist, leave empty
                    print(f"Warning: Motl file for {protein} not found in {exp_name}")
                    row.append("")
            
            # Write the row to the CSV
            writer.writerow(row)
    
    print(f"Metadata file created at: {output_file}")

if __name__ == "__main__":
    main() 