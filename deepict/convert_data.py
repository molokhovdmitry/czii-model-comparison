import os
import numpy as np
import zarr
import pandas as pd
from pathlib import Path
import shutil
import emfile
import mrcfile

def convert_tomogram(zarr_path, output_path, voxel_size):
    """Convert zarr tomogram to MRC/REC format."""
    print(f"Converting tomogram from {zarr_path} to {output_path}")
    
    # Read zarr data - use the full resolution data (array '0')
    z = zarr.open(zarr_path, mode='r')
    data = np.array(z['0'])
    
    print(f"Original data shape: {data.shape}")
    print(f"Original data range: min={np.min(data):.8f}, max={np.max(data):.8f}, mean={np.mean(data):.8f}")
    
    # Normalize data to match typical cryo-EM values
    # First normalize to zero mean and unit variance
    data = (data - np.mean(data)) / (np.std(data) + 1e-10)
    
    # Then scale to match typical cryo-EM range (-300 to 200)
    data = data * 100  # Scale factor to get similar range to TS_026.rec
    
    print(f"Data range after normalization: min={np.min(data):.2f}, max={np.max(data):.2f}, mean={np.mean(data):.2f}")
    
    # Create MRC file with proper metadata
    with mrcfile.new(output_path, overwrite=True) as mrc:
        # Set the data (must be float32)
        mrc.set_data(data.astype(np.float32))
        
        # Set header information
        nx, ny, nz = data.shape
        mrc.header.nx = nx
        mrc.header.ny = ny
        mrc.header.nz = nz
        
        # Set voxel size (in Angstroms)
        # Cell dimensions = number of voxels * voxel size
        mrc.header.cella.x = nx * voxel_size[0]
        mrc.header.cella.y = ny * voxel_size[1]
        mrc.header.cella.z = nz * voxel_size[2]
        
        # Set cell angles (typically 90 degrees)
        mrc.header.cellb.alpha = 90.0
        mrc.header.cellb.beta = 90.0
        mrc.header.cellb.gamma = 90.0
        
        # Set map type (2 = image, 0 = volume)
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3
        
        # Update min/max/mean values
        mrc.update_header_stats()
    
    print(f"Successfully converted tomogram to {output_path}")
    print(f"Output file dimensions: {nx} x {ny} x {nz}")
    print(f"Output file voxel size: {voxel_size[0]:.2f} x {voxel_size[1]:.2f} x {voxel_size[2]:.2f} Angstroms")

def convert_particle_coordinates(json_path, output_path, voxel_size, data_shape):
    """Convert JSON particle coordinates to CSV format with proper scaling."""
    import json
    
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract coordinates from points and scale them
    coordinates = []
    for point in data['points']:
        # Get original coordinates
        x = point['location']['x']
        y = point['location']['y']
        z = point['location']['z']
        
        # Scale coordinates according to voxel size
        # Note: The coordinates in the JSON are in physical units (Angstroms)
        # We need to convert them to voxel indices
        x_idx = int(round(x / voxel_size[2]))  # x is the last dimension
        y_idx = int(round(y / voxel_size[1]))  # y is the middle dimension
        z_idx = int(round(z / voxel_size[0]))  # z is the first dimension
        
        # Ensure coordinates are within bounds
        x_idx = max(0, min(x_idx, data_shape[2] - 1))
        y_idx = max(0, min(y_idx, data_shape[1] - 1))
        z_idx = max(0, min(z_idx, data_shape[0] - 1))
        
        # motl2sph_mask.py expects coordinates in [x, y, z] order
        # The function extract_coordinates reads them as [[2, 1, 0]] which means z,y,x
        # But paste_sphere_in_dataset treats them as cx,cy,cz
        # So we need to use [x, y, z] order here
        coordinates.append([x_idx, y_idx, z_idx])
    
    # Create DataFrame and save as CSV without header
    df = pd.DataFrame(coordinates)
    df.to_csv(output_path, index=False, header=False)
    print(f"Converted particle coordinates to {output_path}")
    print(f"Number of particles: {len(coordinates)}")
    print(f"Coordinate sample (x,y,z): {coordinates[0] if coordinates else 'No particles'}")

def create_lamella_mask(output_path, shape):
    """Create a lamella mask file filled with ones."""
    print(f"Creating lamella mask at {output_path}")
    
    # Create a numpy array filled with ones
    data = np.ones(shape, dtype=np.int32)
    
    # Write the mask to EM file
    emfile.write(output_path, data)
    print(f"Successfully created lamella mask at {output_path}")

def process_experiment(experiment, base_dir, voxel_spacing_dir, data_root_path):
    """Process a single experiment."""
    print(f"\n=== Processing experiment: {experiment} ===\n")
    
    # Extract voxel size from directory name
    voxel_size = float(voxel_spacing_dir.replace('VoxelSpacing', '').replace('.000', ''))
    print(f"Using voxel size: {voxel_size} Angstroms")
    
    # Only check in the train directory
    tomogram_path = data_root_path / 'train' / 'static' / 'ExperimentRuns' / experiment / voxel_spacing_dir / 'wbp.zarr'
    
    # Skip if not found
    if not tomogram_path.exists():
        print(f"Warning: Tomogram path for {experiment} not found in train directory. Skipping experiment.")
        return
    
    print(f"Found tomogram for {experiment} in train directory")
    
    # Set up output path
    output_tomogram = base_dir / experiment / 'eman_filtered_raw_4b.rec'
    output_tomogram.parent.mkdir(parents=True, exist_ok=True)
    
    # Get data shape from zarr file
    z = zarr.open(tomogram_path, mode='r')
    data_shape = z['0'].shape
    
    # Create uniform voxel size array (same size in all dimensions)
    voxel_size_array = np.array([voxel_size, voxel_size, voxel_size])
    
    # Convert tomogram
    convert_tomogram(tomogram_path, output_tomogram, voxel_size_array)
    
    # Create lamella mask
    lamella_mask_path = base_dir / experiment / f'{experiment}_lamellamask.em'
    create_lamella_mask(lamella_mask_path, data_shape)
    
    # Convert particle coordinates
    json_files = [
        'beta-amylase.json',
        'beta-galactosidase.json',
        'ribosome.json',
        'apo-ferritin.json',
        'thyroglobulin.json',
        'virus-like-particle.json'
    ]
    
    for json_file in json_files:
        particle_name = json_file.split('.')[0]
        
        # Create output directory for this particle
        output_dir = base_dir / experiment / 'clean_motls' / particle_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only check in train directory
        json_path = data_root_path / 'train' / 'overlay' / 'ExperimentRuns' / experiment / 'Picks' / json_file
        output_path = output_dir / f'TM_cnnIF4_cnnIF8_cnnIF32_motl_{particle_name}.csv'
        
        if json_path.exists():
            convert_particle_coordinates(json_path, output_path, voxel_size_array, data_shape)
        else:
            print(f"Warning: Particle picks for {particle_name} not found for experiment {experiment}")

def main():
    # Base directories
    base_dir = Path('czii_data')
    data_root_path = Path('/home/dmitry/vkr/czii-model-comparison/czii-cryo-et-object-identification')
    voxel_spacing_dir = 'VoxelSpacing10.000'  # Assuming same for all experiments
    
    # Remove existing data directory if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)
    
    # Create base directory structure
    base_dir.mkdir(parents=True)
    
    # Find all experiments in train directory only
    train_exp_path = data_root_path / 'train' / 'static' / 'ExperimentRuns'
    if train_exp_path.exists():
        train_experiments = [p.name for p in train_exp_path.iterdir() if p.is_dir()]
        train_experiments.sort()  # Sort for consistent processing order
        print(f"Found {len(train_experiments)} experiments in train directory: {', '.join(train_experiments)}")
    else:
        train_experiments = []
        print("Train experiments directory not found!")
    
    # Process each experiment from train directory only
    print(f"\nProcessing {len(train_experiments)} experiments from train directory\n")
    for experiment in train_experiments:
        process_experiment(experiment, base_dir, voxel_spacing_dir, data_root_path)

if __name__ == '__main__':
    main() 