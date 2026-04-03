import anndata as ad
import numpy as np
import scipy.sparse as sp
import yaml
import os
import subprocess
from pathlib import Path
import boto3
from botocore.config import Config

def upload_to_s3(local_path, s3_config, remote_path):
    print(f"  Uploading {local_path} to S3 bucket {s3_config['bucket']}...")
    s3 = boto3.client(
        's3',
        aws_access_key_id=s3_config.get('access_key_id'),
        aws_secret_access_key=s3_config.get('secret_access_key'),
        region_name=s3_config.get('region', 'us-east-1'),
        endpoint_url=s3_config.get('endpoint')
    )

    if local_path.is_dir():
        # Zarr is a directory
        for file in local_path.rglob('*'):
            if file.is_file():
                rel_path = file.relative_to(local_path.parent)
                s3.upload_file(str(file), s3_config['bucket'], str(rel_path))
    else:
        # H5AD is a file
        s3.upload_file(str(local_path), s3_config['bucket'], remote_path)

def generate_dataset(dataset_config, s3_config=None):
    name = dataset_config['name']
    n_obs = dataset_config['n_obs']
    n_vars = dataset_config['n_vars']
    sparsity = dataset_config.get('sparsity', 0.05)
    formats = dataset_config.get('formats', ['h5ad'])

    print(f"Generating dataset: {name} ({n_obs}x{n_vars}, sparsity={sparsity})")
    
    # Generate sparse matrix
    X = sp.random(n_obs, n_vars, density=sparsity, format='csr')
    
    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        obs={'batch': np.random.choice(['A', 'B'], size=n_obs)},
        var={'gene_names': [f"gene_{i}" for i in range(n_vars)]}
    )
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for fmt in formats:
        out_path = data_dir / f"{name}.{fmt}"
        if fmt == 'h5ad':
            print(f"  Saving to {out_path}...")
            adata.write_h5ad(out_path)
        elif fmt == 'zarr':
            print(f"  Saving to {out_path}...")
            # Forcing Zarr V3 where possible, though anndata uses V2 by default
            adata.write_zarr(out_path)
        
        if s3_config and s3_config.get('enabled'):
            upload_to_s3(out_path, s3_config, f"data/{name}.{fmt}")

def main():
    config_path = "benchmark_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    s3_config = config.get('s3')

    for dataset_config in config.get('datasets', []):
        generate_dataset(dataset_config, s3_config)

if __name__ == "__main__":
    main()
