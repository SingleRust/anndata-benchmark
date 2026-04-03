import anndata as ad
import numpy as np
import time
import psutil
import os
import argparse
import json
import gc

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def run_benchmark(dataset_path, operation, params=None):
    initial_mem = get_memory_usage()
    
    # We want to measure the duration of the core operation
    duration = 0
    
    if operation == "backed_read_full":
        adata = ad.read_h5ad(dataset_path) if dataset_path.endswith('.h5ad') else ad.read_zarr(dataset_path)
        start_time = time.time()
        _ = adata.X[:]
        duration = time.time() - start_time
        
    elif operation == "backed_subset_rows":
        # Ensure we use backed mode for Python if we want to test I/O
        adata = ad.read_h5ad(dataset_path, backed='r') if dataset_path.endswith('.h5ad') else ad.read_zarr(dataset_path)
        row_fraction = params.get('row_fraction', 0.1)
        n_rows = int(adata.n_obs * row_fraction)
        start_time = time.time()
        _ = adata.X[:n_rows, :]
        duration = time.time() - start_time
        
    elif operation == "memory_load":
        start_time = time.time()
        adata = ad.read_h5ad(dataset_path) if dataset_path.endswith('.h5ad') else ad.read_zarr(dataset_path)
        # Ensure data is actually loaded
        _ = adata.X
        duration = time.time() - start_time
        
    elif operation in ["in_memory_subset", "in_memory_subset_inplace"]:
        # Pre-load into memory (setup not timed)
        adata = ad.read_h5ad(dataset_path) if dataset_path.endswith('.h5ad') else ad.read_zarr(dataset_path)
        row_fraction = params.get('row_fraction', 0.1)
        n_rows = int(adata.n_obs * row_fraction)
        
        start_time = time.time()
        if operation == "in_memory_subset":
            _ = adata[:n_rows, :].copy()
        else:
            # Simulate true in-place memory reduction:
            # Python must allocate a new copy, overwrite the reference, and force GC
            adata = adata[:n_rows, :].copy()
            gc.collect()
        duration = time.time() - start_time
    
    peak_mem = get_memory_usage()
    
    return {
        "operation": operation,
        "duration_sec": duration,
        "initial_memory_mb": initial_mem,
        "peak_memory_mb": peak_mem,
        "final_memory_mb": get_memory_usage()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--params", type=str, default="{}")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = run_benchmark(args.dataset, args.operation, params)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
