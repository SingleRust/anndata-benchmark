import yaml
import subprocess
import json
import csv
import os
from pathlib import Path

def run_command(cmd, env=None):
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=current_env)
    if result.returncode != 0:
        print(f"Error running command {' '.join(cmd)}:")
        print(result.stderr)
        return None
    return result.stdout.strip()

def main():
    config_path = "benchmark_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    s3_config = config.get('s3', {})
    benchmark_env = os.environ.copy()
    if s3_config.get('enabled'):
        print(f"S3 enabled. Bucket: {s3_config.get('bucket')}")
        if s3_config.get('access_key_id'):
            benchmark_env['AWS_ACCESS_KEY_ID'] = s3_config['access_key_id']
        if s3_config.get('secret_access_key'):
            benchmark_env['AWS_SECRET_ACCESS_KEY'] = s3_config['secret_access_key']
        if s3_config.get('region'):
            benchmark_env['AWS_REGION'] = s3_config['region']
        if s3_config.get('endpoint'):
            benchmark_env['AWS_ENDPOINT_URL'] = s3_config['endpoint']

    # 1. Generate data
    print("Generating synthetic datasets...")
    subprocess.run(["pixi", "run", "generate-data"], check=True, env=benchmark_env)

    results = []
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    datasets = config.get('datasets', [])
    operations = config.get('operations', [])

    for ds in datasets:
        for fmt in ds.get('formats', []):
            dataset_path = data_dir / f"{ds['name']}.{fmt}"
            if not dataset_path.exists():
                print(f"Skipping missing dataset: {dataset_path}")
                continue

            for op in operations:
                print(f"Benchmarking local {ds['name']} ({fmt}) - {op['name']}...")
                params_json = json.dumps(op.get('params', {}))
                
                # Run Python
                print(f"  Running Python...")
                py_out = run_command([
                    "pixi", "run", "benchmark-python",
                    "--dataset", str(dataset_path),
                    "--operation", op['name'],
                    "--params", params_json
                ], env=benchmark_env)
                if py_out:
                    try:
                        res = json.loads(py_out)
                        res.update({'language': 'python', 'dataset': ds['name'], 'format': fmt, 'storage': 'local'})
                        results.append(res)
                    except json.JSONDecodeError:
                        print(f"    Failed to parse Python output: {py_out}")

                # Run Rust
                print(f"  Running Rust...")
                rs_out = run_command([
                    "pixi", "run", "benchmark-rust",
                    "--dataset", str(dataset_path),
                    "--operation", op['name'],
                    "--params", params_json
                ], env=benchmark_env)
                if rs_out:
                    json_res = None
                    for line in rs_out.splitlines():
                        try:
                            json_res = json.loads(line)
                        except:
                            continue
                    if json_res:
                        json_res.update({'language': 'rust', 'dataset': ds['name'], 'format': fmt, 'storage': 'local'})
                        results.append(json_res)

            # S3 Benchmarks (only if Zarr and enabled)
            if fmt == 'zarr' and s3_config.get('enabled'):
                s3_uri = f"s3://{s3_config['bucket']}/data/{ds['name']}.zarr"
                print(f"Benchmarking S3 {ds['name']} (zarr) - s3_zarr_read...")
                rs_s3_out = run_command([
                    "pixi", "run", "benchmark-rust",
                    "--dataset", s3_uri,
                    "--operation", "s3_zarr_read",
                    "--params", "{}"
                ], env=benchmark_env)
                if rs_s3_out:
                    json_res = None
                    for line in rs_s3_out.splitlines():
                        try:
                            json_res = json.loads(line)
                        except:
                            continue
                    if json_res:
                        json_res.update({'language': 'rust', 'dataset': ds['name'], 'format': fmt, 'storage': 's3'})
                        results.append(json_res)

    # Sort results for logical ordering
    results.sort(key=lambda x: (x.get('dataset', ''), x.get('format', ''), x.get('operation', ''), x.get('language', '')))

    # Save to CSV
    csv_path = results_dir / "benchmarks.csv"
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        # Logical order for columns
        primary_keys = ['dataset', 'format', 'operation', 'language', 'storage', 'duration_sec', 'initial_memory_mb', 'peak_memory_mb', 'final_memory_mb']
        remaining_keys = sorted(list(all_keys - set(primary_keys)))
        fieldnames = primary_keys + remaining_keys
        
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nResults saved to {csv_path}")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()
