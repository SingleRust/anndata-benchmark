use anyhow::{Result, Context};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use sysinfo::{System, Pid};

// AnnData imports
use anndata::{AnnData, ArrayData, data::SelectInfoElem, Backend, AnnDataOp, ArrayElemOp};
use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use anndata_memory::{load_h5ad, convert_to_in_memory};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    dataset: String,

    #[arg(short, long)]
    operation: String,

    #[arg(short, long, default_value = "{}")]
    params: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Params {
    #[serde(default = "default_row_fraction")]
    row_fraction: f64,
}

fn default_row_fraction() -> f64 { 0.1 }

#[derive(Serialize)]
struct BenchmarkResult {
    operation: String,
    duration_sec: f64,
    initial_memory_mb: f64,
    peak_memory_mb: f64,
    final_memory_mb: f64,
}

fn get_current_rss_mb(sys: &mut System) -> f64 {
    let pid = Pid::from(std::process::id() as usize);
    sys.refresh_process(pid);
    if let Some(process) = sys.process(pid) {
        process.memory() as f64 / 1024.0 / 1024.0
    } else {
        0.0
    }
}

fn monitor_memory(stop_signal: Arc<AtomicBool>, peak_mem: Arc<parking_lot::Mutex<f64>>) {
    let mut sys = System::new();
    let pid = Pid::from(std::process::id() as usize);
    while !stop_signal.load(Ordering::Relaxed) {
        sys.refresh_process(pid);
        if let Some(process) = sys.process(pid) {
            let current_mem = process.memory() as f64 / 1024.0 / 1024.0;
            let mut peak = peak_mem.lock();
            if current_mem > *peak {
                *peak = current_mem;
            }
        }
        thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let params: Params = serde_json::from_str(&cli.params).unwrap_or(Params { row_fraction: 0.1 });

    let mut sys = System::new();
    let initial_mem = get_current_rss_mb(&mut sys);
    let stop_signal = Arc::new(AtomicBool::new(false));
    let peak_mem = Arc::new(parking_lot::Mutex::new(initial_mem));

    // Start memory monitor thread
    let monitor_stop = Arc::clone(&stop_signal);
    let monitor_peak = Arc::clone(&peak_mem);
    let monitor_handle = thread::spawn(move || {
        monitor_memory(monitor_stop, monitor_peak);
    });

    let start = Instant::now();

    match cli.operation.as_str() {
        "backed_read_full" => {
            if cli.dataset.ends_with(".h5ad") {
                let adata = AnnData::<H5>::open(H5::open_rw(&cli.dataset)?)?;
                let _x: ArrayData = adata.x().get()?.context("X is empty")?;
            } else {
                let adata = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
                let _x: ArrayData = adata.x().get()?.context("X is empty")?;
            }
        }
        "backed_subset_rows" => {
            let n_obs;
            let n_rows;
            if cli.dataset.ends_with(".h5ad") {
                let adata = AnnData::<H5>::open(H5::open_rw(&cli.dataset)?)?;
                n_obs = adata.n_obs();
                n_rows = (n_obs as f64 * params.row_fraction) as usize;
                let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
                let _x: ArrayData = adata.x().slice(&selection)?.context("X subset is empty")?;
            } else {
                let adata = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
                n_obs = adata.n_obs();
                n_rows = (n_obs as f64 * params.row_fraction) as usize;
                let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
                let _x: ArrayData = adata.x().slice(&selection)?.context("X subset is empty")?;
            }
        }
        "memory_load" => {
            if cli.dataset.ends_with(".h5ad") {
                let _adata = load_h5ad(&cli.dataset)?;
            } else {
                let backed = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
                let _imanndata = convert_to_in_memory(backed)?;
            }
        }
        "in_memory_subset" => {
            let adata;
            if cli.dataset.ends_with(".h5ad") {
                adata = load_h5ad(&cli.dataset)?;
            } else {
                let backed = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
                adata = convert_to_in_memory(backed)?;
            }
            let n_obs = adata.n_obs();
            let n_rows = (n_obs as f64 * params.row_fraction) as usize;
            let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
            let selection_ref: Vec<&SelectInfoElem> = selection.iter().collect();
            let _subset = adata.subset(&selection_ref)?;
        }
        "in_memory_subset_inplace" => {
            let mut adata;
            if cli.dataset.ends_with(".h5ad") {
                adata = load_h5ad(&cli.dataset)?;
            } else {
                let backed = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
                adata = convert_to_in_memory(backed)?;
            }
            let n_obs = adata.n_obs();
            let n_rows = (n_obs as f64 * params.row_fraction) as usize;
            let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
            let selection_ref: Vec<&SelectInfoElem> = selection.iter().collect();
            adata.subset_inplace(&selection_ref)?;
        }
        "convert_to_memory" => {
            if cli.dataset.ends_with(".h5ad") {
                let backed = AnnData::<H5>::open(H5::open_rw(&cli.dataset)?)?;
                let _imanndata = convert_to_in_memory(backed)?;
            } else {
                let backed = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
                let _imanndata = convert_to_in_memory(backed)?;
            }
        }
        "s3_zarr_read" => {
            let adata = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
            let _x: ArrayData = adata.x().get()?.context("X is empty")?;
        }
        "s3_zarr_subset" => {
            let adata = AnnData::<Zarr>::open(Zarr::open_rw(&cli.dataset)?)?;
            let n_obs = adata.n_obs();
            let n_rows = (n_obs as f64 * params.row_fraction) as usize;
            let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
            let _x: ArrayData = adata.x().slice(&selection)?.context("X subset is empty")?;
        }
        _ => anyhow::bail!("Unknown operation: {}", cli.operation),
    }

    let duration = start.elapsed().as_secs_f64();
    stop_signal.store(true, Ordering::Relaxed);
    monitor_handle.join().unwrap();

    let final_mem = get_current_rss_mb(&mut sys);
    {
        let mut peak = peak_mem.lock();
        if final_mem > *peak {
            *peak = final_mem;
        }
    }

    let result = BenchmarkResult {
        operation: cli.operation.clone(),
        duration_sec: duration,
        initial_memory_mb: initial_mem,
        peak_memory_mb: *peak_mem.lock(),
        final_memory_mb: final_mem,
    };

    println!("{}", serde_json::to_string(&result)?);

    Ok(())
}
