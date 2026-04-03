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
use anndata_memory::{convert_to_in_memory};

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

/// Force evaluation of the data to ensure actual I/O and processing
fn force_eval(data: ArrayData) {
    // Just accessing the shape or a value is often enough for Rust's owned structures,
    // but we'll do a simple operation to be sure the CPU sees the data.
    match data {
        ArrayData::Array(a) => { let _ = a.shape(); }
        _ => {} // sparse structures in anndata-rs are already materialized on get/slice
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
        thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn run_with_backend<B: Backend>(cli: &Cli, params: &Params) -> Result<f64> {
    let duration;
    match cli.operation.as_str() {
        "backed_read_full" => {
            let adata = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let start = Instant::now();
            let x: ArrayData = adata.x().get()?.context("X is empty")?;
            force_eval(x);
            duration = start.elapsed();
        }
        "backed_subset_rows" => {
            let adata = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let n_rows = (adata.n_obs() as f64 * params.row_fraction) as usize;
            let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
            let start = Instant::now();
            let x: ArrayData = adata.x().slice(&selection)?.context("X subset is empty")?;
            force_eval(x);
            duration = start.elapsed();
        }
        "memory_load" => {
            let backed = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let start = Instant::now();
            let adata = convert_to_in_memory(backed)?;
            // IMArrayElement::get_data() returns Result<ArrayData>
            let x = adata.x().get_data()?;
            force_eval(x);
            duration = start.elapsed();
        }
        "in_memory_subset" | "in_memory_subset_inplace" => {
            let backed = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let mut adata = convert_to_in_memory(backed)?;
            let n_rows = (adata.n_obs() as f64 * params.row_fraction) as usize;
            let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
            let selection_ref: Vec<&SelectInfoElem> = selection.iter().collect();
            
            let start = Instant::now();
            if cli.operation == "in_memory_subset" {
                let subset = adata.subset(&selection_ref)?;
                let x = subset.x().get_data()?;
                force_eval(x);
            } else {
                adata.subset_inplace(&selection_ref)?;
                let x = adata.x().get_data()?;
                force_eval(x);
            }
            duration = start.elapsed();
        }
        "convert_to_memory" => {
            let backed = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let start = Instant::now();
            let _imanndata = convert_to_in_memory(backed)?;
            duration = start.elapsed();
        }
        "s3_zarr_read" => {
            let adata = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let start = Instant::now();
            let x = adata.x().get()?.context("X is empty")?;
            force_eval(x);
            duration = start.elapsed();
        }
        "s3_zarr_subset" => {
            let adata = AnnData::<B>::open(B::open_rw(&cli.dataset)?)?;
            let n_rows = (adata.n_obs() as f64 * params.row_fraction) as usize;
            let selection = [SelectInfoElem::from(0..n_rows), SelectInfoElem::full()];
            let start = Instant::now();
            let x = adata.x().slice(&selection)?.context("X subset is empty")?;
            force_eval(x);
            duration = start.elapsed();
        }
        _ => anyhow::bail!("Unknown operation: {}", cli.operation),
    }
    Ok(duration.as_secs_f64())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let params: Params = serde_json::from_str(&cli.params).unwrap_or(Params { row_fraction: 0.1 });

    let mut sys = System::new();
    let initial_mem = get_current_rss_mb(&mut sys);
    let stop_signal = Arc::new(AtomicBool::new(false));
    let peak_mem = Arc::new(parking_lot::Mutex::new(initial_mem));

    let monitor_stop = Arc::clone(&stop_signal);
    let monitor_peak = Arc::clone(&peak_mem);
    let monitor_handle = thread::spawn(move || {
        monitor_memory(monitor_stop, monitor_peak);
    });

    let duration_sec = if cli.dataset.ends_with(".h5ad") {
        run_with_backend::<H5>(&cli, &params)?
    } else {
        run_with_backend::<Zarr>(&cli, &params)?
    };

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
        duration_sec,
        initial_memory_mb: initial_mem,
        peak_memory_mb: *peak_mem.lock(),
        final_memory_mb: final_mem,
    };

    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
