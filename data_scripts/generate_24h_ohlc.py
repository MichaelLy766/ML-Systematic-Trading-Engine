import sys
import os
sys.path.append(os.path.abspath('..'))

import polars as pl
import research
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import os

def process_file(file_path):
    try:
        df = pl.read_parquet(file_path)
        ts = research.timeseries(df, "1d", research.OHLC_AGGS)
        return ts
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        return pl.DataFrame()

if __name__ == '__main__':
    cache_dir = Path('../cache')
    parquet_files = sorted(list(cache_dir.glob('*.parquet')))

    print(f"Found {len(parquet_files)} parquet files. Processing in parallel...")

    # Limit to reasonable number of cores to avoid OOM
    num_cores = min(os.cpu_count() or 4, 8)
    
    with mp.Pool(processes=num_cores, maxtasksperchild=1) as pool:
        # pool.map preserves order!
        results = list(tqdm(pool.imap(process_file, parquet_files), total=len(parquet_files)))

    # Filter out any empty DataFrames
    results = [ts for ts in results if len(ts) > 0]

    if results:
        final_df = pl.concat(results).sort('datetime')
        final_df.write_csv("../historical_data/BTCUSDT_24h_ohlc_updated.csv")
        print("Successfully saved BTCUSDT_24h_ohlc_updated.csv")
    else:
        print("No data processed.")
