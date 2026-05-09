import sys
import os
sys.path.append(os.path.abspath('..'))

import binance
from datetime import datetime

sym = 'BTCUSDT'
start_date = datetime(2025, 5, 5, 0, 0)
end_date = datetime(2026, 5, 5, 0, 0)

print(f"Starting contiguous data download for {sym} from {start_date.date()} to {end_date.date()}...")
binance.download_date_range(sym, start_date, end_date)
print("Download complete!")
