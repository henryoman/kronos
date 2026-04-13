# Kronos Demo App

Small local web app for swapping Kronos demo instruments and chart intervals from JSON config.

## Run

```bash
uv run --python 3.14 demo_app/app.py
```

Then open:

```text
http://127.0.0.1:7071
```

## Change Markets

Edit `demo_app/config.json` and add another object to `demos`.

```json
{
  "id": "eth_15m_bybit",
  "label": "ETH/USDT Perp 15m",
  "provider": "bybit",
  "market": "linear",
  "symbol": "ETHUSDT",
  "interval": "15m",
  "strategy": {
    "mode": "threshold"
  }
}
```

Supported providers:

- `binance` with `market` as `spot` or `linear`/`futures`
- `bybit` with `market` as `spot`, `linear`, or `inverse`
- `csv` with `source_path` and optional `source_interval` for local OHLCV files

The backend always normalizes candles to:

```text
timestamp, open, high, low, close, volume, amount
```

Kronos settings come from `defaults` unless a demo overrides them.

Example CSV-backed demo:

```json
{
  "id": "nq_1h_kaggle",
  "label": "NASDAQ NQ 1h",
  "provider": "csv",
  "market": "futures",
  "symbol": "CME_MINI:NQ1!",
  "interval": "1h",
  "source_path": "data/demo_app/NQ_in_1_hour.csv",
  "source_interval": "1h",
  "strategy": {
    "mode": "threshold"
  }
}
```

If the CSV omits `amount`, the backend fills it with `0.0`.

## Refresh NQ Data

Pull the public NASDAQ NQ futures files from Kaggle with `uvx`:

```bash
mkdir -p data/demo_app
for file in \
  NQ_in_1_minute.csv \
  NQ_in_5_minute.csv \
  NQ_in_1_hour.csv \
  NQ_in_daily.csv \
  NQ_in_weekly.csv
do
  uvx --from kaggle kaggle datasets download \
    youneseloiarm/nasdaq-cme-future-nq \
    -f "$file" \
    -p data/demo_app \
    --unzip
done
```

## Render Images

Generate chart PNGs for BTC and SOL on 1h and 5m:

```bash
uv run --python 3.14 demo_app/render_images.py --ids btc_1h_binance btc_5m_binance sol_1h_binance sol_5m_binance
```

Outputs:

```text
outputs/demo_app/btc_1h_binance.png
outputs/demo_app/btc_5m_binance.png
outputs/demo_app/sol_1h_binance.png
outputs/demo_app/sol_5m_binance.png
outputs/demo_app/btc_sol_1h_5m_grid.png
```

Render the Nasdaq demo set:

```bash
uv run --python 3.14 demo_app/render_images.py --ids \
  nq_1m_kaggle \
  nq_5m_kaggle \
  nq_1h_kaggle \
  nq_12h_kaggle \
  nq_1d_kaggle \
  nq_2d_kaggle \
  nq_1w_kaggle
```
