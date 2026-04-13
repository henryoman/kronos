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

The backend always normalizes candles to:

```text
timestamp, open, high, low, close, volume, amount
```

Kronos settings come from `defaults` unless a demo overrides them.

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
