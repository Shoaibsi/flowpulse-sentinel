[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Shoaibsi/flowpulse-sentinel/python-app.yml?branch=main)](https://github.com/Shoaibsi/flowpulse-sentinel/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

# FlowPulse Sentinel

An AI-powered trading bot that analyzes unusual options flow and generates trading insights using Temporal Fusion Transformer (TFT) price prediction and Gemini AI analysis.

## Features

- **TFT Price Prediction**: Uses 30-day OHLCV windows with advanced temporal modeling for accurate price forecasts
- **Options Flow Analysis**: Detects unusual options activity based on volume/open interest ratios
- **Gemini AI Integration**: Generates professional trading analysis and recommendations
- **Real-time Alerts**: Sends formatted alerts via Telegram with priority-based filtering
- **Multiple Data Sources**: Primary data from TastyTrade API with Polygon and YFinance fallbacks
- **In-memory Database**: DuckDB for efficient data storage and querying

## Architecture

```
flowpulse/
├── bot/
│   ├── core.py             # Main bot logic
│   ├── data_fetcher.py     # TastyTrade/Polygon/YFinance
│   ├── tft_predictor.py    # TFT model
│   ├── alert_manager.py    # Telegram alerts
│   └── analysis.py         # Gemini integration
├── config.py               # API keys
├── models/                 # Store TFT model
└── tests/                  # Unit tests
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/flowpulse-sentinel.git
   cd flowpulse-sentinel
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys in `flowpulse/config.py`:
   ```python
   TASTYTRADE_USERNAME = "your_tastytrade_username"
   TASTYTRADE_PASSWORD = "your_tastytrade_password"
   POLYGON_KEY = "your_polygon_api_key"
   TELEGRAM_TOKEN = "your_telegram_bot_token"
   HF_TOKEN = "your_huggingface_token"  # For DeepSeek model
   ```

4. Download or train the TFT model:
   ```
   # Place your trained TFT model in the models directory
   mkdir -p models
   # Copy your tft.pt file to models/
   ```

## Usage

Run the bot with default settings:
```
python -m flowpulse
```

Run with custom tickers:
```
python -m flowpulse --tickers SPY,QQQ,AAPL,TSLA
```

Debug mode:
```
python -m flowpulse --debug
```

Dry run (no alerts sent):
```
python -m flowpulse --dry-run
```

## Alert Format

Alerts are sent via Telegram in the following format:

```
🚨 *TICKER EXPIRY $STRIKE(C/P)*
📈 TFT: +2.5%
📊 Flow: 1500/500 (3.0x)
💡 AI: Technical analysis shows bullish momentum with resistance at $450.
    Target: $455-460 within 5 days.
    Risk management: Set stop loss at $445.
```

## Implementation Details

1. **TFT Integration**
   - Uses 30-day OHLCV windows
   - Advanced temporal modeling for accurate predictions
   - Fallback to technical indicators if model fails

2. **Gemini Prompts**
   - Specialized templates for different scenarios:
     - Standard analysis
     - Gamma squeeze detection
     - IV crush analysis
     - Earnings plays

3. **Alert System**
   - Priority-based filtering
   - Rate limiting (configurable alerts per hour)
   - Optional digest mode for hourly summaries

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- DuckDB 0.8+
- Polygon API Client 1.10+
- YFinance 0.2.20+
- Python Telegram Bot 13.0+

## License

MIT License
