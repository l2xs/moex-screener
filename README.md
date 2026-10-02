# moex-screener
Helps find stocks on Moscow Exchange (MOEX) with available entry points for trades

How to use

Install Python from python.org
Open Windows PowerShell
cd "path to moex-screener.py"

python moex-screener.py

python moex2_ticker_dump.py --dump-candles "TICKER"           # Skips the board scan entirely and, for one ticker, prints: A snapshot block (last close, EMA9/21/200, RSI, MACD/signal/hist, Bollinger levels, and now ATR(14), plus volume vs 20-avg) — the real ATR number that was missing before.
The last 50 hourly candles (OHLCV) with every indicator attached, in a clean table.
Saves the same table to TICKER_60min_candles.csv automatically (The table can be used with LLMs, since they can't reach MOEX ISS themselves)
Flags: --rows 100 for more history, --out path.csv to control the filename, and it still respects --board/--interval if you ever want a different board or timeframe.
