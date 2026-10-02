"""
MOEX Equities Screener (ISS API)
--------------------------------
Purpose: same idea as the Binance futures screener, but for Moscow
Exchange shares. Narrows the TQBR board down to a handful of names
that already show multi-indicator convergence on the hourly chart,
so you only run your detailed AI scalping prompt on tickers worth
analyzing.

Data source: MOEX ISS (Informational & Statistical Server), the
exchange's own public JSON API - https://iss.moex.com. No API key
needed, no scraping of Finam/BCS/TradingView pages (those are HTML
pages meant for browsers, not a stable data source to script
against). ISS is the same feed those sites are themselves built on.

Pipeline:
  1. Pull all TQBR board securities with today's reference + live
     market data (last price, % change, today's turnover in RUB).
  2. Filter by a minimum turnover floor (avoid thin/illiquid names -
     this also naturally screens out most qualified-investor-only /
     low-free-float tickers, since they rarely clear real turnover).
  3. Pull hourly candles for each surviving ticker.
  4. Compute RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA(9,21,200).
  5. Score each ticker by how many of your prompt's conditions it
     already satisfies (trend / momentum / volatility alignment).
  6. Print the top N candidates, ranked by score, ready to paste into
     your detailed AI prompt.

Requirements:
  pip install requests pandas numpy

Usage:
  python moex_screener.py                     # default: top 5 candidates
  python moex_screener.py --top 10
  python moex_screener.py --min-value 30000000   # min today's turnover in RUB
  python moex_screener.py --exclude BLNG,KZOS    # always skip specific tickers
  python moex_screener.py --diagnostic           # show top-N regardless of threshold
  python moex_screener.py --dump-candles HYDR    # export candles+ATR for one ticker
  python moex_screener.py --dump-candles HYDR --rows 100 --out hydr.csv

Notes on MOEX specifics vs. the crypto version:
  - MOEX trades in sessions (main ~09:50-18:50 MSK, evening
    ~19:00-23:50 MSK), not 24/7, so hourly candles have gaps and
    "last candle" may be from the previous session if run outside
    trading hours. The script warns if the latest candle looks stale.
  - A handful of MOEX-listed shares are restricted to investors with
    "qualified investor" status and/or trade with very thin real
    volume despite occasional headline price prints. ISS doesn't
    expose a clean boolean for this, so the turnover filter
    (--min-value) is your main defense - raise it if thin names keep
    surfacing, or add them to --exclude once you notice them.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

ISS_BASE = "https://iss.moex.com/iss"
HEADERS = {"User-Agent": "moex-screener/1.0 (personal research script)"}


def _iss_get(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _block_to_df(payload, block_name):
    """ISS responses look like {"block": {"columns": [...], "data": [[...], ...]}}."""
    block = payload.get(block_name)
    if not block or not block.get("data"):
        return pd.DataFrame(columns=block.get("columns", []) if block else [])
    return pd.DataFrame(block["data"], columns=block["columns"])


def get_board_securities(board="TQBR"):
    """
    Reference data + live market data for every share on a given board.
    TQBR = main T+1 board for ordinary shares (the liquid, retail-traded
    board most MOEX equities scalping happens on).
    """
    url = f"{ISS_BASE}/engines/stock/markets/shares/boards/{board}/securities.json"
    payload = _iss_get(url, params={"iss.meta": "off"})

    ref = _block_to_df(payload, "securities")
    md = _block_to_df(payload, "marketdata")

    if ref.empty or md.empty:
        return pd.DataFrame()

    ref = ref[["SECID", "SHORTNAME", "LOTSIZE", "LISTLEVEL"]]
    md = md[["SECID", "LAST", "LASTCHANGEPRCNT", "VALTODAY", "VOLTODAY",
             "TRADINGSTATUS", "UPDATETIME"]]

    df = ref.merge(md, on="SECID", how="inner")
    df = df.rename(columns={
        "SECID": "symbol",
        "SHORTNAME": "name",
        "LAST": "last_price",
        "LASTCHANGEPRCNT": "change_pct",
        "VALTODAY": "turnover_rub",
        "VOLTODAY": "volume_today",
    })
    # Drop halted / no-trade names (no last price or zero turnover today)
    df = df.dropna(subset=["last_price"])
    df = df[df["turnover_rub"].fillna(0) > 0]
    return df


def get_candles(symbol, board="TQBR", interval=60, days_back=45):
    """
    Hourly (interval=60) candles from ISS.
    Interval codes: 1=1min, 10=10min, 60=1h, 24=1day, 7=1week, 31=1month.
    days_back needs to be generous enough that, once MOEX's session
    gaps are accounted for, you still end up with 200+ hourly candles
    for a stable EMA200 (roughly 15-20 trading days' worth).
    """
    till = datetime.now().date()
    frm = till - timedelta(days=days_back)

    url = (f"{ISS_BASE}/engines/stock/markets/shares/boards/{board}/"
           f"securities/{symbol}/candles.json")

    all_rows = []
    start = 0
    while True:
        params = {
            "interval": interval,
            "from": frm.isoformat(),
            "till": till.isoformat(),
            "start": start,
            "iss.meta": "off",
        }
        payload = _iss_get(url, params=params)
        chunk = _block_to_df(payload, "candles")
        if chunk.empty:
            break
        all_rows.append(chunk)
        if len(chunk) < 500:  # ISS pages results in chunks of ~500
            break
        start += 500

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.sort_values("begin").reset_index(drop=True)
    df = df.rename(columns={"value": "quote_value"})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def _ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(df, length=14):
    """Average True Range - needed for real ATR-based TP/SL, not just the
    qualitative BB-zone read the screener's scoring uses."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def compute_indicators(df):
    """EMA(9/21/200), RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14) -
    plain pandas/numpy math, identical logic to the crypto screener so scores
    are directly comparable."""
    df["ema9"] = _ema(df["close"], 9)
    df["ema21"] = _ema(df["close"], 21)
    df["ema200"] = _ema(df["close"], 200)

    df["rsi14"] = _rsi(df["close"], 14)

    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    df["macd"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["atr14"] = _atr(df, 14)
    return df


def score_pair(df, macd_lookback=3, bb_zone_pct=15.0):
    """
    Same convergence logic as the crypto screener:
      - Trend: price vs 200 EMA
      - Momentum: RSI extreme + MACD state vs signal line
      - Volatility: price vs Bollinger Band zone (not just exact touch)
      - Volume: spike vs 20-period average (flagged, not a directional point)
    Returns (direction, score 0-4, notes list).
    """
    if len(df) < 210 or df["ema200"].isna().iloc[-1]:
        return None, 0, ["insufficient history for a stable 200 EMA"]

    last = df.iloc[-1]
    notes = []
    long_pts = 0
    short_pts = 0

    if last["close"] > last["ema200"]:
        long_pts += 1
        notes.append("price>200EMA (uptrend)")
    elif last["close"] < last["ema200"]:
        short_pts += 1
        notes.append("price<200EMA (downtrend)")

    if last["rsi14"] < 30:
        long_pts += 1
        notes.append(f"RSI oversold ({last['rsi14']:.1f})")
    elif last["rsi14"] > 70:
        short_pts += 1
        notes.append(f"RSI overbought ({last['rsi14']:.1f})")

    window = df.iloc[-(macd_lookback + 1):]
    crossed_up_recently = ((window["macd"].shift(1) < window["macd_signal"].shift(1)) &
                            (window["macd"] > window["macd_signal"])).any()
    crossed_down_recently = ((window["macd"].shift(1) > window["macd_signal"].shift(1)) &
                              (window["macd"] < window["macd_signal"])).any()
    if last["macd"] > last["macd_signal"]:
        long_pts += 1
        notes.append("MACD bullish" + (" (fresh cross)" if crossed_up_recently else " (holding above signal)"))
    elif last["macd"] < last["macd_signal"]:
        short_pts += 1
        notes.append("MACD bearish" + (" (fresh cross)" if crossed_down_recently else " (holding below signal)"))

    band_range = last["bb_upper"] - last["bb_lower"]
    if pd.notna(band_range) and band_range > 0:
        pct_from_bottom = (last["close"] - last["bb_lower"]) / band_range * 100
        if pct_from_bottom <= bb_zone_pct:
            long_pts += 1
            notes.append(f"price near lower BB ({pct_from_bottom:.0f}% of band)")
        elif pct_from_bottom >= 100 - bb_zone_pct:
            short_pts += 1
            notes.append(f"price near upper BB ({pct_from_bottom:.0f}% of band)")

    vol_spike = last["volume"] > 1.5 * last["vol_avg20"] if pd.notna(last["vol_avg20"]) else False
    if vol_spike:
        notes.append("volume spike vs 20-period avg")

    stale_hours = (datetime.now() - last["begin"].to_pydatetime()).total_seconds() / 3600
    if stale_hours > 20:
        notes.append(f"NOTE: latest candle is ~{stale_hours:.0f}h old (market likely closed - stale read)")

    if long_pts >= short_pts and long_pts > 0:
        return "Long", long_pts, notes
    elif short_pts > 0:
        return "Short", short_pts, notes
    else:
        return None, 0, notes


def dump_candles(symbol, board="TQBR", interval=60, days_back=45, rows=50, out_path=None):
    """
    Pull hourly candles + full indicator set for ONE ticker and print them
    in a compact, pasteable table - the actual data your detailed AI prompt
    needs to compute real ATR-based entries/targets/stops, instead of
    guessing off the one-line screener summary.
    Also writes a CSV so you don't have to retype anything.
    """
    print(f"Fetching {symbol} {interval}-min candles from MOEX ISS ({board})...\n")
    df = get_candles(symbol, board=board, interval=interval, days_back=days_back)
    if df.empty:
        print(f"No candle data returned for {symbol}. Check the ticker/board code.")
        return
    df = compute_indicators(df)

    last = df.iloc[-1]
    stale_hours = (datetime.now() - last["begin"].to_pydatetime()).total_seconds() / 3600
    print(f"=== {symbol} snapshot @ {last['begin']} "
          f"{'(STALE - market likely closed)' if stale_hours > 20 else ''} ===")
    print(f"Last close:  {last['close']:.4f}")
    print(f"EMA9/21/200: {last['ema9']:.4f} / {last['ema21']:.4f} / {last['ema200']:.4f}")
    print(f"RSI(14):     {last['rsi14']:.1f}")
    print(f"MACD/Signal/Hist: {last['macd']:.5f} / {last['macd_signal']:.5f} / {last['macd_hist']:.5f}")
    print(f"Bollinger upper/mid/lower: {last['bb_upper']:.4f} / {last['bb_mid']:.4f} / {last['bb_lower']:.4f}")
    print(f"ATR(14):     {last['atr14']:.4f}  "
          f"(=> a min 1:2 setup risking 1x ATR targets >= 2x ATR)")
    print(f"Volume vs 20-avg: {last['volume']:.0f} vs {last['vol_avg20']:.0f}\n")

    cols = ["begin", "open", "high", "low", "close", "volume",
            "rsi14", "macd", "macd_signal", "bb_upper", "bb_lower", "ema200", "atr14"]
    table = df[cols].tail(rows).copy()
    table["begin"] = table["begin"].dt.strftime("%Y-%m-%d %H:%M")
    for c in ["open", "high", "low", "close", "rsi14", "macd", "macd_signal",
              "bb_upper", "bb_lower", "ema200", "atr14"]:
        table[c] = table[c].round(5)

    print(f"Last {len(table)} hourly candles + indicators "
          f"(paste this block, or the CSV, into the detailed prompt):\n")
    print(table.to_string(index=False))

    if out_path is None:
        out_path = f"{symbol}_{interval}min_candles.csv"
    table.to_csv(out_path, index=False)
    print(f"\nAlso saved to {out_path}")


def run_screen(min_value, top_n, board, interval, sleep_between, min_score, exclude, diagnostic):
    print("Fetching TQBR board securities from MOEX ISS...")
    board_df = get_board_securities(board=board)
    if board_df.empty:
        print("No data returned from ISS - check network/board code.")
        return

    if exclude:
        excl = {t.strip().upper() for t in exclude.split(",") if t.strip()}
        board_df = board_df[~board_df["symbol"].isin(excl)]

    liquid = board_df[board_df["turnover_rub"] >= min_value].sort_values(
        "turnover_rub", ascending=False
    )
    print(f"{len(liquid)} tickers pass the {min_value:,.0f} RUB today's-turnover floor "
          f"(out of {len(board_df)} traded on {board} today).\n")

    all_scored = []
    for i, row in enumerate(liquid.itertuples(), start=1):
        symbol = row.symbol
        try:
            df = get_candles(symbol, board=board, interval=interval)
            if df.empty:
                continue
            df = compute_indicators(df)
            direction, score, notes = score_pair(df)
            if direction:
                all_scored.append({
                    "symbol": symbol,
                    "name": row.name,
                    "direction": direction,
                    "score": score,
                    "notes": "; ".join(notes),
                    "last_price": df["close"].iloc[-1],
                    "today_turnover_rub": row.turnover_rub,
                    "today_change_pct": row.change_pct,
                })
        except Exception as e:
            print(f"  [skip] {symbol}: {e}", file=sys.stderr)

        if sleep_between:
            time.sleep(sleep_between)

    if not all_scored:
        print("No tickers currently meet 3+ indicator convergence. NO SIGNAL universe-wide.")
        return

    scored_df = pd.DataFrame(all_scored).sort_values("score", ascending=False)
    results = scored_df[scored_df["score"] >= min_score]

    if diagnostic:
        print(f"DIAGNOSTIC MODE: showing top {top_n} tickers by score, regardless of threshold.\n")
        out = scored_df.head(top_n)
    elif len(results) == 0:
        print(f"No tickers currently meet the {min_score}+ indicator threshold.\n"
              f"Closest tickers (run with --diagnostic to always see this):\n")
        out = scored_df.head(top_n)
    else:
        out = results.head(top_n)
        print(f"Top {len(out)} candidate(s) meeting {min_score}+ indicators for your detailed prompt:\n")

    for r in out.itertuples():
        print(f"{r.symbol} ({r.name})  |  {r.direction}  |  score {r.score}/4  |  "
              f"price {r.last_price:.4g}  |  today turnover {r.today_turnover_rub:,.0f} RUB  |  "
              f"today chg {r.today_change_pct}%")
        print(f"   conditions: {r.notes}\n")

    if not diagnostic and len(results) > 0:
        print("Paste one of these tickers + its candle data into your detailed MOEX scalping prompt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen MOEX shares for indicator convergence.")
    parser.add_argument("--board", type=str, default="TQBR",
                         help="MOEX board code. Default TQBR (main T+1 board, ordinary shares).")
    parser.add_argument("--min-value", type=float, default=30_000_000,
                         help="Minimum today's turnover in RUB (liquidity floor). Default 30M.")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to show.")
    parser.add_argument("--interval", type=int, default=60,
                         help="ISS candle interval code (60=1h, matches your prompt). ")
    parser.add_argument("--sleep", type=float, default=0.2,
                         help="Seconds to sleep between API calls (be polite to ISS).")
    parser.add_argument("--min-score", type=int, default=3,
                         help="Minimum indicators that must align (out of 4) to count as a signal. Default 3.")
    parser.add_argument("--exclude", type=str, default="",
                         help="Comma-separated tickers to always skip, e.g. BLNG,KZOS "
                              "(useful once you spot qualified-investor-only / illiquid names).")
    parser.add_argument("--diagnostic", action="store_true",
                         help="Always show the top-scoring tickers, even below the threshold.")
    parser.add_argument("--dump-candles", type=str, default=None, metavar="TICKER",
                         help="Skip the full board scan and instead print + save the last "
                              "--rows hourly candles with all indicators (incl. ATR) for one "
                              "ticker - the data your detailed AI prompt needs for real "
                              "ATR-based entry/TP/SL, not just the one-line screener summary.")
    parser.add_argument("--rows", type=int, default=50,
                         help="How many recent candles to print/save with --dump-candles. Default 50.")
    parser.add_argument("--out", type=str, default=None,
                         help="CSV output path for --dump-candles. Defaults to "
                              "<TICKER>_<interval>min_candles.csv in the current directory.")
    args = parser.parse_args()

    if args.dump_candles:
        dump_candles(args.dump_candles.upper(), board=args.board, interval=args.interval,
                     rows=args.rows, out_path=args.out)
    else:
        run_screen(args.min_value, args.top, args.board, args.interval,
                   args.sleep, args.min_score, args.exclude, args.diagnostic)
