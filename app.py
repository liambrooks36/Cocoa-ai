from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime, timezone
import os
import time
import re
import json

import requests
import feedparser
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI


# ============================================================
# APP SETUP
# ============================================================

ROOT = Path(__file__).parent
app = FastAPI(title="Silver AI V1.6.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# ENVIRONMENT
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

ai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# DATABASE
# ============================================================

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")

    return psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
        connect_timeout=10,
    )


def database_status():
    if not DATABASE_URL:
        return False, "DATABASE_URL is not configured"

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchone()
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def ensure_tables():
    if not DATABASE_URL:
        return

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        prediction TEXT NOT NULL,
                        confidence INTEGER,
                        price DOUBLE PRECISION,
                        technical TEXT,
                        weather TEXT,
                        macro TEXT,
                        entry TEXT,
                        price_24h DOUBLE PRECISION,
                        return_24h DOUBLE PRECISION,
                        result_24h TEXT,
                        graded_24h_at TIMESTAMPTZ
                    )
                    """
                )

                for sql in (
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS technical TEXT",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS weather TEXT",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS macro TEXT",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS entry TEXT",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS price_24h DOUBLE PRECISION",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS return_24h DOUBLE PRECISION",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS result_24h TEXT",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS graded_24h_at TIMESTAMPTZ",
                ):
                    cur.execute(sql)

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS predictions_created_at_idx
                    ON predictions(created_at DESC)
                    """
                )
    except Exception as exc:
        print("Database table setup warning:", type(exc).__name__, str(exc))


@app.on_event("startup")
def startup():
    ensure_tables()


# ============================================================
# HOME / HEALTH
# ============================================================

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health():
    db_ok, db_error = database_status()

    return {
        "ok": True,
        "service": "silver-ai-v1.0",
        "database_configured": bool(DATABASE_URL),
        "database_ok": db_ok,
        "database_error": db_error,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL if OPENAI_API_KEY else None,
        "analysis_mode": "DAY_TRADING",
        "twelve_data_configured": bool(TWELVE_DATA_API_KEY),
        "primary_market_feed": "Twelve Data XAG/USD" if TWELVE_DATA_API_KEY else "Yahoo SI=F fallback",
    }


# ============================================================
# SILVER MARKET CANDLES
# ============================================================

CANDLE_CACHE = {}
CANDLE_CACHE_SECONDS = {
    "1m": 20,
    "5m": 30,
    "15m": 45,
    "30m": 60,
    "60m": 60,
    "1h": 60,
    "1d": 300,
}


def _cached_candles(interval, range_name):
    key = (interval, range_name)
    item = CANDLE_CACHE.get(key)
    if not item:
        return None

    age = time.time() - item["timestamp"]
    ttl = CANDLE_CACHE_SECONDS.get(interval, 60)

    if age <= ttl:
        data = dict(item["data"])
        data["cached"] = True
        data["cache_age_seconds"] = round(age, 1)
        return data

    return None


def _stale_cached_candles(interval, range_name):
    key = (interval, range_name)
    item = CANDLE_CACHE.get(key)
    if not item:
        return None

    data = dict(item["data"])
    data["cached"] = True
    data["stale"] = True
    data["cache_age_seconds"] = round(time.time() - item["timestamp"], 1)
    return data


def _fetch_yahoo_chart(symbol, yahoo_interval, range_name):
    last_error = None

    params = {
        "interval": yahoo_interval,
        "range": range_name,
        "includePrePost": "false",
        "events": "div,splits",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SilverAI/1.0)",
        "Accept": "application/json,text/plain,*/*",
        "Cache-Control": "no-cache",
    }

    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{requests.utils.quote(symbol, safe='')}"

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:
                last_error = RuntimeError(f"{host} returned HTTP 429")
                continue

            resp.raise_for_status()
            payload = resp.json()

            chart = payload.get("chart", {})
            if chart.get("error"):
                last_error = RuntimeError(str(chart["error"]))
                continue

            results = chart.get("result") or []
            if not results:
                last_error = RuntimeError(f"{host} returned no chart result")
                continue

            return results[0], host
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Both Yahoo chart hosts failed: {type(last_error).__name__}: {last_error}"
    )


def _twelve_interval(interval):
    return {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "1h",
        "1h": "1h",
        "1d": "1day",
    }[interval]


def _twelve_outputsize(interval, range_name):
    sizes = {
        ("1m", "1d"): 500,
        ("5m", "5d"): 500,
        ("15m", "5d"): 300,
        ("1h", "1mo"): 500,
        ("60m", "1mo"): 500,
        ("1d", "1y"): 365,
    }
    return sizes.get((interval, range_name), 500)


def _fetch_twelve_silver(interval, range_name):
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is not configured")

    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": "XAG/USD",
            "interval": _twelve_interval(interval),
            "outputsize": _twelve_outputsize(interval, range_name),
            "timezone": "UTC",
            "order": "ASC",
            "apikey": TWELVE_DATA_API_KEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") == "error" or not payload.get("values"):
        raise RuntimeError(payload.get("message") or "Twelve Data returned no XAG/USD candles")

    rows = []
    for x in payload["values"]:
        try:
            dt = datetime.strptime(x["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            rows.append({
                "t": int(dt.timestamp()),
                "o": float(x["open"]),
                "h": float(x["high"]),
                "l": float(x["low"]),
                "c": float(x["close"]),
                "v": float(x.get("volume") or 0),
            })
        except (KeyError, TypeError, ValueError):
            continue

    rows.sort(key=lambda x: x["t"])
    if not rows:
        raise RuntimeError("Twelve Data returned zero usable XAG/USD OHLC candles")

    return rows, payload.get("meta") or {}


def _yahoo_silver_response(interval, range_name):
    symbol = "SI=F"
    yahoo_interval = "60m" if interval == "1h" else interval
    result, yahoo_host = _fetch_yahoo_chart(symbol, yahoo_interval, range_name)

    timestamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
    opens, highs, lows, closes, volumes = [
        quote.get(k) or [] for k in ("open", "high", "low", "close", "volume")
    ]

    out = []
    for i, ts in enumerate(timestamps):
        try:
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            l = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else 0

            if None in (o, h, l, c):
                continue

            out.append({
                "t": int(ts),
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "v": float(v or 0),
            })
        except (TypeError, ValueError, IndexError):
            pass

    if not out:
        raise RuntimeError("Yahoo returned zero usable OHLC candles")

    meta = result.get("meta") or {}

    # FIX: range_name must be returned here, NOT Python's built-in range function.
    return {
        "symbol": symbol,
        "source": f"Yahoo Finance SI=F fallback ({yahoo_host})",
        "provider": "yahoo",
        "delayed": True,
        "fallback": True,
        "cached": False,
        "stale": False,
        "interval": interval,
        "range": range_name,
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName"),
        "regular_market_price": meta.get("regularMarketPrice"),
        "candles": out,
    }


@app.get("/api/candles")
def candles(
    interval: str = Query("1h", pattern="^(1m|5m|15m|30m|60m|1h|1d)$"),
    range: str = Query("1mo", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"),
):
    cached = _cached_candles(interval, range)
    if cached is not None:
        return cached

    primary_error = None

    try:
        rows, meta = _fetch_twelve_silver(interval, range)
        data = {
            "symbol": "XAG/USD",
            "source": "Twelve Data XAG/USD",
            "provider": "twelve_data",
            "delayed": False,
            "fallback": False,
            "cached": False,
            "stale": False,
            "interval": interval,
            "range": range,
            "currency": meta.get("currency", "USD"),
            "exchange": meta.get("exchange"),
            "regular_market_price": rows[-1]["c"],
            "candles": rows,
        }
        CANDLE_CACHE[(interval, range)] = {"timestamp": time.time(), "data": data}
        return data
    except Exception as exc:
        primary_error = f"{type(exc).__name__}: {exc}"

    try:
        data = _yahoo_silver_response(interval, range)
        data["primary_feed_error"] = primary_error

        CANDLE_CACHE[(interval, range)] = {
            "timestamp": time.time() - max(0, CANDLE_CACHE_SECONDS.get(interval, 60) - 10),
            "data": data,
        }
        return data
    except Exception as fallback_exc:
        stale = _stale_cached_candles(interval, range)
        if stale is not None:
            stale["warning"] = (
                f"Twelve Data failed ({primary_error}); "
                f"Yahoo failed ({type(fallback_exc).__name__}: {fallback_exc})"
            )
            return stale

        return {
            "symbol": "XAG/USD",
            "source": "No market feed",
            "provider": "none",
            "delayed": True,
            "fallback": True,
            "cached": False,
            "stale": False,
            "interval": interval,
            "range": range,
            "candles": [],
            "error": (
                f"Twelve Data: {primary_error}; "
                f"Yahoo: {type(fallback_exc).__name__}: {fallback_exc}"
            ),
        }


# ============================================================
# TECHNICAL HELPERS
# ============================================================

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None

    gains = 0.0
    losses = 0.0

    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def aggregate_candles(candle_rows, group_size=4):
    if group_size <= 1:
        return candle_rows

    out = []
    for i in range(0, len(candle_rows), group_size):
        group = candle_rows[i:i + group_size]
        if len(group) < group_size:
            continue

        out.append({
            "t": group[0].get("t"),
            "o": group[0]["o"],
            "h": max(x["h"] for x in group),
            "l": min(x["l"] for x in group),
            "c": group[-1]["c"],
            "v": sum(float(x.get("v") or 0) for x in group),
        })

    return out


def market_metrics(candle_rows):
    if not candle_rows:
        return {}

    rows = [
        x for x in candle_rows
        if all(x.get(k) is not None for k in ("o", "h", "l", "c"))
    ]

    if len(rows) < 2:
        return {}

    closes = [float(x["c"]) for x in rows]
    highs = [float(x["h"]) for x in rows]
    lows = [float(x["l"]) for x in rows]
    volumes = [float(x.get("v") or 0) for x in rows]

    last = closes[-1]
    previous = closes[-2]
    latest_move = ((last - previous) / previous) * 100 if previous else 0
    rsi = calc_rsi(closes)

    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None

    move_3 = ((last - closes[-4]) / closes[-4]) * 100 if len(closes) >= 4 and closes[-4] else None
    move_5 = ((last - closes[-6]) / closes[-6]) * 100 if len(closes) >= 6 and closes[-6] else None
    move_10 = ((last - closes[-11]) / closes[-11]) * 100 if len(closes) >= 11 and closes[-11] else None

    trend = "neutral"
    if sma20 is not None:
        if last > sma20 * 1.0015:
            trend = "bullish"
        elif last < sma20 * 0.9985:
            trend = "bearish"

    avg_volume_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    last3 = rows[-3:] if len(rows) >= 3 else rows
    recent8 = rows[-8:]

    green_last3 = sum(1 for x in last3 if float(x["c"]) > float(x["o"]))
    red_last3 = sum(1 for x in last3 if float(x["c"]) < float(x["o"]))
    green_last8 = sum(1 for x in recent8 if float(x["c"]) > float(x["o"]))
    red_last8 = sum(1 for x in recent8 if float(x["c"]) < float(x["o"]))

    higher_lows = len(rows) >= 3 and lows[-1] > lows[-2] > lows[-3]
    lower_highs = len(rows) >= 3 and highs[-1] < highs[-2] < highs[-3]

    prior_high_10 = max(highs[-11:-1]) if len(highs) >= 11 else max(highs[:-1])
    prior_low_10 = min(lows[-11:-1]) if len(lows) >= 11 else min(lows[:-1])

    breakout_up = last > prior_high_10
    breakout_down = last < prior_low_10

    sma20_reclaim_up = sma20 is not None and previous <= sma20 and last > sma20
    sma20_reclaim_down = sma20 is not None and previous >= sma20 and last < sma20

    return {
        "last_price": round(last, 2),
        "latest_candle_move_pct": round(latest_move, 3),
        "move_3_bars_pct": round(move_3, 3) if move_3 is not None else None,
        "move_5_bars_pct": round(move_5, 3) if move_5 is not None else None,
        "move_10_bars_pct": round(move_10, 3) if move_10 is not None else None,
        "rsi_14": round(rsi, 2) if rsi is not None else None,
        "sma20": round(sma20, 2) if sma20 is not None else None,
        "sma50": round(sma50, 2) if sma50 is not None else None,
        "trend_vs_sma20": trend,
        "recent_high_10": round(max(highs[-10:]), 2),
        "recent_low_10": round(min(lows[-10:]), 2),
        "recent_high_20": round(max(highs[-20:]), 2),
        "recent_low_20": round(min(lows[-20:]), 2),
        "recent_high_50": round(max(highs[-50:]), 2),
        "recent_low_50": round(min(lows[-50:]), 2),
        "last_volume": round(volumes[-1], 2) if volumes else None,
        "avg_volume_20": round(avg_volume_20, 2) if avg_volume_20 is not None else None,
        "volume_vs_avg20": round(volumes[-1] / avg_volume_20, 2) if avg_volume_20 else None,
        "green_candles_last_3": green_last3,
        "red_candles_last_3": red_last3,
        "green_candles_last_8": green_last8,
        "red_candles_last_8": red_last8,
        "three_higher_lows": higher_lows,
        "three_lower_highs": lower_highs,
        "breakout_above_prior_10_bar_high": breakout_up,
        "breakdown_below_prior_10_bar_low": breakout_down,
        "sma20_reclaim_up": sma20_reclaim_up,
        "sma20_reclaim_down": sma20_reclaim_down,
        "candle_count": len(closes),
    }


# ============================================================
# SILVER MACRO CONTEXT
# ============================================================

MACRO_CACHE = {"timestamp": 0, "data": None}
MACRO_CACHE_SECONDS = 60


def _symbol_candles(symbol, interval="5m", range_name="5d"):
    yahoo_interval = "60m" if interval == "1h" else interval

    try:
        result, yahoo_host = _fetch_yahoo_chart(symbol, yahoo_interval, range_name)
        timestamps = result.get("timestamp") or []
        quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        out = []
        for i, ts in enumerate(timestamps):
            try:
                o = opens[i] if i < len(opens) else None
                h = highs[i] if i < len(highs) else None
                l = lows[i] if i < len(lows) else None
                c = closes[i] if i < len(closes) else None
                v = volumes[i] if i < len(volumes) else 0

                if None in (o, h, l, c):
                    continue

                out.append({
                    "t": int(ts),
                    "o": float(o),
                    "h": float(h),
                    "l": float(l),
                    "c": float(c),
                    "v": float(v or 0),
                })
            except (TypeError, ValueError, IndexError):
                continue

        return {
            "symbol": symbol,
            "source": f"Yahoo Finance chart API ({yahoo_host})",
            "delayed": True,
            "candles": out,
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "source": "Yahoo Finance chart API",
            "delayed": True,
            "candles": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _macro_item(symbol, label):
    feed = _symbol_candles(symbol, interval="5m", range_name="5d")
    rows = feed.get("candles") or []
    metrics = market_metrics(rows)

    return {
        "label": label,
        "symbol": symbol,
        "source": feed.get("source"),
        "error": feed.get("error"),
        **metrics,
    }


@app.get("/api/macro")
def macro():
    now = time.time()

    if MACRO_CACHE["data"] is not None and now - MACRO_CACHE["timestamp"] < MACRO_CACHE_SECONDS:
        cached = dict(MACRO_CACHE["data"])
        cached["cached"] = True
        cached["cache_age_seconds"] = int(now - MACRO_CACHE["timestamp"])
        return cached

    data = {
        "source": "Yahoo Finance supporting markets",
        "cached": False,
        "markets": {
            "gold": _macro_item("GC=F", "Gold Futures"),
            "dxy": _macro_item("DX-Y.NYB", "US Dollar Index"),
            "us10y": _macro_item("^TNX", "US 10Y Yield"),
        },
    }

    MACRO_CACHE["timestamp"] = now
    MACRO_CACHE["data"] = data
    return data


# ============================================================
# NEWS
# ============================================================

NEWS_CACHE = {"timestamp": 0, "data": None}
NEWS_CACHE_SECONDS = 10 * 60


@app.get("/api/news")
def news():
    now = time.time()

    if NEWS_CACHE["data"] is not None and now - NEWS_CACHE["timestamp"] < NEWS_CACHE_SECONDS:
        cached = dict(NEWS_CACHE["data"])
        cached["cached"] = True
        return cached

    query = (
        '"silver" OR "silver futures" OR XAG OR COMEX '
        'OR "US dollar" OR DXY OR "Federal Reserve" '
        'OR inflation OR "Treasury yields" OR gold'
    )
    rss = "https://news.google.com/rss/search"

    params = {"q": query, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SilverAI/1.0.1)"}

    try:
        resp = requests.get(rss, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        items = []
        for entry in feed.entries[:12]:
            source = ""
            if hasattr(entry, "source") and isinstance(entry.source, dict):
                source = entry.source.get("title", "")

            title = re.sub(r"\s+", " ", getattr(entry, "title", "")).strip()

            items.append({
                "title": title,
                "published_at": getattr(entry, "published", "Recent"),
                "source": source or "Google News",
                "link": getattr(entry, "link", ""),
            })

        data = {"source": "Google News RSS", "cached": False, "items": items}
        NEWS_CACHE["timestamp"] = now
        NEWS_CACHE["data"] = data
        return data

    except Exception as exc:
        if NEWS_CACHE["data"] is not None:
            cached = dict(NEWS_CACHE["data"])
            cached["cached"] = True
            cached["stale"] = True
            return cached

        return {
            "source": "Google News RSS",
            "items": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# PREDICTIONS
# ============================================================

@app.post("/api/predictions")
def save_prediction(payload: dict):
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database is not configured")

    prediction = str(payload.get("prediction", payload.get("pred", "NO TRADE")))

    try:
        confidence = int(payload.get("confidence", payload.get("conf", 0)) or 0)
    except Exception:
        confidence = 0

    try:
        price = float(payload.get("price", 0) or 0)
    except Exception:
        price = 0

    technical = str(payload.get("technical", payload.get("technical_score", "")) or "")
    macro_text = str(payload.get("macro", payload.get("macro_score", "")) or "")
    entry = str(payload.get("entry", payload.get("entry_quality", "")) or "")

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                    (prediction, confidence, price, technical, macro, entry)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at, prediction, confidence, price,
                              technical, macro, entry, price_24h, return_24h,
                              result_24h, graded_24h_at
                    """,
                    (prediction, confidence, price, technical, macro_text, entry),
                )
                row = cur.fetchone()

        return {"ok": True, "prediction": row}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database save failed: {type(exc).__name__}: {exc}",
        )


@app.get("/api/predictions")
def get_predictions(limit: int = Query(100, ge=1, le=1000)):
    if not DATABASE_URL:
        return {"database": False, "items": []}

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, created_at, prediction, confidence, price,
                           technical, macro, entry, price_24h, return_24h,
                           result_24h, graded_24h_at
                    FROM predictions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        return {"database": True, "items": rows}

    except Exception as exc:
        return {
            "database": True,
            "items": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
def performance():
    if not DATABASE_URL:
        return {
            "total_predictions": 0,
            "graded_24h": 0,
            "wins_24h": 0,
            "losses_24h": 0,
            "win_rate": None,
            "avg_return": None,
        }

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total_predictions,
                        COUNT(*) FILTER (WHERE graded_24h_at IS NOT NULL) AS graded_24h,
                        COUNT(*) FILTER (WHERE result_24h = 'WIN') AS wins_24h,
                        COUNT(*) FILTER (WHERE result_24h = 'LOSS') AS losses_24h,
                        AVG(return_24h) FILTER (WHERE graded_24h_at IS NOT NULL) AS avg_return
                    FROM predictions
                    """
                )
                row = cur.fetchone()

        total = int(row.get("total_predictions", 0) or 0)
        graded = int(row.get("graded_24h", 0) or 0)
        wins = int(row.get("wins_24h", 0) or 0)
        losses = int(row.get("losses_24h", 0) or 0)

        win_rate = round(wins / graded * 100, 2) if graded else None
        avg_return = float(row["avg_return"]) if row.get("avg_return") is not None else None

        return {
            "total_predictions": total,
            "graded_24h": graded,
            "wins_24h": wins,
            "losses_24h": losses,
            "win_rate": win_rate,
            "avg_return": avg_return,
        }

    except Exception as exc:
        return {
            "total_predictions": 0,
            "graded_24h": 0,
            "wins_24h": 0,
            "losses_24h": 0,
            "win_rate": None,
            "avg_return": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# AI SNAPSHOT â DAY TRADING FIRST
# ============================================================

def build_ai_snapshot():
    one_min = candles(interval="1m", range="1d")
    five_min = candles(interval="5m", range="5d")
    fifteen_min = candles(interval="15m", range="5d")
    one_hour = candles(interval="1h", range="1mo")

    one_rows = one_min.get("candles") or []
    five_rows = five_min.get("candles") or []
    fifteen_rows = fifteen_min.get("candles") or []
    one_hour_rows = one_hour.get("candles") or []

    macro_data = macro()
    news_data = news()
    perf_data = performance()

    headlines = []
    for item in (news_data.get("items") or [])[:12]:
        headlines.append({
            "title": item.get("title"),
            "published_at": item.get("published_at"),
            "source": item.get("source"),
        })

    feeds = [one_min, five_min, fifteen_min]
    primary_live = all(x.get("provider") == "twelve_data" and not x.get("delayed") for x in feeds)

    return {
        "asset": "Silver / XAGUSD",
        "symbol": "XAG/USD",
        "analysis_style": "SCALPING_1_TO_15_MINUTES",
        "primary_prediction_window": "1 to 15 minutes",
        "maximum_trade_horizon": "15 minutes",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),

        "important_data_warning": (
            "Primary silver candles are from Twelve Data XAG/USD when available. "
            "If any required timeframe falls back to Yahoo SI=F, treat that timeframe "
            "as potentially delayed and reduce confidence or return NO_TRADE when "
            "freshness makes the trigger unreliable."
        ),

        "primary_feed_live": primary_live,

        "market": {
            "1m": market_metrics(one_rows),
            "5m": market_metrics(five_rows),
            "15m": market_metrics(fifteen_rows),
            "1h_context_only": market_metrics(one_hour_rows),
        },

        "market_sources": {
            "1m": one_min.get("source"),
            "5m": five_min.get("source"),
            "15m": fifteen_min.get("source"),
            "1h_context_only": one_hour.get("source"),
        },

        "market_providers": {
            "1m": one_min.get("provider"),
            "5m": five_min.get("provider"),
            "15m": fifteen_min.get("provider"),
            "1h_context_only": one_hour.get("provider"),
        },

        "market_delayed": {
            "1m": one_min.get("delayed"),
            "5m": five_min.get("delayed"),
            "15m": fifteen_min.get("delayed"),
            "1h_context_only": one_hour.get("delayed"),
        },

        "market_errors": {
            "1m": one_min.get("error") or one_min.get("primary_feed_error"),
            "5m": five_min.get("error") or five_min.get("primary_feed_error"),
            "15m": fifteen_min.get("error") or fifteen_min.get("primary_feed_error"),
            "1h": one_hour.get("error") or one_hour.get("primary_feed_error"),
        },

        "macro_context": macro_data.get("markets", {}),
        "macro_source": macro_data.get("source"),
        "news": headlines,
        "news_source": news_data.get("source"),
        "prediction_performance": perf_data,
    }


# ============================================================
# AI OUTPUT SCHEMA
# ============================================================

AI_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["LONG", "SHORT", "NO_TRADE"]},
        "bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "time_horizon": {"type": "string", "enum": ["1-5m", "5-15m"]},
        "entry_quality": {"type": "string", "enum": ["POOR", "FAIR", "GOOD", "EXCELLENT"]},
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "technical_score": {"type": "integer", "minimum": -10, "maximum": 10},
        "news_score": {"type": "integer", "minimum": -10, "maximum": 10},
        "macro_score": {"type": "integer", "minimum": -10, "maximum": 10},
        "entry_min": {"type": ["number", "null"]},
        "entry_max": {"type": ["number", "null"]},
        "invalidation": {"type": ["number", "null"]},
        "target_1": {"type": ["number", "null"]},
        "target_2": {"type": ["number", "null"]},
        "nearest_support": {"type": ["number", "null"]},
        "nearest_resistance": {"type": ["number", "null"]},
        "summary": {"type": "string"},
        "technical_reason": {"type": "string"},
        "news_reason": {"type": "string"},
        "macro_reason": {"type": "string"},
        "entry_reason": {"type": "string"},
        "what_changes_the_view": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
    },
    "required": [
        "signal", "bias", "confidence", "time_horizon", "entry_quality",
        "risk_level", "technical_score", "news_score", "macro_score",
        "entry_min", "entry_max", "invalidation", "target_1", "target_2",
        "nearest_support", "nearest_resistance", "summary", "technical_reason",
        "news_reason", "macro_reason", "entry_reason", "what_changes_the_view",
    ],
    "additionalProperties": False,
}


def enforce_scalping_rules(analysis):
    if not isinstance(analysis, dict):
        return analysis

    try:
        confidence = int(analysis.get("confidence", 0) or 0)
    except Exception:
        confidence = 0

    confidence = max(0, min(100, confidence))
    analysis["confidence"] = confidence

    original_signal = str(analysis.get("signal", "NO_TRADE")).upper().strip()
    horizon = str(analysis.get("time_horizon", "")).strip()

    if horizon not in {"1-5m", "5-15m"}:
        analysis["time_horizon"] = "5-15m"

    if confidence < 60:
        analysis["signal"] = "NO_TRADE"

        if original_signal == "LONG" and analysis.get("bias") == "NEUTRAL":
            analysis["bias"] = "BULLISH"
        elif original_signal == "SHORT" and analysis.get("bias") == "NEUTRAL":
            analysis["bias"] = "BEARISH"

        reason = str(analysis.get("entry_reason") or "")
        prefix = (
            f"NO_TRADE enforced: confidence {confidence}% is below the "
            "60% minimum for an actionable 1â15 minute trade. "
        )
        if not reason.startswith("NO_TRADE enforced:"):
            analysis["entry_reason"] = prefix + reason

    return analysis


# ============================================================
# OPENAI DAY-TRADING ANALYSIS
# ============================================================

DAY_TRADING_INSTRUCTIONS = """
You are Silver AI, a specialised silver/XAGUSD 1â15 MINUTE SCALPING engine.

Your task is to decide whether there is an ACTIONABLE silver trade RIGHT NOW.

Allowed signals:
LONG
SHORT
NO_TRADE

Allowed horizons:
1-5m
5-15m

HARD DECISION FRAMEWORK

1m = TRIGGER.
5m = CONFIRMATION.
15m = STRUCTURE / ROOM.
1h = CONTEXT ONLY.

A LONG or SHORT is invalid unless BOTH a concrete 1m trigger AND 5m confirmation exist.

LONG requires:
- a concrete bullish 1m event such as SMA20 reclaim, higher-low sequence,
  failed breakdown/reclaim, break of a recent 1m swing high, or bullish impulse
  with follow-through;
AND
- 5m confirmation such as positive momentum, higher-low structure, SMA20 reclaim,
  trend above SMA20, or RSI recovering from oversold.

SHORT requires:
- a concrete bearish 1m event such as SMA20 loss/rejection, lower-high sequence,
  failed breakout/rejection, break of a recent 1m swing low, or bearish impulse
  with follow-through;
AND
- 5m confirmation such as negative momentum, lower-high structure, SMA20 loss,
  trend below SMA20, or RSI rolling down from overbought.

NO_TRADE IS THE DEFAULT.

Return NO_TRADE when:
- no explicit 1m trigger exists,
- 1m and 5m disagree,
- price is choppy,
- price is directly into nearby 15m support/resistance,
- the move has already happened and entry would be chasing,
- target 1 does not offer at least about 1.2R versus invalidation,
- data freshness is questionable,
- confidence is below 60.

CONFIDENCE RULE:
Below 60 = NO_TRADE.
60-69 = only with an explicit trigger, 5m confirmation and acceptable R:R.
70-79 = good trigger/confirmation with one moderate risk.
80+ = unusually clean alignment.

DATA FRESHNESS:
The primary silver feed is Twelve Data XAG/USD.
Do NOT penalise confidence merely because Yahoo is mentioned elsewhere.
Only reduce confidence for feed latency when the actual 1m/5m/15m market provider
shows Yahoo fallback, delayed=true, stale=true, missing candles, or a feed error.

SILVER MACRO CONTEXT

Use supporting markets as context, never as a substitute for the silver trigger:
- Gold (GC=F)
- US Dollar Index (DXY)
- US 10Y yield (^TNX)
- Fresh Fed, inflation, payrolls, Treasury-yield, dollar, geopolitical and metals news.

NEWS

Prioritise fresh market-moving headlines involving silver, gold, the US dollar,
Federal Reserve policy, CPI/PCE/inflation, jobs data, Treasury yields,
geopolitical shocks and major industrial-demand news.

LEVEL DISCIPLINE

nearest_support must be below current silver price.
nearest_resistance must be above current silver price.

For LONG:
invalidation below the setup failure point; targets above entry.

For SHORT:
invalidation above the setup failure point; targets below entry.

Use null when a level cannot be established reliably.

REASONING

technical_reason MUST explicitly state:
1) the exact 1m trigger, or "NO VALID 1m TRIGGER";
2) whether 5m confirms;
3) the important 15m structure/level.

macro_reason MUST explain what gold, DXY and US 10Y are doing and whether that
supports, conflicts with, or is neutral for the silver scalp.

entry_reason must explain why the entry is preferable to chasing.

Return only the requested structured JSON.
"""


def run_ai_analysis():
    if ai_client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")

    snapshot = build_ai_snapshot()

    try:
        response = ai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=DAY_TRADING_INSTRUCTIONS,
            input=(
                "Analyse this current Silver AI snapshot for an immediate "
                "day-trading decision:\n\n"
                + json.dumps(snapshot, default=str)
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "silver_ai_scalp_analysis",
                    "strict": True,
                    "schema": AI_SCHEMA,
                }
            },
        )

        raw = response.output_text

        if not raw:
            raise RuntimeError("OpenAI returned no output text")

        analysis = json.loads(raw)
        analysis = enforce_scalping_rules(analysis)

        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "mode": "SCALPING_1_15M",
            "analysis": analysis,
            "snapshot": snapshot,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {type(exc).__name__}: {exc}",
        )


@app.post("/api/analyse")
def analyse_silver():
    return run_ai_analysis()


@app.get("/api/ai-signal")
def ai_signal():
    return run_ai_analysis()
