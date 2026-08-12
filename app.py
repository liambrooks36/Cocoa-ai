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
app = FastAPI(title="Cocoa AI V1.6.1.1")

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
                        entry TEXT,
                        price_24h DOUBLE PRECISION,
                        return_24h DOUBLE PRECISION,
                        result_24h TEXT,
                        graded_24h_at TIMESTAMPTZ
                    )
                    """
                )

                # Safe upgrades for existing older tables.
                for sql in (
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS technical TEXT",
                    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS weather TEXT",
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
        "service": "cocoa-ai-v1.6.1.1",
        "database_configured": bool(DATABASE_URL),
        "database_ok": db_ok,
        "database_error": db_error,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL if OPENAI_API_KEY else None,
        "analysis_mode": "DAY_TRADING",
    }


# ============================================================
# COCOA MARKET CANDLES
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
        "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.6.1)",
        "Accept": "application/json,text/plain,*/*",
        "Cache-Control": "no-cache",
    }

    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = (
            f"https://{host}/v8/finance/chart/"
            f"{requests.utils.quote(symbol, safe='')}"
        )

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


@app.get("/api/candles")
def candles(
    interval: str = Query(
        "1h",
        pattern="^(1m|5m|15m|30m|60m|1h|1d)$"
    ),
    range: str = Query(
        "1mo",
        pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"
    ),
):
    symbol = "CC=F"

    cached = _cached_candles(interval, range)
    if cached is not None:
        return cached

    yahoo_interval = "60m" if interval == "1h" else interval

    try:
        result, yahoo_host = _fetch_yahoo_chart(symbol, yahoo_interval, range)

        timestamps = result.get("timestamp") or []
        quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])

        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        candles_out = []

        for i, ts in enumerate(timestamps):
            try:
                o = opens[i] if i < len(opens) else None
                h = highs[i] if i < len(highs) else None
                l = lows[i] if i < len(lows) else None
                c = closes[i] if i < len(closes) else None
                v = volumes[i] if i < len(volumes) else 0

                if None in (o, h, l, c):
                    continue

                candles_out.append(
                    {
                        "t": int(ts),
                        "o": float(o),
                        "h": float(h),
                        "l": float(l),
                        "c": float(c),
                        "v": float(v or 0),
                    }
                )

            except (TypeError, ValueError, IndexError):
                continue

        if not candles_out:
            raise RuntimeError("Yahoo returned zero usable OHLC candles")

        meta = result.get("meta") or {}

        data = {
            "symbol": symbol,
            "source": f"Yahoo Finance chart API ({yahoo_host})",
            "delayed": True,
            "cached": False,
            "stale": False,
            "interval": interval,
            "range": range,
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName"),
            "regular_market_price": meta.get("regularMarketPrice"),
            "candles": candles_out,
        }

        CANDLE_CACHE[(interval, range)] = {
            "timestamp": time.time(),
            "data": data,
        }

        return data

    except Exception as exc:
        stale = _stale_cached_candles(interval, range)
        if stale is not None:
            stale["warning"] = (
                f"Fresh Yahoo request failed: {type(exc).__name__}: {exc}"
            )
            return stale

        return {
            "symbol": symbol,
            "source": "Yahoo Finance chart API",
            "delayed": True,
            "cached": False,
            "stale": False,
            "interval": interval,
            "range": range,
            "candles": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# WEATHER
# ============================================================

WEATHER_POINTS = {
    "san_pedro": (4.7485, -6.6363),
    "daloa": (6.8774, -6.4502),
    "kumasi": (6.6885, -1.6244),
    "sunyani": (7.3399, -2.3268),
}

WEATHER_CACHE = {"timestamp": 0, "data": None}
WEATHER_CACHE_SECONDS = 30 * 60


def weather_risk(rain, tmax):
    if tmax >= 34:
        return "HEAT RISK", "#ff4f45"
    if rain < 15:
        return "DRY RISK", "#ffbd2e"
    if rain > 110:
        return "WET RISK", "#ffbd2e"
    return "NORMAL", "#27d45c"


@app.get("/api/weather")
def weather():
    now = time.time()

    if (
        WEATHER_CACHE["data"] is not None
        and now - WEATHER_CACHE["timestamp"] < WEATHER_CACHE_SECONDS
    ):
        cached = dict(WEATHER_CACHE["data"])
        cached["cached"] = True
        cached["cache_age_seconds"] = int(now - WEATHER_CACHE["timestamp"])
        return cached

    try:
        keys = list(WEATHER_POINTS.keys())
        coordinates = list(WEATHER_POINTS.values())

        latitudes = ",".join(str(lat) for lat, lon in coordinates)
        longitudes = ",".join(str(lon) for lat, lon in coordinates)

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "daily": "precipitation_sum,temperature_2m_max",
            "timezone": "auto",
            "forecast_days": 7,
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.6.1.1)",
            "Accept": "application/json,text/plain,*/*",
        }

        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()

        results = payload if isinstance(payload, list) else [payload]
        locations = {}

        for index, key in enumerate(keys):
            if index >= len(results):
                continue

            result = results[index]
            daily = result.get("daily") or {}

            rain_values = daily.get("precipitation_sum") or []
            temp_values = daily.get("temperature_2m_max") or []

            rain = sum(float(x or 0) for x in rain_values)
            valid_temps = [float(x) for x in temp_values if x is not None]
            tmax = max(valid_temps) if valid_temps else 0

            label, color = weather_risk(rain, tmax)

            locations[key] = {
                "rain_7d_mm": round(rain, 2),
                "max_temp_c": round(tmax, 1),
                "risk_label": label,
                "risk_color": color,
            }

        if not locations:
            raise RuntimeError("Open-Meteo returned no usable weather locations")

        data = {
            "source": "Open-Meteo",
            "cached": False,
            "stale": False,
            "cache_age_seconds": 0,
            "locations": locations,
        }

        WEATHER_CACHE["timestamp"] = now
        WEATHER_CACHE["data"] = data

        return data

    except Exception as exc:
        if WEATHER_CACHE["data"] is not None:
            cached = dict(WEATHER_CACHE["data"])
            cached["cached"] = True
            cached["stale"] = True
            cached["cache_age_seconds"] = int(now - WEATHER_CACHE["timestamp"])
            cached["warning"] = (
                f"Fresh weather request failed: {type(exc).__name__}: {exc}"
            )
            return cached

        return {
            "source": "Open-Meteo",
            "cached": False,
            "stale": False,
            "locations": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# NEWS
# ============================================================

NEWS_CACHE = {"timestamp": 0, "data": None}
NEWS_CACHE_SECONDS = 10 * 60


@app.get("/api/news")
def news():
    now = time.time()

    if (
        NEWS_CACHE["data"] is not None
        and now - NEWS_CACHE["timestamp"] < NEWS_CACHE_SECONDS
    ):
        cached = dict(NEWS_CACHE["data"])
        cached["cached"] = True
        return cached

    query = 'cocoa OR cacao Ghana OR "Ivory Coast"'
    rss = "https://news.google.com/rss/search"

    params = {
        "q": query,
        "hl": "en-GB",
        "gl": "GB",
        "ceid": "GB:en",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.6.1.1)",
    }

    try:
        resp = requests.get(rss, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        items = []

        for entry in feed.entries[:12]:
            source = ""

            if hasattr(entry, "source") and isinstance(entry.source, dict):
                source = entry.source.get("title", "")

            title = re.sub(
                r"\s+",
                " ",
                getattr(entry, "title", ""),
            ).strip()

            items.append(
                {
                    "title": title,
                    "published_at": getattr(entry, "published", "Recent"),
                    "source": source or "Google News",
                    "link": getattr(entry, "link", ""),
                }
            )

        data = {
            "source": "Google News RSS",
            "cached": False,
            "items": items,
        }

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

    technical = str(
        payload.get("technical", payload.get("technical_score", "")) or ""
    )

    weather_text = str(
        payload.get("weather", payload.get("weather_score", "")) or ""
    )

    entry = str(
        payload.get("entry", payload.get("entry_quality", "")) or ""
    )

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                    (
                        prediction,
                        confidence,
                        price,
                        technical,
                        weather,
                        entry
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING
                        id,
                        created_at,
                        prediction,
                        confidence,
                        price,
                        technical,
                        weather,
                        entry,
                        price_24h,
                        return_24h,
                        result_24h,
                        graded_24h_at
                    """,
                    (
                        prediction,
                        confidence,
                        price,
                        technical,
                        weather_text,
                        entry,
                    ),
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
                    SELECT
                        id,
                        created_at,
                        prediction,
                        confidence,
                        price,
                        technical,
                        weather,
                        entry,
                        price_24h,
                        return_24h,
                        result_24h,
                        graded_24h_at
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
                        AVG(return_24h)
                            FILTER (WHERE graded_24h_at IS NOT NULL)
                            AS avg_return
                    FROM predictions
                    """
                )

                row = cur.fetchone()

        total = int(row.get("total_predictions", 0) or 0)
        graded = int(row.get("graded_24h", 0) or 0)
        wins = int(row.get("wins_24h", 0) or 0)
        losses = int(row.get("losses_24h", 0) or 0)

        win_rate = round(wins / graded * 100, 2) if graded else None

        avg_return = (
            float(row["avg_return"])
            if row.get("avg_return") is not None
            else None
        )

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

        out.append(
            {
                "t": group[0].get("t"),
                "o": group[0]["o"],
                "h": max(x["h"] for x in group),
                "l": min(x["l"] for x in group),
                "c": group[-1]["c"],
                "v": sum(float(x.get("v") or 0) for x in group),
            }
        )

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
    opens = [float(x["o"]) for x in rows]
    highs = [float(x["h"]) for x in rows]
    lows = [float(x["l"]) for x in rows]
    volumes = [float(x.get("v") or 0) for x in rows]

    last = closes[-1]
    previous = closes[-2]
    latest_move = ((last - previous) / previous) * 100 if previous else 0
    rsi = calc_rsi(closes)

    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None

    move_3 = (
        ((last - closes[-4]) / closes[-4]) * 100
        if len(closes) >= 4 and closes[-4]
        else None
    )
    move_5 = (
        ((last - closes[-6]) / closes[-6]) * 100
        if len(closes) >= 6 and closes[-6]
        else None
    )
    move_10 = (
        ((last - closes[-11]) / closes[-11]) * 100
        if len(closes) >= 11 and closes[-11]
        else None
    )

    trend = "neutral"
    if sma20 is not None:
        # Small threshold because this is a scalp engine.
        if last > sma20 * 1.0015:
            trend = "bullish"
        elif last < sma20 * 0.9985:
            trend = "bearish"

    avg_volume_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None

    last3 = rows[-3:] if len(rows) >= 3 else rows
    green_last3 = sum(1 for x in last3 if float(x["c"]) > float(x["o"]))
    red_last3 = sum(1 for x in last3 if float(x["c"]) < float(x["o"]))

    recent8 = rows[-8:]
    green_last8 = sum(1 for x in recent8 if float(x["c"]) > float(x["o"]))
    red_last8 = sum(1 for x in recent8 if float(x["c"]) < float(x["o"]))

    higher_lows = len(rows) >= 3 and lows[-1] > lows[-2] > lows[-3]
    lower_highs = len(rows) >= 3 and highs[-1] < highs[-2] < highs[-3]

    prior_high_10 = (
        max(highs[-11:-1])
        if len(highs) >= 11
        else max(highs[:-1])
    )
    prior_low_10 = (
        min(lows[-11:-1])
        if len(lows) >= 11
        else min(lows[:-1])
    )

    breakout_up = last > prior_high_10
    breakout_down = last < prior_low_10

    sma20_reclaim_up = (
        sma20 is not None
        and previous <= sma20
        and last > sma20
    )
    sma20_reclaim_down = (
        sma20 is not None
        and previous >= sma20
        and last < sma20
    )

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
        "volume_vs_avg20": (
            round(volumes[-1] / avg_volume_20, 2)
            if avg_volume_20
            else None
        ),
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

    weather_data = weather()
    news_data = news()
    perf_data = performance()

    headlines = []
    for item in (news_data.get("items") or [])[:10]:
        headlines.append(
            {
                "title": item.get("title"),
                "published_at": item.get("published_at"),
                "source": item.get("source"),
            }
        )

    return {
        "asset": "ICE Cocoa Futures",
        "symbol": "CC=F",
        "analysis_style": "SCALPING_1_TO_15_MINUTES",
        "primary_prediction_window": "1 to 15 minutes",
        "maximum_trade_horizon": "15 minutes",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),

        "important_data_warning": (
            "Yahoo cocoa futures data may be delayed relative to the user's broker quote. "
            "For a 1â15 minute signal, stale data is a major limitation. Reduce confidence "
            "or return NO_TRADE if freshness makes the trigger unreliable."
        ),

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

        "market_errors": {
            "1m": one_min.get("error"),
            "5m": five_min.get("error"),
            "15m": fifteen_min.get("error"),
            "1h": one_hour.get("error"),
        },

        "weather": weather_data.get("locations", {}),
        "weather_source": weather_data.get("source"),
        "weather_stale": weather_data.get("stale", False),

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
        "signal": {
            "type": "string",
            "enum": ["LONG", "SHORT", "NO_TRADE"]
        },
        "bias": {
            "type": "string",
            "enum": ["BULLISH", "BEARISH", "NEUTRAL"]
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        },
        "time_horizon": {
            "type": "string",
            "enum": ["1-5m", "5-15m"]
        },
        "entry_quality": {
            "type": "string",
            "enum": ["POOR", "FAIR", "GOOD", "EXCELLENT"]
        },
        "risk_level": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"]
        },
        "technical_score": {
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        },
        "news_score": {
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        },
        "weather_score": {
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        },
        "entry_min": {
            "type": ["number", "null"]
        },
        "entry_max": {
            "type": ["number", "null"]
        },
        "invalidation": {
            "type": ["number", "null"]
        },
        "target_1": {
            "type": ["number", "null"]
        },
        "target_2": {
            "type": ["number", "null"]
        },
        "nearest_support": {
            "type": ["number", "null"]
        },
        "nearest_resistance": {
            "type": ["number", "null"]
        },
        "summary": {
            "type": "string"
        },
        "technical_reason": {
            "type": "string"
        },
        "news_reason": {
            "type": "string"
        },
        "weather_reason": {
            "type": "string"
        },
        "entry_reason": {
            "type": "string"
        },
        "what_changes_the_view": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5
        }
    },
    "required": [
        "signal",
        "bias",
        "confidence",
        "time_horizon",
        "entry_quality",
        "risk_level",
        "technical_score",
        "news_score",
        "weather_score",
        "entry_min",
        "entry_max",
        "invalidation",
        "target_1",
        "target_2",
        "nearest_support",
        "nearest_resistance",
        "summary",
        "technical_reason",
        "news_reason",
        "weather_reason",
        "entry_reason",
        "what_changes_the_view"
    ],
    "additionalProperties": False
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
You are Cocoa AI, a specialised ICE cocoa-futures 1â15 MINUTE SCALPING engine.

Your task is to decide whether there is an ACTIONABLE trade RIGHT NOW.

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
  trend above SMA20, or RSI RECOVERING from oversold.

SHORT requires:
- a concrete bearish 1m event such as SMA20 loss/rejection, lower-high sequence,
  failed breakout/rejection, break of a recent 1m swing low, or bearish impulse
  with follow-through;
AND
- 5m confirmation such as negative momentum, lower-high structure, SMA20 loss,
  trend below SMA20, or RSI ROLLING DOWN from overbought.

IMPORTANT:
5m being oversold is NOT a long trigger.
5m being overbought is NOT a short trigger.
15m direction alone is NOT a trigger.
1h direction alone is NOT a trigger.

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
Below 60 = NO_TRADE. This is a hard rule.
60-69 = only with an explicit trigger, 5m confirmation and acceptable R:R.
70-79 = good trigger/confirmation with one moderate risk.
80+ = unusually clean alignment.

NEWS / WEATHER:
For a 1â15 minute trade, fresh price action dominates.
Old supply/policy headlines are background only.
Normal weather is background only.
Only genuinely fresh market-moving cocoa news or an acute new crop event
should materially alter an immediate scalp.

LEVEL DISCIPLINE:
nearest_support must be below current price.
nearest_resistance must be above current price.
For LONG: invalidation below the setup failure point; targets above entry.
For SHORT: invalidation above the setup failure point; targets below entry.
Do not copy targets into support/resistance fields.
Use null when a level cannot be established reliably.

REASONING:
technical_reason MUST explicitly say:
1) the exact 1m trigger, or "NO VALID 1m TRIGGER";
2) whether 5m confirms;
3) the important 15m structure/level.

entry_reason must explain why this entry is preferable to chasing.

Yahoo may be delayed relative to the broker quote. If that makes the 1m trigger
unreliable, reduce confidence or return NO_TRADE.

Return only the requested structured JSON.
"""


def run_ai_analysis():
    if ai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured"
        )

    snapshot = build_ai_snapshot()

    try:
        response = ai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=DAY_TRADING_INSTRUCTIONS,
            input=(
                "Analyse this current Cocoa AI snapshot for an immediate "
                "day-trading decision:\n\n"
                + json.dumps(snapshot, default=str)
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "cocoa_ai_day_trade_analysis",
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
def analyse_cocoa():
    return run_ai_analysis()


# Handy GET endpoint for the website dashboard.
@app.get("/api/ai-signal")
def ai_signal():
    return run_ai_analysis()
