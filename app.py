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

app = FastAPI(title="Cocoa AI V1.2")

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

# Keep this configurable in Render later if we want.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()

ai_client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)


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

                # Upgrade an existing older table safely.
                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS technical TEXT
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS weather TEXT
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS entry TEXT
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS price_24h DOUBLE PRECISION
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS return_24h DOUBLE PRECISION
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS result_24h TEXT
                    """
                )

                cur.execute(
                    """
                    ALTER TABLE predictions
                    ADD COLUMN IF NOT EXISTS graded_24h_at TIMESTAMPTZ
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    predictions_created_at_idx
                    ON predictions(created_at DESC)
                    """
                )

    except Exception as exc:
        print(
            "Database table setup warning:",
            type(exc).__name__,
            str(exc),
        )


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
        "service": "cocoa-ai-v1.2",
        "database_configured": bool(DATABASE_URL),
        "database_ok": db_ok,
        "database_error": db_error,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL if OPENAI_API_KEY else None,
    }


# ============================================================
# COCOA MARKET CANDLES
# ============================================================

@app.get("/api/candles")
def candles(
    interval: str = Query(
        "1h",
        pattern="^(5m|15m|30m|60m|1h|1d)$"
    ),
    range: str = Query(
        "1mo",
        pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"
    ),
):

    symbol = "CC=F"

    yahoo_interval = "60m" if interval == "1h" else interval

    url = (
        "https://query1.finance.yahoo.com/"
        "v8/finance/chart/"
        f"{requests.utils.quote(symbol, safe='')}"
    )

    params = {
        "interval": yahoo_interval,
        "range": range,
        "includePrePost": "false",
        "events": "div,splits",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.2)",
        "Accept": "application/json,text/plain,*/*",
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20,
        )

        resp.raise_for_status()
        payload = resp.json()

        chart = payload.get("chart", {})

        if chart.get("error"):
            return {
                "symbol": symbol,
                "source": "Yahoo Finance chart API",
                "delayed": True,
                "interval": interval,
                "range": range,
                "candles": [],
                "error": chart["error"],
            }

        results = chart.get("result") or []

        if not results:
            return {
                "symbol": symbol,
                "source": "Yahoo Finance chart API",
                "delayed": True,
                "interval": interval,
                "range": range,
                "candles": [],
                "error": "Yahoo returned no chart result",
            }

        result = results[0]

        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = indicators.get("quote") or []
        quote = quotes[0] if quotes else {}

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

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

        meta = result.get("meta") or {}

        return {
            "symbol": symbol,
            "source": "Yahoo Finance chart API",
            "delayed": True,
            "interval": interval,
            "range": range,
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName"),
            "regular_market_price": meta.get("regularMarketPrice"),
            "candles": candles_out,
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "source": "Yahoo Finance chart API",
            "delayed": True,
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

WEATHER_CACHE = {
    "timestamp": 0,
    "data": None,
}

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
        and
        now - WEATHER_CACHE["timestamp"] < WEATHER_CACHE_SECONDS
    ):
        cached = dict(WEATHER_CACHE["data"])
        cached["cached"] = True
        cached["cache_age_seconds"] = int(
            now - WEATHER_CACHE["timestamp"]
        )
        return cached

    try:
        keys = list(WEATHER_POINTS.keys())
        coordinates = list(WEATHER_POINTS.values())

        latitudes = ",".join(
            str(lat)
            for lat, lon in coordinates
        )

        longitudes = ",".join(
            str(lon)
            for lat, lon in coordinates
        )

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "daily": "precipitation_sum,temperature_2m_max",
            "timezone": "auto",
            "forecast_days": 7,
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.2)",
            "Accept": "application/json,text/plain,*/*",
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20,
        )

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

            rain = sum(
                float(x or 0)
                for x in rain_values
            )

            valid_temps = [
                float(x)
                for x in temp_values
                if x is not None
            ]

            tmax = max(valid_temps) if valid_temps else 0

            label, color = weather_risk(rain, tmax)

            locations[key] = {
                "rain_7d_mm": round(rain, 2),
                "max_temp_c": round(tmax, 1),
                "risk_label": label,
                "risk_color": color,
            }

        if not locations:
            raise RuntimeError(
                "Open-Meteo returned no usable weather locations"
            )

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

            cached["cache_age_seconds"] = int(
                now - WEATHER_CACHE["timestamp"]
            )

            cached["warning"] = (
                "Fresh weather request failed: "
                f"{type(exc).__name__}: {exc}"
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

NEWS_CACHE = {
    "timestamp": 0,
    "data": None,
}

NEWS_CACHE_SECONDS = 10 * 60


@app.get("/api/news")
def news():

    now = time.time()

    if (
        NEWS_CACHE["data"] is not None
        and
        now - NEWS_CACHE["timestamp"] < NEWS_CACHE_SECONDS
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
        "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.2)",
    }

    try:
        resp = requests.get(
            rss,
            params=params,
            headers=headers,
            timeout=15,
        )

        resp.raise_for_status()

        feed = feedparser.parse(resp.content)

        items = []

        for entry in feed.entries[:12]:

            source = ""

            if (
                hasattr(entry, "source")
                and
                isinstance(entry.source, dict)
            ):
                source = entry.source.get("title", "")

            title = re.sub(
                r"\s+",
                " ",
                getattr(entry, "title", ""),
            ).strip()

            items.append(
                {
                    "title": title,
                    "published_at": getattr(
                        entry,
                        "published",
                        "Recent",
                    ),
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
        raise HTTPException(
            status_code=503,
            detail="Database is not configured",
        )

    prediction = str(
        payload.get(
            "prediction",
            payload.get("pred", "NO TRADE"),
        )
    )

    try:
        confidence = int(
            payload.get(
                "confidence",
                payload.get("conf", 0),
            )
            or 0
        )
    except Exception:
        confidence = 0

    try:
        price = float(
            payload.get("price", 0)
            or 0
        )
    except Exception:
        price = 0

    technical = str(payload.get("technical", "") or "")
    weather_text = str(payload.get("weather", "") or "")
    entry = str(payload.get("entry", "") or "")

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
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
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

        return {
            "ok": True,
            "prediction": row,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Database save failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


@app.get("/api/predictions")
def get_predictions(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
    )
):

    if not DATABASE_URL:
        return {
            "database": False,
            "items": [],
        }

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

        return {
            "database": True,
            "items": rows,
        }

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

                        COUNT(*)
                        FILTER
                        (
                            WHERE graded_24h_at IS NOT NULL
                        )
                        AS graded_24h,

                        COUNT(*)
                        FILTER
                        (
                            WHERE result_24h = 'WIN'
                        )
                        AS wins_24h,

                        COUNT(*)
                        FILTER
                        (
                            WHERE result_24h = 'LOSS'
                        )
                        AS losses_24h,

                        AVG(return_24h)
                        FILTER
                        (
                            WHERE graded_24h_at IS NOT NULL
                        )
                        AS avg_return

                    FROM predictions
                    """
                )

                row = cur.fetchone()

        total = int(row.get("total_predictions", 0) or 0)
        graded = int(row.get("graded_24h", 0) or 0)
        wins = int(row.get("wins_24h", 0) or 0)
        losses = int(row.get("losses_24h", 0) or 0)

        win_rate = (
            round(wins / graded * 100, 2)
            if graded
            else None
        )

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
# TECHNICAL HELPERS FOR AI
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


def market_metrics(candle_rows):

    if not candle_rows:
        return {}

    closes = [
        float(x["c"])
        for x in candle_rows
        if x.get("c") is not None
    ]

    highs = [
        float(x["h"])
        for x in candle_rows
        if x.get("h") is not None
    ]

    lows = [
        float(x["l"])
        for x in candle_rows
        if x.get("l") is not None
    ]

    if len(closes) < 2:
        return {}

    last = closes[-1]
    previous = closes[-2]

    latest_move = (
        ((last - previous) / previous) * 100
        if previous
        else 0
    )

    rsi = calc_rsi(closes)

    sma20 = (
        sum(closes[-20:]) / 20
        if len(closes) >= 20
        else None
    )

    sma50 = (
        sum(closes[-50:]) / 50
        if len(closes) >= 50
        else None
    )

    trend = "neutral"

    if sma20 is not None:
        if last > sma20 * 1.005:
            trend = "bullish"

        elif last < sma20 * 0.995:
            trend = "bearish"

    return {
        "last_price": round(last, 2),
        "latest_move_pct": round(latest_move, 3),

        "rsi_14": (
            round(rsi, 2)
            if rsi is not None
            else None
        ),

        "sma20": (
            round(sma20, 2)
            if sma20 is not None
            else None
        ),

        "sma50": (
            round(sma50, 2)
            if sma50 is not None
            else None
        ),

        "trend_vs_sma20": trend,

        "recent_high": (
            round(max(highs[-30:]), 2)
            if highs
            else None
        ),

        "recent_low": (
            round(min(lows[-30:]), 2)
            if lows
            else None
        ),

        "candle_count": len(closes),
    }


# ============================================================
# AI SNAPSHOT
# ============================================================

def build_ai_snapshot():

    five_min = candles(
        interval="5m",
        range="5d",
    )

    fifteen_min = candles(
        interval="15m",
        range="5d",
    )

    one_hour = candles(
        interval="1h",
        range="1mo",
    )

    daily = candles(
        interval="1d",
        range="6mo",
    )

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

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "important_data_warning": (
            "Yahoo cocoa futures data may be delayed. "
            "Do not assume the final candle equals the user's "
            "current executable broker quote."
        ),

        "market": {
            "5m": market_metrics(
                five_min.get("candles") or []
            ),

            "15m": market_metrics(
                fifteen_min.get("candles") or []
            ),

            "1h": market_metrics(
                one_hour.get("candles") or []
            ),

            "1d": market_metrics(
                daily.get("candles") or []
            ),
        },

        "market_sources": {
            "5m": five_min.get("source"),
            "15m": fifteen_min.get("source"),
            "1h": one_hour.get("source"),
            "1d": daily.get("source"),
        },

        "weather": weather_data.get(
            "locations",
            {}
        ),

        "weather_source": weather_data.get(
            "source"
        ),

        "weather_stale": weather_data.get(
            "stale",
            False
        ),

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
            "enum": [
                "LONG",
                "SHORT",
                "NO_TRADE"
            ]
        },

        "bias": {
            "type": "string",
            "enum": [
                "BULLISH",
                "BEARISH",
                "NEUTRAL"
            ]
        },

        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        },

        "time_horizon": {
            "type": "string"
        },

        "entry_quality": {
            "type": "string",
            "enum": [
                "POOR",
                "FAIR",
                "GOOD",
                "EXCELLENT"
            ]
        },

        "risk_level": {
            "type": "string",
            "enum": [
                "LOW",
                "MEDIUM",
                "HIGH"
            ]
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
            "type": [
                "number",
                "null"
            ]
        },

        "entry_max": {
            "type": [
                "number",
                "null"
            ]
        },

        "invalidation": {
            "type": [
                "number",
                "null"
            ]
        },

        "target_1": {
            "type": [
                "number",
                "null"
            ]
        },

        "target_2": {
            "type": [
                "number",
                "null"
            ]
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

            "items": {
                "type": "string"
            },

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
        "summary",
        "technical_reason",
        "news_reason",
        "weather_reason",
        "entry_reason",
        "what_changes_the_view"
    ],

    "additionalProperties": False
}


# ============================================================
# OPENAI ANALYSIS
# ============================================================

@app.post("/api/analyse")
def analyse_cocoa():

    if ai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured"
        )

    snapshot = build_ai_snapshot()

    instructions = """
You are Cocoa AI, a specialised cocoa-futures analysis engine.

You analyse cocoa futures only from the evidence supplied to you.

The purpose is analytical decision support.
Do not claim certainty or guaranteed profit.

EVIDENCE PRIORITY

1. Actual price action and market structure.
2. Agreement or disagreement across timeframes.
3. Technical momentum.
4. Cocoa-specific news and supply/demand information.
5. West African cocoa-belt weather.
6. Historical model performance, but only when enough graded observations exist.

CORE RULES

- Never invent missing facts.
- Never invent news.
- Never invent prices.
- Never invent weather observations.
- Missing data reduces confidence.
- Stale data reduces confidence.

- A bullish directional bias does not automatically mean LONG.
- A bearish directional bias does not automatically mean SHORT.

- Separate direction from entry quality.

Example:
A market can be bearish but already too extended to short.
In that case signal should be NO_TRADE with BEARISH bias.

- Do not force trades.
- NO_TRADE is a valid and desirable result when evidence is conflicted.
- Avoid chasing large already-completed moves.
- Prefer pullbacks, rejection, breakout/retest and clean confirmation.

NEWS RULES

Interpret headlines in the context of cocoa.

Potentially bullish cocoa examples:
- crop loss
- disease
- drought
- excessive rain
- heat damage
- lower production
- lower arrivals
- supply shortage
- political/logistical disruption

Potentially bearish cocoa examples:
- improving production
- strong arrivals
- better crop estimates
- improved weather
- demand destruction
- weak grindings
- excess supply

Do not blindly classify a headline from one word.

Repeated information may already be priced in.

WEATHER RULES

Interpret weather only in terms of cocoa production.

Dry risk can be bullish if it threatens pods or crop development.

Excessive rain can also be bullish if it increases disease,
harvest or drying problems.

Normal weather should not automatically create a bullish score.

TECHNICAL RULES

Technical score:
-10 = strongly bearish
0 = neutral
+10 = strongly bullish

News score:
-10 = strongly bearish
0 = neutral
+10 = strongly bullish

Weather score:
-10 = strongly bearish cocoa
0 = neutral
+10 = strongly bullish cocoa

IMPORTANT PRICE DATA LIMITATION

The supplied Yahoo cocoa data can be delayed relative to a broker's
current executable quote.

Therefore:
- do not claim ultra-short-term precision,
- reduce confidence when the last price may be stale,
- do not create overly precise entry prices without sufficient support.

OUTPUT

Return the requested structured JSON only.

Keep all explanations concise and evidence-based.
"""

    try:

        response = ai_client.responses.create(
            model=OPENAI_MODEL,

            instructions=instructions,

            input=(
                "Analyse this current Cocoa AI market snapshot:\n\n"
                + json.dumps(
                    snapshot,
                    default=str,
                )
            ),

            text={
                "format": {
                    "type": "json_schema",
                    "name": "cocoa_ai_analysis",
                    "strict": True,
                    "schema": AI_SCHEMA,
                }
            },
        )

        raw = response.output_text

        if not raw:
            raise RuntimeError(
                "OpenAI returned no output text"
            )

        result = json.loads(raw)

        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "analysis": result,
            "snapshot": snapshot,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI analysis failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
