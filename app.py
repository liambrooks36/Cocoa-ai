from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import time
import re

import requests
import feedparser
import psycopg
from psycopg.rows import dict_row


# ============================================================
# APP SETUP
# ============================================================

ROOT = Path(__file__).parent

app = FastAPI(title="Cocoa AI V1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


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
    """
    Creates the prediction table if it does not exist.
    Safe to run repeatedly.
    """

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
        "service": "cocoa-ai-v1.1",
        "database_configured": bool(DATABASE_URL),
        "database_ok": db_ok,
        "database_error": db_error,
    }


# ============================================================
# HELPERS
# ============================================================

def clean_num(v):
    try:
        if v is None:
            return None

        return float(v)

    except (TypeError, ValueError):
        return None


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

    """
    Fetch delayed cocoa futures data from Yahoo Finance.

    Symbol:
    CC=F

    4-hour candles are created in the frontend from 1-hour data.
    """

    symbol = "CC=F"

    yahoo_interval = (
        "60m"
        if interval == "1h"
        else interval
    )

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
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; CocoaAI/1.1)"
        ),
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


# Refresh weather only every 30 minutes.
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

    # --------------------------------------------------------
    # RETURN FRESH CACHE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FETCH WEATHER FROM OPEN-METEO
    # --------------------------------------------------------

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
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; CocoaAI/1.1)"
            ),
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

        if isinstance(payload, list):
            results = payload
        else:
            results = [payload]

        locations = {}

        for index, key in enumerate(keys):

            if index >= len(results):
                continue

            result = results[index]

            daily = result.get("daily") or {}

            rain_values = (
                daily.get("precipitation_sum")
                or []
            )

            temp_values = (
                daily.get("temperature_2m_max")
                or []
            )

            rain = sum(
                float(x or 0)
                for x in rain_values
            )

            valid_temps = [
                float(x)
                for x in temp_values
                if x is not None
            ]

            if valid_temps:
                tmax = max(valid_temps)
            else:
                tmax = 0

            label, color = weather_risk(
                rain,
                tmax,
            )

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

        # If Open-Meteo temporarily fails but we have
        # previously fetched data, keep serving that data.

        if WEATHER_CACHE["data"] is not None:

            cached = dict(
                WEATHER_CACHE["data"]
            )

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
# COCOA NEWS
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

        cached = dict(
            NEWS_CACHE["data"]
        )

        cached["cached"] = True

        return cached


    query = (
        'cocoa OR cacao Ghana '
        'OR "Ivory Coast"'
    )

    rss = (
        "https://news.google.com/"
        "rss/search"
    )

    params = {
        "q": query,
        "hl": "en-GB",
        "gl": "GB",
        "ceid": "GB:en",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; CocoaAI/1.1)"
        ),
    }

    try:

        resp = requests.get(
            rss,
            params=params,
            headers=headers,
            timeout=15,
        )

        resp.raise_for_status()

        feed = feedparser.parse(
            resp.content
        )

        items = []

        for entry in feed.entries[:12]:

            source = ""

            if (
                hasattr(entry, "source")
                and isinstance(
                    entry.source,
                    dict,
                )
            ):
                source = entry.source.get(
                    "title",
                    "",
                )

            title = re.sub(
                r"\s+",
                " ",
                getattr(
                    entry,
                    "title",
                    "",
                ),
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
                    "link": getattr(
                        entry,
                        "link",
                        "",
                    ),
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

            cached = dict(
                NEWS_CACHE["data"]
            )

            cached["cached"] = True
            cached["stale"] = True

            return cached


        return {
            "source": "Google News RSS",
            "items": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# PREDICTIONS — SAVE TO SUPABASE POSTGRES
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
            payload.get(
                "pred",
                "NO TRADE",
            ),
        )
    )

    try:
        confidence = int(
            payload.get(
                "confidence",
                payload.get(
                    "conf",
                    0,
                ),
            )
            or 0
        )
    except Exception:
        confidence = 0


    try:
        price = float(
            payload.get(
                "price",
                0,
            )
            or 0
        )
    except Exception:
        price = 0


    technical = str(
        payload.get(
            "technical",
            "",
        )
        or ""
    )

    weather_text = str(
        payload.get(
            "weather",
            "",
        )
        or ""
    )

    entry = str(
        payload.get(
            "entry",
            "",
        )
        or ""
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


# ============================================================
# GET SAVED PREDICTIONS
# ============================================================

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
# PERFORMANCE SUMMARY
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


        total = int(
            row.get(
                "total_predictions",
                0,
            )
            or 0
        )

        graded = int(
            row.get(
                "graded_24h",
                0,
            )
            or 0
        )

        wins = int(
            row.get(
                "wins_24h",
                0,
            )
            or 0
        )

        losses = int(
            row.get(
                "losses_24h",
                0,
            )
            or 0
        )


        win_rate = (
            round(
                wins / graded * 100,
                2,
            )
            if graded
            else None
        )


        avg_return = (
            float(
                row["avg_return"]
            )
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
