from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, re, json
import requests
import feedparser
import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).parent
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="Cocoa AI V1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def init_db():
    if not DATABASE_URL:
        print("DATABASE_URL not configured; database features disabled.")
        return
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        prediction TEXT NOT NULL,
                        directional_bias TEXT,
                        confidence INTEGER NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
                        price DOUBLE PRECISION NOT NULL,
                        technical_score TEXT,
                        weather_score TEXT,
                        entry_quality TEXT,
                        timeframe TEXT,
                        reasoning TEXT,
                        feature_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_outcomes (
                        id BIGSERIAL PRIMARY KEY,
                        prediction_id BIGINT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
                        horizon TEXT NOT NULL,
                        target_time TIMESTAMPTZ NOT NULL,
                        graded_at TIMESTAMPTZ,
                        future_price DOUBLE PRECISION,
                        return_pct DOUBLE PRECISION,
                        result TEXT,
                        UNIQUE(prediction_id, horizon)
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                    ON predictions(created_at DESC);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_outcomes_target_time
                    ON prediction_outcomes(target_time);
                """)
            conn.commit()
        print("Database initialized.")
    except Exception as exc:
        print(f"Database init failed: {type(exc).__name__}: {exc}")

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")

@app.get("/health")
def health():
    db_ok = False
    db_error = None
    if DATABASE_URL:
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    db_ok = cur.fetchone()["ok"] == 1
        except Exception as exc:
            db_error = f"{type(exc).__name__}: {exc}"
    return {
        "ok": True,
        "service": "cocoa-ai-v1.1",
        "database_configured": bool(DATABASE_URL),
        "database_ok": db_ok,
        "database_error": db_error,
    }

def yahoo_chart(interval: str = "1h", range_: str = "1mo"):
    symbol = "CC=F"
    yahoo_interval = "60m" if interval == "1h" else interval

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe='')}"
    params = {
        "interval": yahoo_interval,
        "range": range_,
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.1)",
        "Accept": "application/json,text/plain,*/*",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(str(chart["error"]))

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo returned no chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [])
    quote = quotes[0] if quotes else {}

    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    candles = []
    for i, ts in enumerate(timestamps):
        try:
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            l = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else 0
            if None in (o, h, l, c):
                continue
            candles.append({
                "t": int(ts),
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "v": float(v or 0),
            })
        except (TypeError, ValueError, IndexError):
            continue

    return result.get("meta") or {}, candles

@app.get("/api/candles")
def candles(
    interval: str = Query("1h", pattern="^(5m|15m|30m|60m|1h|1d)$"),
    range: str = Query("1mo", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$"),
):
    try:
        meta, candles_out = yahoo_chart(interval, range)
        return {
            "symbol": "CC=F",
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
            "symbol": "CC=F",
            "source": "Yahoo Finance chart API",
            "delayed": True,
            "interval": interval,
            "range": range,
            "candles": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

def cocoa_price_near(target_dt: datetime):
    meta, candles = yahoo_chart("1h", "1mo")
    if not candles:
        raise RuntimeError("No candle history available")
    target_ts = int(target_dt.timestamp())
    nearest = min(candles, key=lambda c: abs(c["t"] - target_ts))
    if abs(nearest["t"] - target_ts) > 6 * 3600:
        raise RuntimeError("No candle close near requested grading time")
    return float(nearest["c"])

WEATHER_POINTS = {
    "san_pedro": (4.7485, -6.6363),
    "daloa": (6.8774, -6.4502),
    "kumasi": (6.6885, -1.6244),
    "sunyani": (7.3399, -2.3268),
}

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
    locs = {}
    for key, (lat, lon) in WEATHER_POINTS.items():
        u = "https://api.open-meteo.com/v1/forecast"
        p = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max",
            "timezone": "auto",
            "forecast_days": 7,
        }
        r = requests.get(u, params=p, timeout=15)
        r.raise_for_status()
        j = r.json()
        rain = sum(x or 0 for x in j["daily"]["precipitation_sum"])
        tmax = max(j["daily"]["temperature_2m_max"])
        label, color = weather_risk(rain, tmax)
        locs[key] = {
            "rain_7d_mm": float(rain),
            "max_temp_c": float(tmax),
            "risk_label": label,
            "risk_color": color,
        }
    return {"source": "Open-Meteo", "locations": locs}

@app.get("/api/news")
def news():
    query = 'cocoa OR cacao Ghana OR "Ivory Coast"'
    rss = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"}
    resp = requests.get(rss, params=params, timeout=15)
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)

    items = []
    for e in feed.entries[:12]:
        src = ""
        if hasattr(e, "source") and isinstance(e.source, dict):
            src = e.source.get("title", "")
        items.append({
            "title": re.sub(r"\s+", " ", getattr(e, "title", "")).strip(),
            "published_at": getattr(e, "published", "Recent"),
            "source": src or "Google News",
            "link": getattr(e, "link", ""),
        })
    return {"source": "Google News RSS", "items": items}

class PredictionCreate(BaseModel):
    prediction: str = Field(min_length=1, max_length=80)
    directional_bias: str | None = Field(default=None, max_length=20)
    confidence: int = Field(ge=0, le=100)
    price: float = Field(gt=0)
    technical_score: str | None = Field(default=None, max_length=30)
    weather_score: str | None = Field(default=None, max_length=30)
    entry_quality: str | None = Field(default=None, max_length=80)
    timeframe: str | None = Field(default=None, max_length=30)
    reasoning: str | None = None
    feature_snapshot: dict = Field(default_factory=dict)

HORIZONS = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "24h": timedelta(hours=24),
    "3d": timedelta(days=3),
    "7d": timedelta(days=7),
}

@app.post("/api/predictions")
def create_prediction(p: PredictionCreate):
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    prediction, directional_bias, confidence, price,
                    technical_score, weather_score, entry_quality,
                    timeframe, reasoning, feature_snapshot
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                RETURNING id, created_at
                """,
                (
                    p.prediction,
                    p.directional_bias,
                    p.confidence,
                    p.price,
                    p.technical_score,
                    p.weather_score,
                    p.entry_quality,
                    p.timeframe,
                    p.reasoning,
                    json.dumps(p.feature_snapshot),
                ),
            )
            row = cur.fetchone()
            created_at = row["created_at"]

            for horizon, delta in HORIZONS.items():
                cur.execute(
                    """
                    INSERT INTO prediction_outcomes (
                        prediction_id, horizon, target_time
                    )
                    VALUES (%s,%s,%s)
                    ON CONFLICT (prediction_id, horizon) DO NOTHING
                    """,
                    (row["id"], horizon, created_at + delta),
                )
        conn.commit()

    return {"ok": True, "id": row["id"], "created_at": created_at}

@app.get("/api/predictions")
def list_predictions(limit: int = Query(50, ge=1, le=500)):
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.*,
                    COALESCE(
                        json_agg(
                            json_build_object(
                                'horizon', o.horizon,
                                'target_time', o.target_time,
                                'graded_at', o.graded_at,
                                'future_price', o.future_price,
                                'return_pct', o.return_pct,
                                'result', o.result
                            )
                            ORDER BY o.target_time
                        ) FILTER (WHERE o.id IS NOT NULL),
                        '[]'::json
                    ) AS outcomes
                FROM predictions p
                LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id
                GROUP BY p.id
                ORDER BY p.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return {"items": rows}

def grade_direction(prediction: str, bias: str | None, ret: float):
    text = f"{prediction or ''} {bias or ''}".upper()
    if "NO TRADE" in text:
        return "neutral"
    if "LONG" in text or "BULL" in text:
        if ret > 0.15:
            return "correct"
        if ret < -0.15:
            return "incorrect"
        return "neutral"
    if "SHORT" in text or "BEAR" in text:
        if ret < -0.15:
            return "correct"
        if ret > 0.15:
            return "incorrect"
        return "neutral"
    return "neutral"

@app.post("/api/grade")
def grade_due_predictions():
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

    now = datetime.now(timezone.utc)
    graded = 0
    skipped = 0
    errors = []

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    o.id AS outcome_id,
                    o.target_time,
                    p.id AS prediction_id,
                    p.prediction,
                    p.directional_bias,
                    p.price AS start_price
                FROM prediction_outcomes o
                JOIN predictions p ON p.id = o.prediction_id
                WHERE o.graded_at IS NULL
                  AND o.target_time <= %s
                ORDER BY o.target_time ASC
                LIMIT 100
                """,
                (now,),
            )
            due = cur.fetchall()

            for row in due:
                try:
                    future_price = cocoa_price_near(row["target_time"])
                    ret = ((future_price - row["start_price"]) / row["start_price"]) * 100.0
                    result = grade_direction(row["prediction"], row["directional_bias"], ret)
                    cur.execute(
                        """
                        UPDATE prediction_outcomes
                        SET graded_at=%s,
                            future_price=%s,
                            return_pct=%s,
                            result=%s
                        WHERE id=%s
                        """,
                        (now, future_price, ret, result, row["outcome_id"]),
                    )
                    graded += 1
                except Exception as exc:
                    skipped += 1
                    errors.append({
                        "prediction_id": row["prediction_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        conn.commit()

    return {"ok": True, "graded": graded, "skipped": skipped, "errors": errors[:10]}

@app.get("/api/performance")
def performance():
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM predictions")
            total_predictions = cur.fetchone()["n"]

            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE result='correct') AS correct,
                    COUNT(*) FILTER (WHERE result='incorrect') AS incorrect,
                    COUNT(*) FILTER (WHERE result='neutral') AS neutral,
                    AVG(return_pct) FILTER (WHERE graded_at IS NOT NULL) AS avg_return
                FROM prediction_outcomes
                WHERE horizon='24h'
            """)
            stats = cur.fetchone()

    directional = (stats["correct"] or 0) + (stats["incorrect"] or 0)
    win_rate = None
    if directional > 0:
        win_rate = (stats["correct"] / directional) * 100.0

    return {
        "total_predictions": total_predictions,
        "graded_24h_correct": stats["correct"] or 0,
        "graded_24h_incorrect": stats["incorrect"] or 0,
        "graded_24h_neutral": stats["neutral"] or 0,
        "win_rate_24h": win_rate,
        "avg_return_24h": float(stats["avg_return"]) if stats["avg_return"] is not None else None,
    }
