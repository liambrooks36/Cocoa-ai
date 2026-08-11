from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import math, time, re
import pandas as pd
import requests
import feedparser

ROOT = Path(__file__).parent
app = FastAPI(title="Cocoa AI V1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")

@app.get("/health")
def health():
    return {"ok": True, "service": "cocoa-ai-v1"}

def clean_num(v):
    try:
        if pd.isna(v): return None
        return float(v)
    except Exception:
        return None

@app.get("/api/candles")
def candles(
    interval: str = Query("1h", pattern="^(5m|15m|30m|60m|1h|1d)$"),
    range: str = Query("1mo", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|max)$")
):
    """
    Fetch delayed cocoa futures candles directly from Yahoo Finance's chart API.

    This avoids yfinance failures that can happen on some cloud hosts.
    """
    symbol = "CC=F"

    # Yahoo commonly accepts 60m rather than 1h in the chart endpoint.
    yahoo_interval = "60m" if interval == "1h" else interval

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe='')}"
    params = {
        "interval": yahoo_interval,
        "range": range,
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CocoaAI/1.0)",
        "Accept": "application/json,text/plain,*/*",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
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

        result = (chart.get("result") or [])
        if not result:
            return {
                "symbol": symbol,
                "source": "Yahoo Finance chart API",
                "delayed": True,
                "interval": interval,
                "range": range,
                "candles": [],
                "error": "Yahoo returned no chart result",
            }

        result = result[0]
        timestamps = result.get("timestamp") or []
        quotes = ((result.get("indicators") or {}).get("quote") or [])
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

                candles_out.append({
                    "t": int(ts),
                    "o": float(o),
                    "h": float(h),
                    "l": float(l),
                    "c": float(c),
                    "v": float(v or 0),
                })
            except (TypeError, ValueError, IndexError):
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
        # Return a structured error instead of a generic FastAPI 500.
        return {
            "symbol": symbol,
            "source": "Yahoo Finance chart API",
            "delayed": True,
            "interval": interval,
            "range": range,
            "candles": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

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
    locs={}
    for key,(lat,lon) in WEATHER_POINTS.items():
        u="https://api.open-meteo.com/v1/forecast"
        p={
            "latitude":lat,"longitude":lon,
            "daily":"precipitation_sum,temperature_2m_max",
            "timezone":"auto","forecast_days":7
        }
        r=requests.get(u,params=p,timeout=15)
        r.raise_for_status()
        j=r.json()
        rain=sum(x or 0 for x in j["daily"]["precipitation_sum"])
        tmax=max(j["daily"]["temperature_2m_max"])
        label,color=weather_risk(rain,tmax)
        locs[key]={
            "rain_7d_mm":float(rain),
            "max_temp_c":float(tmax),
            "risk_label":label,
            "risk_color":color
        }
    return {"source":"Open-Meteo","locations":locs}

@app.get("/api/news")
def news():
    # Server-side RSS avoids browser CORS problems.
    query='cocoa OR cacao Ghana OR "Ivory Coast"'
    rss="https://news.google.com/rss/search"
    params={"q":query,"hl":"en-GB","gl":"GB","ceid":"GB:en"}
    resp=requests.get(rss,params=params,timeout=15)
    resp.raise_for_status()
    feed=feedparser.parse(resp.content)
    items=[]
    for e in feed.entries[:12]:
        src=""
        if hasattr(e,"source") and isinstance(e.source,dict):
            src=e.source.get("title","")
        items.append({
            "title": re.sub(r"\s+"," ",getattr(e,"title","")).strip(),
            "published_at": getattr(e,"published","Recent"),
            "source": src or "Google News",
            "link": getattr(e,"link","")
        })
    return {"source":"Google News RSS","items":items}
