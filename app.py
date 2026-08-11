from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import math, time, re
import pandas as pd
import yfinance as yf
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
    # Yahoo's cocoa futures symbol. Data is delayed, which is fine for V1 research/monitoring.
    ticker = yf.Ticker("CC=F")
    df = ticker.history(period=range, interval=interval, auto_adjust=False, actions=False)
    if df.empty:
        return {"source":"Yahoo Finance / yfinance","delayed":True,"candles":[]}

    out=[]
    for ts,row in df.iterrows():
        o,h,l,c = map(clean_num,[row.get("Open"),row.get("High"),row.get("Low"),row.get("Close")])
        if None in (o,h,l,c): continue
        out.append({
            "t": int(ts.timestamp()),
            "o": o, "h": h, "l": l, "c": c,
            "v": clean_num(row.get("Volume")) or 0
        })
    return {
        "symbol":"CC=F",
        "source":"Yahoo Finance / yfinance",
        "delayed":True,
        "interval":interval,
        "range":range,
        "candles":out
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
