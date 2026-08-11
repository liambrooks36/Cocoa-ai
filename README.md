# Cocoa AI V1

This is the first backend-ready build.

## What is real when the backend is running
- Cocoa futures candles from Yahoo Finance/yfinance using `CC=F` (delayed data).
- 5m / 15m / 1h / 4h / 1D chart data (4h is aggregated from 1h candles).
- RSI and daily-move technical input calculated from real candles.
- Four-region cocoa-belt weather from Open-Meteo.
- Cocoa news fetched server-side from Google News RSS.
- Browser CORS problems are avoided because the dashboard talks to its own backend.

## What is still prototype
- AI LONG/SHORT/NO TRADE score.
- Priced-in score.
- Historical-similarity statistics.
- News-to-price reaction grading.
- Automatic prediction outcomes.
- Supabase database.
- LLM analysis.

## Run locally
pip install -r requirements.txt
uvicorn app:app --reload

Open http://127.0.0.1:8000

## Deploy on Render
The included render.yaml is ready for a basic Python web service.
