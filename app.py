import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timezone
import urllib.parse
from email.utils import parsedate_to_datetime

# --- 1. System Setup & Masquerade Session ---
st.set_page_config(page_title="AlphaSense | Multi-Asset Terminal", page_icon="📈", layout="wide")

@st.cache_resource
def setup_nlp():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('sentiment/vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nlp()

if 'last_price_memory' not in st.session_state:
    st.session_state.last_price_memory = {}
if 'current_asset' not in st.session_state:
    st.session_state.current_asset = None

# --- 2. Enterprise Asset Universe (European Focus) ---
ASSET_CLASSES = {
    "European Equities (CAC 40 & DAX)": {
        "Tickers": {"MC.PA": "LVMH", "TTE.PA": "TotalEnergies", "BNP.PA": "BNP Paribas", "ASML.AS": "ASML Holding", "SAP.DE": "SAP"},
        "Query_Suffix": "europe stock earnings market", "Symbol": "€"
    },
    "Global Indices": {
        "Tickers": {"^FCHI": "CAC 40 (France)", "^STOXX50E": "Euro Stoxx 50", "^GDAXI": "DAX (Germany)", "^GSPC": "S&P 500 (US)"},
        "Query_Suffix": "index economy european market", "Symbol": "Pts "
    },
    "Forex / Currencies": {
        "Tickers": {"EURUSD=X": "EUR/USD", "EURGBP=X": "EUR/GBP", "EURCHF=X": "EUR/CHF", "EURJPY=X": "EUR/JPY"},
        "Query_Suffix": "forex ECB central bank euro", "Symbol": ""
    },
    "Commodities & Energy": {
        "Tickers": {"BZ=F": "Brent Crude Oil (UK)", "GC=F": "Gold", "TTF=F": "Dutch TTF Gas"},
        "Query_Suffix": "commodity futures energy europe", "Symbol": "$"
    },
    "Cryptocurrencies (EUR Pairings)": {
        "Tickers": {"BTC-EUR": "Bitcoin (EUR)", "ETH-EUR": "Ethereum (EUR)", "SOL-EUR": "Solana (EUR)"},
        "Query_Suffix": "crypto token blockchain regulation", "Symbol": "€"
    }
}

# --- 3. Quantitative Functions ---
def calculate_rsi(data, periods=14):
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    rsi = 100 - (100 / (1 + (ma_up / ma_down)))
    return rsi

def calculate_volatility(data, periods=14):
    return ((data['High'] - data['Low']) / data['Close']).rolling(window=periods).mean().iloc[-1]

# --- 4. Resilient Data Pipeline ---
@st.cache_data(ttl=15, show_spinner=False)
def fetch_live_news(asset_name, query_suffix):
    encoded_query = urllib.parse.quote(f"{asset_name} {query_suffix}")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-GB&gl=GB&ceid=GB:en"
    
    # Add the header directly inside the function
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(response.content)
        return [{'title': item.find('title').text, 'published': item.find('pubDate').text} for item in root.findall('./channel/item')[:10]]
    except: return []

@st.cache_data(ttl=15, show_spinner=False)
def fetch_market_data(ticker_symbol):
    
    # Removed the session argument. Let yfinance use curl_cffi natively.
    ticker = yf.Ticker(ticker_symbol) 
    
    try:
        hist = ticker.history(period="1d", interval="1m")
        status = "🟢 OPEN"
        if hist.empty:
            hist = ticker.history(period="5d", interval="1h")
            status = "🔴 CLOSED/EXTENDED"
        
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist)
        return hist, status, calculate_volatility(hist), ticker.fast_info.get('lastPrice', hist['Close'].iloc[-1])
    except: return None, "RATE LIMITED", 0, 0

# --- 5. Dashboard Execution ---
st.title("🏛️ AlphaSense: Multi-Asset Quant Terminal")
with st.sidebar:
    st.header("⚙️ Selection")
    s_class = st.selectbox("Market", list(ASSET_CLASSES.keys()))
    s_ticker = st.selectbox("Asset", list(ASSET_CLASSES[s_class]["Tickers"].keys()), format_func=lambda x: f"{x} ({ASSET_CLASSES[s_class]['Tickers'][x]})")
    trade_size = st.number_input(f"Size ({ASSET_CLASSES[s_class]['Symbol']})", value=10000)
    live = st.toggle("Enable Live Polling", value=True)
    if s_ticker != st.session_state.current_asset:
        st.toast(f"Routing feed for {s_ticker}...", icon="📡")
        st.session_state.current_asset = s_ticker

def run_cycle():
    hist, status, vol, price = fetch_market_data(s_ticker)
    news = fetch_live_news(ASSET_CLASSES[s_class]["Tickers"][s_ticker], ASSET_CLASSES[s_class]["Query_Suffix"])
    if hist is None:
        st.error("Rate limit hit. Waiting 15s...")
        return

    # NLP and Logic
    scores = [sia.polarity_scores(n['title'])['compound'] for n in news]
    avg_s = np.mean(scores) if scores else 0
    delta = price - st.session_state.last_price_memory.get(s_ticker, price)
    st.session_state.last_price_memory[s_ticker] = price
    
    # UI Elements
    st.caption(f"⚡ Sync: {datetime.now().strftime('%H:%M:%S')} | Status: {status}")
    k = st.columns(6)
    sym = ASSET_CLASSES[s_class]['Symbol']
    k[0].metric("Price", f"{sym}{price:,.2f}", f"{delta:+.4f}")
    k[1].metric("RSI (14)", f"{hist['RSI'].iloc[-1]:.1f}")
    k[2].metric("SMA (20)", f"{sym}{hist['SMA_20'].iloc[-1]:,.2f}")
    k[3].metric("NLP Sentiment", f"{avg_s:.3f}")
    k[4].metric("24H Impact", f"{sym}{trade_size * vol:,.2f}")
    
    sig, col = ("HOLD", "#fec036")
    if avg_s > 0.1 and price > hist['SMA_20'].iloc[-1]: sig, col = ("BUY", "#00cc96")
    elif avg_s < -0.1 and price < hist['SMA_20'].iloc[-1]: sig, col = ("SELL", "#ff4b4b")
    k[5].markdown(f"<div style='background:{col}15;padding:10px;border:2px solid {col};text-align:center;'><h4 style='margin:0;color:{col};'>{sig}</h4></div>", unsafe_allow_html=True)
    
    # Visuals
    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], line=dict(color='orange')))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🗞️ Contextual NLP Feed")
    processed_news = []
    for n in news:
        try:
            dt = parsedate_to_datetime(n['published'])
            age = int((datetime.now(timezone.utc) - dt.replace(tzinfo=timezone.utc)).total_seconds() / 60)
            time_label = f"{age}m ago" if age < 60 else dt.strftime('%d %b %Y')
        except: time_label = "Live"
        processed_news.append({"Headline": n['title'], "NLP Score": sia.polarity_scores(n['title'])['compound'], "Published": time_label})
    
    st.dataframe(pd.DataFrame(processed_news).style.map(lambda x: 'color:#00cc96' if x > 0.05 else ('color:#ff4b4b' if x < -0.05 else ''), subset=['NLP Score']), use_container_width=True, hide_index=True)

run_cycle()
if live:
    time.sleep(15)
    st.rerun()