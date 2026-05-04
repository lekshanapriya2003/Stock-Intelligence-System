
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from transformers import pipeline as hf_pipeline
from PIL import Image
from datetime import datetime

from utils import (
    create_features, make_decision, load_artifacts,
    model_exists, get_real_news,
    score_headlines_finbert, backtest_strategy,
    FEATURE_COLS
)

#   ─ Page Config    ───
st.set_page_config(
    page_title="Stock Intelligence System v3",
    page_icon="       ",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_BG   = "#0d0d14"
GOLD      = "#e8c97a"
ACCENT    = "#f0a050"
RED_COL   = "#e05050"
GREEN_COL = "#60c870"
BLUE_COL  = "#6090e8"
TEXT_COL  = "#e8e6e0"
GRID_COL  = "#1e1e2a"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6e0; }
.hero-title { font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700; color:#e8c97a; letter-spacing:-0.02em; line-height:1.1; }
.hero-sub   { font-size:0.88rem; color:#666; letter-spacing:0.12em; text-transform:uppercase; margin-top:4px; }
.kpi-card   { background:#13131a; border:1px solid #2a2a38; border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.kpi-value  { font-family:'Space Mono',monospace; font-size:1.8rem; font-weight:700; }
.kpi-label  { font-size:0.72rem; color:#666; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px; }
.section-title { font-family:'Space Mono',monospace; font-size:1rem; color:#e8c97a; border-bottom:1px solid #2a2a38; padding-bottom:6px; margin:1.2rem 0 0.8rem; }
.news-card  { background:#13131a; border:0.5px solid #2a2a38; border-radius:10px; padding:0.8rem 1rem; margin-bottom:8px; }
.news-headline { font-size:0.85rem; color:#e8e6e0; line-height:1.5; font-weight:500; margin-bottom:4px; }
.news-meta  { font-size:0.72rem; color:#555; }
.badge-pos  { display:inline-block; background:#0d2010; color:#60c870; border:1px solid #60c87044; border-radius:20px; font-size:10px; padding:1px 8px; margin-left:6px; }
.badge-neg  { display:inline-block; background:#200d0d; color:#e05050; border:1px solid #e0505044; border-radius:20px; font-size:10px; padding:1px 8px; margin-left:6px; }
.badge-neu  { display:inline-block; background:#1a1708; color:#e8c97a; border:1px solid #e8c97a44; border-radius:20px; font-size:10px; padding:1px 8px; margin-left:6px; }
.warn-box   { background:#1a1410; border:1px solid #5a3a10; border-radius:8px; padding:0.75rem 1rem; color:#e8a050; font-size:0.83rem; margin-bottom:0.8rem; }
.info-box   { background:#101520; border:1px solid #1a2a4a; border-radius:8px; padding:0.75rem 1rem; color:#6090e8; font-size:0.83rem; margin-bottom:0.8rem; }
.key-box    { background:#101a10; border:1px solid #1a4a1a; border-radius:8px; padding:0.75rem 1rem; color:#60c870; font-size:0.83rem; margin-bottom:0.8rem; }
.finbert-box{ background:#0d1520; border:1px solid #1a3a5a; border-radius:8px; padding:0.75rem 1rem; color:#80b0e8; font-size:0.83rem; margin-bottom:0.8rem; }
.feature-row{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #1e1e2a; font-size:0.83rem; }
.decision-hero { border-radius:20px; padding:2.8rem 2rem 2.4rem; text-align:center; font-family:'Space Mono',monospace; margin:1.2rem 0 1.6rem; }
.decision-ticker { font-size:0.78rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.18em; margin-bottom:8px; }
.decision-label-hero { font-size:0.85rem; text-transform:uppercase; letter-spacing:0.2em; opacity:0.65; margin-bottom:10px; }
.decision-value-hero { font-size:6rem; font-weight:700; letter-spacing:0.04em; line-height:1; margin:0.1rem 0 0.6rem; }
.decision-reason-hero{ font-size:0.95rem; opacity:0.75; font-family:'DM Sans',sans-serif; margin-top:8px; }
.decision-scores { display:flex; justify-content:center; gap:2.5rem; margin-top:1.4rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.1); }
.score-item  { text-align:center; }
.score-val   { font-size:1.5rem; font-weight:700; }
.score-lbl   { font-size:0.7rem; opacity:0.55; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px; }
.stButton > button { background:linear-gradient(135deg,#e8c97a,#f0a050); color:#0a0a0f; border:none; border-radius:8px; font-weight:500; padding:0.5rem 1.8rem; }
[data-testid="stSidebar"] { background:#0d0d14; border-right:1px solid #1e1e2a; }
</style>
""", unsafe_allow_html=True)


def _fig(w=12, h=5):
    return plt.figure(figsize=(w, h), facecolor=DARK_BG)

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(GOLD)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.6)
    if title:  ax.set_title(title, fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


@st.cache_resource(show_spinner=False)
def get_model():
    try:
        if model_exists():
            return load_artifacts()
        else:
            st.error("❌ Model files not found. Please ensure model/ directory contains ann_model.h5 and scaler.pkl")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    """
    Loads FinBERT — trained on financial text.
    Falls back to DistilBERT if FinBERT unavailable.
    FinBERT paper: https://arxiv.org/abs/1908.10063
    """
    try:
        model = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",    # ← KEY UPGRADE from DistilBERT
            top_k=1
        )
        return model, "FinBERT"
    except Exception:
        pass
    try:
        model = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return model, "DistilBERT (fallback)"
    except Exception:
        return None, "unavailable"



# SIDEBAR

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem;">Settings</div>', unsafe_allow_html=True)
    st.markdown("---")

    all_tickers = [
        "AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "JPM", "BAC", "WMT",
        "PG", "JNJ", "KO", "XOM", "CVX", "SLB", "HAL", "PFE", "UNH", "ABT", "MRNA",
        "HD", "MCD", "NKE", "SBUX", "PLTR", "COIN", "SQ", "ROKU", "DIS", "NFLX",
        "ADBE", "CRM", "PYPL", "INTC", "AMD", "QCOM", "TXN", "CSCO", "IBM", "ORCL",
        "BA", "CAT", "GE", "MMM", "UPS", "FDX", "GS", "MS", "AXP", "V", "MA",
        "JPM", "BAC", "WFC", "C", "USB", "PNC", "T", "VZ", "TMUS", "CMCSA", "T",
        "INTU", "NOW", "ADP", "PAYC", "ZM", "DOCU", "SNOW", "PLTR", "GME", "AMC",
        "BB", "NOK", "BABA", "JD", "PDD", "BIDU", "TME", "NTES", "IQ", "KE Holdings",
        "LI", "XPEV", "NIO", "XPeng", "BYD", "KNDI", "QS", "RIVN", "LCID", "FSR",
        "CHPT", "BLNK", "CCIV", "SPCE", "RKDA", "ASTR", "MAXR", "RKT", "HOOD",
        "UBER", "LYFT", "DKNG", "MGM", "CZR", "LVS", "WYNN", "PENN", "DKNG", "FRC"
    ]
    
    ticker = st.selectbox(
        "Stock Ticker",
        all_tickers,
        index=0
    ).upper().strip()
    
    period    = st.selectbox("Data period", ["6mo","1y","2y","3y"], index=2)
    threshold = st.slider("ANN decision threshold", 0.30, 0.70, 0.55, 0.05,
                          help="Probability above this = BUY signal.")

    st.markdown("---")

    st.markdown('<div style="font-size:0.8rem; color:#e8c97a; font-weight:500; margin-bottom:6px;">Finnhub API Key</div>', unsafe_allow_html=True)
    finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
    if not finnhub_key:
        finnhub_key = st.text_input(
            "Finnhub API Key",
            value="",
            type="password",
            placeholder="paste key here",
            label_visibility="collapsed"
        )

    news_days = st.slider("News lookback (days)", 3, 30, 7)

    if finnhub_key:
        st.markdown('<div class="key-box">Finnhub News</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">      No key — add one above for live headlines</div>', unsafe_allow_html=True)

    st.markdown("---")
    plots_dir = "plots"
    if os.path.isdir(plots_dir):
        plot_files    = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
        selected_plot = st.selectbox("View training plot", ["—"] + plot_files)
    else:
        selected_plot = "—"
        st.caption("Run train.py first to generate plots.")




    
# HEADER
    
st.markdown('''
<div style="text-align: center; margin: 2rem 0;">
    <h1 style="
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #ffc107 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: shine 3s ease-in-out infinite;
        margin-bottom: 0.5rem;
    ">Stock Intelligence</h1>
    <style>
    @keyframes shine {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    </style>
</div>
''', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

model, scaler = get_model()
if model is None:
    st.markdown('<div class="warn-box">      No trained model found. Run <code>python train.py --ticker AAPL</code> first.</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="info-box">Initializing sentiment engine...</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyse = st.button(
        f"Analyse {ticker}",
        use_container_width=True,
        type="primary"
    )

if not analyse:
    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:#333;">
        <div style="font-size:3rem;">       </div>
        <div style="font-size:0.9rem; margin-top:0.5rem;">Enter a ticker in the sidebar and click Analyse</div>
    </div>""", unsafe_allow_html=True)
    if selected_plot != "—":
        st.image(Image.open(os.path.join(plots_dir, selected_plot)),
                 caption=selected_plot, use_container_width=True)
    st.stop()


    
# ANALYSIS
    

with st.spinner(f"Downloading {ticker} price data..."):
    raw_df = yf.download(ticker, period=period, progress=False)

if raw_df.empty:
    st.error(f"No data for '{ticker}'. Check the ticker symbol.")
    st.stop()

with st.spinner("Engineering 16 features..."):
    df = create_features(raw_df)

#    KPIs         
latest        = df.iloc[-1]
current_price = float(latest['Close'])
daily_return  = float(latest['Return'])
rsi_val       = float(latest['RSI'])
volatility    = float(latest['Volatility'])
week_change   = float(df['Close'].pct_change(5).iloc[-1])

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, color in [
    (c1, f"${current_price:.2f}", "Current Price",  GREEN_COL if daily_return >= 0 else RED_COL),
    (c2, f"{daily_return:+.2%}",  "Daily Return",   GREEN_COL if daily_return >= 0 else RED_COL),
    (c3, f"{week_change:+.2%}",   "5-Day Change",   GREEN_COL if week_change  >= 0 else RED_COL),
    (c4, f"{rsi_val:.1f}",        "RSI (14)",       GOLD),
    (c5, f"{volatility:.4f}",     "Volatility",     BLUE_COL),
]:
    with col:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#    ANN prediction    ─
feature_row    = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
feature_scaled = scaler.transform(feature_row)
ann_prob       = float(model.predict(feature_scaled, verbose=0)[0][0])

#    FinBERT sentiment  
sent_model_obj, sent_model_name = get_sentiment_model()

if finnhub_key:
    with st.spinner("Fetching real news from Finnhub..."):
        news_articles  = get_real_news(ticker, finnhub_key, news_days)
        news_headlines = [a["headline"] for a in news_articles]
else:
    news_articles  = []
    news_headlines = []

with st.spinner(f"Running {sent_model_name} sentiment analysis..."):
    sentiment_score, headline_sentiments = score_headlines_finbert(news_headlines, sent_model_obj)

#    Decision    ───────
decision, reason = make_decision(ann_prob, sentiment_score, threshold)

#    HERO BANNER    ────
dec_cfg = {
    "BUY":  {"fg":"#60c870","bg":"#071410","border":"#60c87060","glow":"#60c87022"},
    "SELL": {"fg":"#e05050","bg":"#140707","border":"#e0505060","glow":"#e0505022"},
    "HOLD": {"fg":"#e8c97a","bg":"#141108","border":"#e8c97a60","glow":"#e8c97a22"},
}
cfg        = dec_cfg[decision]
sent_color = GREEN_COL if sentiment_score > 0.15 else (RED_COL if sentiment_score < -0.15 else GOLD)
sent_label = "POSITIVE" if sentiment_score > 0.15 else ("NEGATIVE" if sentiment_score < -0.15 else "NEUTRAL")
ann_bar    = GREEN_COL if ann_prob > threshold else (RED_COL if ann_prob < (1 - threshold) else GOLD)
news_tag   = f"LIVE NEWS ({len(news_headlines)} articles · {sent_model_name})" if news_headlines else "NO NEWS KEY"

st.markdown(f"""
<div class="decision-hero" style="background:{cfg['bg']}; border:2px solid {cfg['border']}; box-shadow:0 0 60px {cfg['glow']};">
    <div class="decision-ticker">{ticker} · {period} data · {news_tag}</div>
    <div class="decision-label-hero" style="color:{cfg['fg']};">Final Recommendation</div>
    <div class="decision-value-hero" style="color:{cfg['fg']};">{decision}</div>
    <div class="decision-reason-hero" style="color:{cfg['fg']};">{reason}</div>
    <div class="decision-scores">
        <div class="score-item">
            <div class="score-val" style="color:{ann_bar};">{ann_prob:.1%}</div>
            <div class="score-lbl" style="color:{cfg['fg']}88;">ANN UP probability</div>
        </div>
        <div class="score-item">
            <div class="score-val" style="color:{sent_color};">{sent_label}</div>
            <div class="score-lbl" style="color:{cfg['fg']}88;">FinBERT sentiment ({sentiment_score:+.2f})</div>
        </div>
        <div class="score-item">
            <div class="score-val" style="color:{GOLD};">{rsi_val:.1f}</div>
            <div class="score-lbl" style="color:{cfg['fg']}88;">RSI (14)</div>
        </div>
        <div class="score-item">
            <div class="score-val" style="color:{BLUE_COL};">{volatility:.4f}</div>
            <div class="score-lbl" style="color:{cfg['fg']}88;">Volatility</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

#    FinBERT indicator  
if sent_model_name == "FinBERT":
    st.markdown(f'<div class="finbert-box"><b>FinBERT</b> — Trained on financial corpora - Stock news.</div>', unsafe_allow_html=True)
elif sent_model_name != "unavailable":
    st.markdown(f'<div class="warn-box">      FinBERT unavailable — using {sent_model_name}. Install with: <code>pip install transformers torch</code> and retry.</div>', unsafe_allow_html=True)


    
# TABS
    
tab1, tab2, tab3 = st.tabs(["Charts & News", " Backtest", " Data Preview"])

with tab1:
    st.markdown('<div class="section-title">Technical Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        fig_price = _fig(10, 6)
        ax_price = fig_price.add_subplot(111)
        ax_price.plot(df['Date'], df['Close'], color=TEXT_COL, lw=1.5, alpha=0.9, label='Close Price')
        ax_price.plot(df['Date'], df['MA10'], color=GOLD, lw=1.8, label='MA10')
        ax_price.plot(df['Date'], df['MA50'], color=ACCENT, lw=1.8, label='MA50')
        ax_price.fill_between(df['Date'], df['Lower_Band'], df['Upper_Band'],
                             alpha=0.15, color=BLUE_COL, label='Bollinger Bands')
        ax_price.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.7)
        ax_price.grid(True, alpha=0.2)
        _style(ax_price, f"{ticker} Price Action", ylabel="USD")
        st.pyplot(fig_price)
        plt.close(fig_price)
    
    with col2:
        st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
        
        fig_rsi = _fig(8, 3)
        ax_rsi = fig_rsi.add_subplot(111)
        ax_rsi.plot(df['Date'], df['RSI'], color=BLUE_COL, lw=2)
        ax_rsi.axhline(70, color=RED_COL, lw=2, ls='--', alpha=0.7)
        ax_rsi.axhline(30, color=GREEN_COL, lw=2, ls='--', alpha=0.7)
        ax_rsi.fill_between(df['Date'], df['RSI'], 70, where=(df['RSI'] >= 70), alpha=0.3, color=RED_COL)
        ax_rsi.fill_between(df['Date'], df['RSI'], 30, where=(df['RSI'] <= 30), alpha=0.3, color=GREEN_COL)
        ax_rsi.axhline(df['RSI'].iloc[-1], color=GOLD, lw=2, alpha=0.8)
        ax_rsi.text(df['Date'].iloc[-1], df['RSI'].iloc[-1], f' {rsi_val:.1f}', 
                    color=GOLD, fontsize=10, va='center')
        _style(ax_rsi, "RSI (14)", ylabel="RSI")
        st.pyplot(fig_rsi)
        plt.close(fig_rsi)
        
        fig_macd = _fig(8, 3)
        ax_macd = fig_macd.add_subplot(111)
        macd_hist = df['MACD'] - df['Signal_Line']
        ax_macd.bar(df['Date'], macd_hist,
                   color=[GREEN_COL if v >= 0 else RED_COL for v in macd_hist],
                   width=1.0, alpha=0.8)
        ax_macd.plot(df['Date'], df['MACD'], color=GOLD, lw=2, label='MACD')
        ax_macd.plot(df['Date'], df['Signal_Line'], color=ACCENT, lw=2, label='Signal')
        ax_macd.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.7)
        _style(ax_macd, "MACD", ylabel="Value")
        st.pyplot(fig_macd)
        plt.close(fig_macd)
    
    st.markdown("---")
    st.markdown('<div class="section-title">News Sentiment</div>', unsafe_allow_html=True)
    
    if news_headlines:
        for i, article in enumerate(news_headlines[:5]):
            if isinstance(article, dict) and 'sentiment' in article:
                sentiment_color = GREEN_COL if article['sentiment'] > 0.15 else (RED_COL if article['sentiment'] < -0.15 else GOLD)
                st.markdown(f"""
                <div style="padding:8px; margin:4px 0; border-left:3px solid {sentiment_color}; background:rgba(255,255,255,0.05);">
                    <small><b>{article.get('headline', 'No headline')}</b></small><br>
                    <small style="opacity:0.7;">Score: {article.get('sentiment', 0):+.2f} | {article.get('source', 'Unknown')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:8px; margin:4px 0; border-left:3px solid #888; background:rgba(255,255,255,0.05);">
                    <small><b>{str(article)[:100]}...</b></small><br>
                    <small style="opacity:0.7;">Unknown format</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Add Finnhub API key to see live news sentiment")
    
    st.markdown("---")
    st.markdown('<div class="section-title">Return Distribution</div>', unsafe_allow_html=True)
    fig_dist, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    returns = df['Return'].dropna()
    ax.hist(returns[returns >= 0], bins=60, color=GREEN_COL, alpha=0.7, label="Positive", edgecolor='none', density=True)
    ax.hist(returns[returns < 0],  bins=60, color=RED_COL,   alpha=0.7, label="Negative", edgecolor='none', density=True)
    ax.axvline(returns.mean(),   color=GOLD,   lw=1.5, ls='--', label=f"Mean {returns.mean():.4f}")
    ax.axvline(returns.median(), color=ACCENT, lw=1.5, ls=':',  label=f"Median {returns.median():.4f}")
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style(ax, "Daily Return Distribution", "Return", "Density")
    st.pyplot(fig_dist)
    plt.close(fig_dist)

with tab2:
    st.markdown('<div class="section-title">Strategy Backtest</div>', unsafe_allow_html=True)
    
    if len(df) > 50:
        signals = []
        for i in range(50, len(df)):
            if df['RSI'].iloc[i] < 30 and df['Close'].iloc[i] > df['MA10'].iloc[i]:
                signals.append(1)
            elif df['RSI'].iloc[i] > 70 and df['Close'].iloc[i] < df['MA10'].iloc[i]:
                signals.append(-1)
            else:
                signals.append(0)
        
        returns = []
        position = 0
        entry_price = 0
        
        for i, signal in enumerate(signals):
            current_price = df['Close'].iloc[i + 50]
            if signal == 1 and position == 0:
                position = 1
                entry_price = current_price
            elif signal == -1 and position == 1:
                returns.append((current_price - entry_price) / entry_price)
                position = 0
        
        if returns:
            total_return = sum(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col3:
                st.metric("Trades", f"{len(returns)}")
            with col4:
                st.metric("Avg Return", f"{sum(returns)/len(returns):.2%}")
            
            equity = [1]
            for r in returns:
                equity.append(equity[-1] * (1 + r))
            
            fig_bt, ax_bt = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
            ax_bt.plot(equity, color=GOLD, lw=2)
            ax_bt.fill_between(range(len(equity)), equity, alpha=0.3, color=GOLD)
            ax_bt.set_title("Strategy Equity Curve", color=TEXT_COL, fontsize=12)
            ax_bt.grid(True, alpha=0.2)
            ax_bt.set_facecolor(DARK_BG)
            ax_bt.tick_params(colors=TEXT_COL)
            st.pyplot(fig_bt)
            plt.close(fig_bt)
        else:
            st.info("No trades executed in the selected period")
    else:
        st.warning("Insufficient data for backtesting")

with tab3:
    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Latest Feature Values (16)</div>', unsafe_allow_html=True)
    feature_row_df = df[FEATURE_COLS].iloc[-1]
    for feat, val in feature_row_df.items():
        st.markdown(f"""
        <div class="feature-row">
            <span style="color:#888;">{feat}</span>
            <span style="color:{TEXT_COL}; font-family:monospace;">{val:.5f}</span>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-title">Raw Data (Last 10 Records)</div>', unsafe_allow_html=True)
    st.dataframe(df.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="font-size:0.72rem; color:#444; text-align:center; line-height:1.7;">
      Educational project only. Not financial advice. Past patterns do not guarantee future returns.<br>
Sentiment: FinBERT (ProsusAI/finbert) | Price data: Yahoo Finance | News: Finnhub
</div>""", unsafe_allow_html=True)
