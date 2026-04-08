
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
    if model_exists():
        return load_artifacts()
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
    st.markdown('<div class="hero-title" style="font-size:1.5rem;">⚙ Settings</div>', unsafe_allow_html=True)
    st.markdown("---")

    ticker    = st.text_input("Stock Ticker", "AAPL").upper().strip()
    period    = st.selectbox("Data period", ["6mo","1y","2y","3y"], index=2)
    threshold = st.slider("ANN decision threshold", 0.30, 0.70, 0.55, 0.05,
                          help="Probability above this = BUY signal.")

    st.markdown("---")

    st.markdown('<div style="font-size:0.8rem; color:#e8c97a; font-weight:500; margin-bottom:6px;">Finnhub API Key</div>', unsafe_allow_html=True)
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
    
st.markdown('<div class="hero-title">Multi-Modal Stock Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ANN Price Prediction · FinBERT NLP · Backtest · v3</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

model, scaler = get_model()
if model is None:
    st.markdown('<div class="warn-box">      No trained model found. Run <code>python train.py --ticker AAPL</code> first.</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="info-box">Initializing sentiment engine...</div>', unsafe_allow_html=True)

analyse = st.button("         Analyse " + ticker)

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

#         TAB 1               
with tab1:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-title">Price & Technical Indicators</div>', unsafe_allow_html=True)

        fig = _fig(11, 9)
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['Date'], df['Close'],  color=TEXT_COL, lw=0.9, alpha=0.9, label='Close')
        ax1.plot(df['Date'], df['MA10'],   color=GOLD,     lw=1.2, label='MA10')
        ax1.plot(df['Date'], df['MA50'],   color=ACCENT,   lw=1.2, label='MA50')
        ax1.fill_between(df['Date'], df['Lower_Band'], df['Upper_Band'],
                         alpha=0.10, color=BLUE_COL, label='Bollinger')
        ax1.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
        _style(ax1, f"{ticker} — Price", ylabel="USD")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df['Date'], df['RSI'], color=BLUE_COL, lw=1.1)
        ax2.axhline(70, color=RED_COL,   lw=0.9, ls='--')
        ax2.axhline(30, color=GREEN_COL, lw=0.9, ls='--')
        ax2.fill_between(df['Date'], df['RSI'], 70, where=(df['RSI'] >= 70), alpha=0.18, color=RED_COL)
        ax2.fill_between(df['Date'], df['RSI'], 30, where=(df['RSI'] <= 30), alpha=0.18, color=GREEN_COL)
        ax2_stoch = ax2.twinx()
        ax2_stoch.plot(df['Date'], df['Stoch_K'], color=ACCENT, lw=0.7, alpha=0.5, label='Stoch %K')
        ax2_stoch.tick_params(colors=ACCENT, labelsize=7)
        ax2_stoch.set_ylabel("Stoch %K", color=ACCENT, fontsize=7)
        _style(ax2, "RSI (14)  +  Stochastic %K", ylabel="RSI")

        ax3 = fig.add_subplot(gs[2])
        macd_hist = df['MACD'] - df['Signal_Line']
        ax3.bar(df['Date'], macd_hist,
                color=[GREEN_COL if v >= 0 else RED_COL for v in macd_hist],
                width=1.0, alpha=0.7)
        ax3.plot(df['Date'], df['MACD'],        color=GOLD,   lw=1.1, label='MACD')
        ax3.plot(df['Date'], df['Signal_Line'], color=ACCENT, lw=1.1, label='Signal')
        ax3.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
        _style(ax3, "MACD", ylabel="Value")

        st.pyplot(fig)
        plt.close(fig)

        # Return distribution
        st.markdown('<div class="section-title">Return Distribution</div>', unsafe_allow_html=True)
        fig2, ax = plt.subplots(figsize=(10, 3.5), facecolor=DARK_BG)
        returns = df['Return'].dropna()
        ax.hist(returns[returns >= 0], bins=60, color=GREEN_COL, alpha=0.7, label="Positive", edgecolor='none', density=True)
        ax.hist(returns[returns < 0],  bins=60, color=RED_COL,   alpha=0.7, label="Negative", edgecolor='none', density=True)
        ax.axvline(returns.mean(),   color=GOLD,   lw=1.5, ls='--', label=f"Mean {returns.mean():.4f}")
        ax.axvline(returns.median(), color=ACCENT, lw=1.5, ls=':',  label=f"Median {returns.median():.4f}")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
        _style(ax, "Daily Return Distribution", "Return", "Density")
        st.pyplot(fig2)
        plt.close(fig2)

        # ATR + Williams %R
        st.markdown('<div class="section-title">Volatility  ·  ATR  ·  Williams %R</div>', unsafe_allow_html=True)
        fig3, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(10, 5), facecolor=DARK_BG)
        ax_v.fill_between(df['Date'], df['Volatility'], alpha=0.35, color=ACCENT)
        ax_v.plot(df['Date'], df['Volatility'], color=ACCENT, lw=1.2)
        ax_v_twin = ax_v.twinx()
        ax_v_twin.plot(df['Date'], df['ATR'], color=BLUE_COL, lw=1.0, alpha=0.7, label='ATR')
        ax_v_twin.tick_params(colors=BLUE_COL, labelsize=7)
        ax_v_twin.set_ylabel("ATR", color=BLUE_COL, fontsize=7)
        _style(ax_v, "Volatility + ATR", "Date", "Std Dev")
        ax_w.plot(df['Date'], df['Williams_R'], color=GOLD, lw=1.1)
        ax_w.axhline(-20, color=RED_COL,   lw=0.9, ls='--', label='Overbought −20')
        ax_w.axhline(-80, color=GREEN_COL, lw=0.9, ls='--', label='Oversold −80')
        ax_w.fill_between(df['Date'], df['Williams_R'], -20, where=(df['Williams_R'] >= -20), alpha=0.18, color=RED_COL)
        ax_w.fill_between(df['Date'], df['Williams_R'], -80, where=(df['Williams_R'] <= -80), alpha=0.18, color=GREEN_COL)
        ax_w.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
        _style(ax_w, "Williams %R", "Date", "%R")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with right:
        # ANN gauge
        st.markdown('<div class="section-title">ANN Prediction</div>', unsafe_allow_html=True)
        fig_g, ax_g = plt.subplots(figsize=(6, 1.2), facecolor=DARK_BG)
        ax_g.set_facecolor(DARK_BG)
        ax_g.barh(0, 1, color=GRID_COL, height=0.5)
        bar_color = GREEN_COL if ann_prob > threshold else (RED_COL if ann_prob < (1 - threshold) else GOLD)
        ax_g.barh(0, ann_prob, color=bar_color, height=0.5, alpha=0.88)
        ax_g.axvline(threshold, color=TEXT_COL, lw=1.2, ls='--', alpha=0.5)
        ax_g.set_xlim(0, 1)
        ax_g.axis('off')
        ax_g.text(ann_prob / 2, 0, f"{ann_prob:.1%}", ha='center', va='center',
                  color=DARK_BG, fontsize=11, fontweight='bold')
        ax_g.text(0.5, -0.6, f"ANN UP probability  |  threshold = {threshold}",
                  ha='center', va='top', color=TEXT_COL, fontsize=7.5, transform=ax_g.transData)
        fig_g.patch.set_facecolor(DARK_BG)
        st.pyplot(fig_g)
        plt.close(fig_g)

        # News
        st.markdown(f'<div class="section-title">News Sentiment ({sent_model_name})</div>', unsafe_allow_html=True)

        if not finnhub_key:
            st.markdown("""
            <div class="warn-box">
            <b>How to get a free Finnhub key:</b><br>
            1. Go to <b>finnhub.io/register</b><br>
            2. Sign up with just your email<br>
            3. Copy key from your dashboard<br>
            4. Paste it into the sidebar field<br>
            <i>Free tier · No credit card needed</i>
            </div>""", unsafe_allow_html=True)
        elif not news_headlines:
            st.markdown(f'<div class="warn-box">      No news found for {ticker} in last {news_days} days. Try increasing the lookback slider.</div>', unsafe_allow_html=True)
        else:
            for i, (headline, score, conf, label) in enumerate(headline_sentiments[:8]):
                badge_class = "badge-pos" if score > 0 else ("badge-neg" if score < 0 else "badge-neu")
                badge_label = label.upper()
                art = news_articles[i] if i < len(news_articles) else {}
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-headline">{headline}<span class="{badge_class}">{badge_label} {conf:.0%}</span></div>
                    <div class="news-meta">{art.get('source','')}&nbsp;·&nbsp;{art.get('datetime','')}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align:center; margin:0.8rem 0 0.4rem;">
                <span style="font-size:1.6rem; font-weight:700; color:{sent_color}; font-family:monospace;">{sent_label}</span>
                <span style="color:#555; font-size:0.9rem;"> ({sentiment_score:+.2f})</span>
                <div style="font-size:0.72rem; color:#555; margin-top:3px;">{len(news_headlines)} articles · last {news_days} days · {sent_model_name}</div>
            </div>""", unsafe_allow_html=True)

        # Feature values
        st.markdown('<div class="section-title">Latest Feature Values (16)</div>', unsafe_allow_html=True)
        feature_row_df = df[FEATURE_COLS].iloc[-1]
        for feat, val in feature_row_df.items():
            st.markdown(f"""
            <div class="feature-row">
                <span style="color:#888;">{feat}</span>
                <span style="color:{TEXT_COL}; font-family:monospace;">{val:.5f}</span>
            </div>""", unsafe_allow_html=True)


#         TAB 2: BACKTEST        
with tab2:
    st.markdown('<div class="section-title">        Strategy Backtest — ANN Signal vs Buy & Hold</div>', unsafe_allow_html=True)

    bt_threshold = st.slider("Backtest signal threshold", 0.30, 0.70, float(threshold), 0.05,
                             key="bt_thresh",
                             help="BUY signal fired when ANN probability exceeds this. Higher = fewer, more confident trades.")
    starting_cap = st.number_input("Starting capital ($)", value=10000, step=1000, min_value=1000)

    # Run backtest on test period (last 20% of data)
    test_start = int(len(df) * 0.8)
    test_df    = df.iloc[test_start:].copy()

    # Need ANN predictions over the whole test window
    X_test_full   = test_df[FEATURE_COLS].values
    X_test_scaled = scaler.transform(X_test_full)
    all_probs     = model.predict(X_test_scaled, verbose=0).flatten()

    bt = backtest_strategy(df, all_probs, threshold=bt_threshold,
                           starting_capital=float(starting_cap),
                           test_start_idx=test_start)

    # KPIs
    b1, b2, b3, b4 = st.columns(4)
    kpis = [
        (f"{bt['strat_return']:+.1%}", "Strategy Return", GREEN_COL if bt['strat_return'] > 0 else RED_COL),
        (f"{bt['bh_return']:+.1%}",    "Buy & Hold",      GREEN_COL if bt['bh_return'] > 0 else RED_COL),
        (f"{bt['sharpe']:.2f}",        "Sharpe Ratio",    GOLD),
        (f"{bt['max_drawdown']:.1%}",  "Max Drawdown",    RED_COL),
    ]
    for col, (val, label, color) in zip([b1,b2,b3,b4], kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity curve
    fig_bt, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(11, 7), facecolor=DARK_BG,
                                           gridspec_kw={'height_ratios': [3, 1]})
    dates = bt['dates']

    ax_eq.set_facecolor(DARK_BG)
    ax_eq.plot(dates, bt['strat_equity'], color=GOLD,     lw=2.0, label=f"ANN Strategy  {bt['strat_return']:+.1%}")
    ax_eq.plot(dates, bt['bh_equity'],    color=BLUE_COL, lw=1.8, label=f"Buy & Hold    {bt['bh_return']:+.1%}", ls='--')
    ax_eq.fill_between(dates, bt['strat_equity'], bt['bh_equity'],
                       where=(bt['strat_equity'] >= bt['bh_equity']),
                       alpha=0.12, color=GREEN_COL, label='Outperforming')
    ax_eq.fill_between(dates, bt['strat_equity'], bt['bh_equity'],
                       where=(bt['strat_equity'] < bt['bh_equity']),
                       alpha=0.12, color=RED_COL, label='Underperforming')
    ax_eq.legend(fontsize=9, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style(ax_eq, f"{ticker} — ANN Strategy vs Buy & Hold  |  {bt['n_trades']} trades",
           ylabel=f"Portfolio ($)")

    roll_max = np.maximum.accumulate(bt['strat_equity'])
    drawdown = (bt['strat_equity'] - roll_max) / roll_max * 100
    ax_dd.set_facecolor(DARK_BG)
    ax_dd.fill_between(dates, drawdown, 0, color=RED_COL, alpha=0.6)
    ax_dd.axhline(bt['max_drawdown'] * 100, color=ACCENT, lw=0.8, ls='--',
                  label=f"Max DD {bt['max_drawdown']:.1%}")
    ax_dd.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style(ax_dd, "Drawdown (%)", "Date", "%")
    plt.tight_layout()
    st.pyplot(fig_bt)
    plt.close(fig_bt)

    # Alpha
    alpha = bt['strat_return'] - bt['bh_return']
    alpha_color = GREEN_COL if alpha > 0 else RED_COL
    alpha_label = "outperformed" if alpha > 0 else "underperformed"
    st.markdown(f"""
    <div style="text-align:center; padding:1rem; background:#13131a; border-radius:10px; margin-top:0.5rem;">
        <span style="color:{alpha_color}; font-size:1.4rem; font-family:monospace; font-weight:700;">{alpha:+.1%}</span>
        <span style="color:#555; font-size:0.9rem;"> — ANN strategy {alpha_label} buy-and-hold over the test period</span>
    </div>""", unsafe_allow_html=True)


#         TAB 3: DATA            
with tab3:
    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
    view_tab = st.radio("View", ["Head","Tail","Describe","Null Check"], horizontal=True, label_visibility="collapsed")
    show_cols = ['Date','Close','Return','MA10','RSI','Stoch_K','MACD','Target']
    if view_tab == "Head":
        st.dataframe(df[show_cols].head(10), use_container_width=True, hide_index=True)
    elif view_tab == "Tail":
        st.dataframe(df[show_cols].tail(10), use_container_width=True, hide_index=True)
    elif view_tab == "Describe":
        st.dataframe(df[FEATURE_COLS].describe().round(4), use_container_width=True)
    else:
        null_df = df[FEATURE_COLS+['Target']].isnull().sum().reset_index()
        null_df.columns = ['Feature','Null Count']
        null_df['Status'] = null_df['Null Count'].apply(lambda x: "✓ OK" if x==0 else f"      {x}")
        st.dataframe(null_df, use_container_width=True, hide_index=True)


# Training plot viewer
if selected_plot != "—":
    st.markdown("---")
    st.markdown(f'<div class="section-title">Training Plot: {selected_plot}</div>', unsafe_allow_html=True)
    st.image(Image.open(os.path.join(plots_dir, selected_plot)), use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="font-size:0.72rem; color:#444; text-align:center; line-height:1.7;">
      Educational project only. Not financial advice. Past patterns do not guarantee future returns.<br>
Sentiment: FinBERT (ProsusAI/finbert) | Price data: Yahoo Finance | News: Finnhub
</div>""", unsafe_allow_html=True)
