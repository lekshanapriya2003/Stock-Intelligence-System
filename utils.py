
import numpy as np
import pandas as pd
import joblib
import os
import requests
from datetime import datetime, timedelta

MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "ann_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

#16 features
FEATURE_COLS = [
    'Return', 'MA5', 'MA10', 'MA20', 'MA50',
    'Volatility', 'RSI', 'MACD', 'Signal_Line',
    'Upper_Band', 'Lower_Band', 'Volume_Change',
    'Stoch_K', 'Williams_R', 'ATR', 'OBV_Change'
]

                   
# FEATURE ENGINEERING

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_stochastic(df, period=14):
    """Stochastic Oscillator %K — momentum (0–100)."""
    low_min  = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    return 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-8)


def compute_williams_r(df, period=14):
    """Williams %R — overbought/oversold (−100 to 0)."""
    high_max = df['High'].rolling(period).max()
    low_min  = df['Low'].rolling(period).min()
    return -100 * (high_max - df['Close']) / (high_max - low_min + 1e-8)


def compute_atr(df, period=14):
    """Average True Range — volatility measure."""
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_obv(df):
    """On-Balance Volume — volume momentum."""
    direction = np.where(df['Close'] > df['Close'].shift(1), 1, -1)
    obv = (df['Volume'] * direction).cumsum()
    return obv


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers 16 features from raw OHLCV data."""
    df = df.copy()

    # flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    #    Original 12 features   
    df['Return']       = df['Close'].pct_change()
    df['MA5']          = df['Close'].rolling(5).mean()
    df['MA10']         = df['Close'].rolling(10).mean()
    df['MA20']         = df['Close'].rolling(20).mean()
    df['MA50']         = df['Close'].rolling(50).mean()
    df['Volatility']   = df['Return'].rolling(10).std()
    df['RSI']          = compute_rsi(df['Close'])

    ema12              = df['Close'].ewm(span=12, adjust=False).mean()
    ema26              = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']         = ema12 - ema26
    df['Signal_Line']  = df['MACD'].ewm(span=9, adjust=False).mean()

    rolling_mean       = df['Close'].rolling(20).mean()
    rolling_std        = df['Close'].rolling(20).std()
    df['Upper_Band']   = rolling_mean + 2 * rolling_std
    df['Lower_Band']   = rolling_mean - 2 * rolling_std

    df['Volume_Change'] = df['Volume'].pct_change() if 'Volume' in df.columns else 0.0

    #    4 new features   
    if 'High' in df.columns and 'Low' in df.columns:
        df['Stoch_K']    = compute_stochastic(df)
        df['Williams_R'] = compute_williams_r(df)
        df['ATR']        = compute_atr(df)
    else:
        df['Stoch_K']    = 50.0
        df['Williams_R'] = -50.0
        df['ATR']        = df['Volatility'] * df['Close']

    if 'Volume' in df.columns:
        obv              = compute_obv(df)
        df['OBV_Change'] = obv.pct_change()
    else:
        df['OBV_Change'] = 0.0

    #    Target   
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=False, inplace=True)
    return df


                   
# WALK-FORWARD VALIDATION  (honest accuracy estimate)
                   

def walk_forward_validate(df: pd.DataFrame, model_builder_fn,
                          n_splits: int = 5, test_pct: float = 0.15):
    """
    Splits data into n time-ordered folds.
    Each fold: train on all previous data, test on next slice.
    Returns per-fold accuracy + mean ± std.

    This is the honest way to evaluate a stock model.
    A single train/test split can be lucky or unlucky.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    X = df[FEATURE_COLS].values
    y = df['Target'].values
    n = len(X)

    fold_size   = int(n * test_pct)
    train_start = int(n * 0.4)   # need at least 40% of data for first training window

    results = []
    fold_idx = 0

    print(f"\n── Walk-Forward Validation ({n_splits} folds)        ────")
    for i in range(n_splits):
        test_end   = n - (n_splits - 1 - i) * fold_size
        test_start = test_end - fold_size
        if test_start < train_start:
            continue

        X_train, y_train = X[:test_start], y[:test_start]
        X_test,  y_test  = X[test_start:test_end], y[test_start:test_end]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)

        model = model_builder_fn(X_tr_sc.shape[1])
        # Quiet training inside walk-forward
        import tensorflow as tf
        model.fit(X_tr_sc, y_train, epochs=30, batch_size=32, verbose=0,
                  validation_split=0.1,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      patience=5, restore_best_weights=True, monitor='val_auc', mode='max')])

        probs = model.predict(X_te_sc, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)
        acc   = accuracy_score(y_test, preds)
        results.append(acc)
        fold_idx += 1
        print(f"   Fold {fold_idx}: train={test_start} rows, test={len(X_test)} rows → acc={acc:.1%}")

    mean_acc = np.mean(results)
    std_acc  = np.std(results)
    print(f"\n   Walk-Forward Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"   (This is your honest expected accuracy range: {mean_acc-std_acc:.1%} – {mean_acc+std_acc:.1%})")
    return results, mean_acc, std_acc


                   
# BACKTEST  (did following the signal make money historically?)
                   

def backtest_strategy(df: pd.DataFrame, probs: np.ndarray,
                      threshold: float = 0.55,
                      starting_capital: float = 10000.0,
                      test_start_idx: int = None) -> dict:
    """
    Simulates: BUY next day if ANN prob > threshold, else stay in cash.
    Compares vs buy-and-hold over the same period.

    Returns dict with equity curves and key stats.
    No transaction costs (pessimistic assumption — real costs would reduce returns).
    """
    if test_start_idx is None:
        test_start_idx = int(len(df) * 0.8)

    test_df = df.iloc[test_start_idx: test_start_idx + len(probs)].copy()
    returns = test_df['Return'].values[:len(probs)]

    # Strategy: hold when ANN says BUY, cash otherwise
    signals          = (probs > threshold).astype(float)
    strategy_returns = signals * returns

    # Equity curves
    strat_equity = starting_capital * np.cumprod(1 + strategy_returns)
    bh_equity    = starting_capital * np.cumprod(1 + returns)

    strat_final  = float(strat_equity[-1]) if len(strat_equity) > 0 else starting_capital
    bh_final     = float(bh_equity[-1])    if len(bh_equity) > 0 else starting_capital

    # Stats
    n_trades      = int(signals.sum())
    strat_ret     = (strat_final - starting_capital) / starting_capital
    bh_ret        = (bh_final - starting_capital) / starting_capital

    # Sharpe (annualised, daily returns, 252 trading days)
    if strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    roll_max   = np.maximum.accumulate(strat_equity)
    drawdowns  = (strat_equity - roll_max) / roll_max
    max_dd     = float(drawdowns.min())

    return {
        "strat_equity":  strat_equity,
        "bh_equity":     bh_equity,
        "strat_return":  strat_ret,
        "bh_return":     bh_ret,
        "n_trades":      n_trades,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "starting_cap":  starting_capital,
        "strat_final":   strat_final,
        "bh_final":      bh_final,
        "dates":         test_df['Date'].values[:len(probs)],
    }


                   
# FINNHUB API
                   

def get_real_news(ticker: str, finnhub_key: str, days_back: int = 7) -> list[dict]:
    """
    Fetches last `days_back` days of real company news from Finnhub.
    Returns list of dicts with keys: headline, source, datetime, url, summary
    Falls back to empty list on any error.

    FREE tier: 60 calls/min, 1 year history, no credit card needed.
    Get key at: https://finnhub.io/register
    """
    if not finnhub_key or finnhub_key.strip() == "":
        return []

    to_date   = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from":   from_date,
        "to":     to_date,
        "token":  finnhub_key.strip()
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not isinstance(data, list):
            return []

        news = []
        for item in data[:10]:
            headline = item.get("headline", "").strip()
            if headline and len(headline) > 10:
                news.append({
                    "headline": headline,
                    "source":   item.get("source", ""),
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                    "url":      item.get("url", ""),
                    "summary":  item.get("summary", "")[:200]
                })
        return news

    except Exception:
        return []


def get_news_headlines_only(ticker: str, finnhub_key: str, days_back: int = 7) -> list[str]:
    """Returns just the headline strings for NLP scoring."""
    articles = get_real_news(ticker, finnhub_key, days_back)
    return [a["headline"] for a in articles] if articles else []


                   
# FINBERT SENTIMENT  (upgraded from DistilBERT)
                   

def score_headlines_finbert(headlines: list[str], sent_model) -> tuple[float, list[tuple]]:
    """
    Scores headlines using FinBERT (finance-specific) or fallback model.

    FinBERT returns 3 labels: positive / negative / neutral
    Each mapped to: +1 / -1 / 0
    Weighted by confidence score.

    Returns:
        sentiment_score  — float in [-1, +1]
        per_headline     — list of (headline, score, confidence, label)
    """
    if not sent_model or not headlines:
        return 0.0, []

    label_map = {
        # FinBERT labels
        "positive": +1.0,
        "negative": -1.0,
        "neutral":   0.0,
        # DistilBERT fallback labels
        "POSITIVE": +1.0,
        "NEGATIVE": -1.0,
    }

    scores   = []
    per_item = []

    for headline in headlines:
        try:
            res   = sent_model(headline[:512])[0]
            label = res['label']
            conf  = float(res['score'])
            score = label_map.get(label, 0.0) * conf   # weight by confidence
            scores.append(score)
            per_item.append((headline, score, conf, label))
        except Exception:
            scores.append(0.0)
            per_item.append((headline, 0.0, 0.5, "neutral"))

    sentiment = float(np.mean(scores)) if scores else 0.0
    # clamp to [-1, +1]
    sentiment = max(-1.0, min(1.0, sentiment))
    return sentiment, per_item


                   
# DECISION ENGINE  
                   

def make_decision(prediction: float, sentiment: float,
                  threshold: float = 0.55) -> tuple[str, str]:
    """
    ANN carries 70% weight, FinBERT sentiment 30%.
    sentiment is already in [-1, +1] from score_headlines_finbert().
    Returns (decision, reason).
    """
    sent_norm = (sentiment + 1) / 2.0
    combined  = 0.70 * prediction + 0.30 * sent_norm

    ann_label  = "bullish" if prediction >= threshold else ("bearish" if prediction <= (1 - threshold) else "neutral")
    sent_label = "positive" if sentiment > 0.15 else ("negative" if sentiment < -0.15 else "neutral")

    if combined >= threshold:
        decision = "BUY"
        reason   = f"ANN {prediction:.0%} (w=70%) + {sent_label} news → combined {combined:.0%}"
    elif combined <= (1 - threshold):
        decision = "SELL"
        reason   = f"ANN {prediction:.0%} (w=70%) + {sent_label} news → combined {combined:.0%}"
    else:
        decision = "HOLD"
        if ann_label == "neutral":
            reason = f"ANN neutral ({prediction:.0%}), combined {combined:.0%} — no edge"
        else:
            reason = f"ANN {ann_label} but news {sent_label} — conflicting, combined {combined:.0%}"

    return decision, reason


                   
# MODEL I/O
                   

def save_artifacts(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Model  → {MODEL_PATH}")
    print(f"  ✓ Scaler → {SCALER_PATH}")


def load_artifacts():
    from tensorflow.keras.models import load_model
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
