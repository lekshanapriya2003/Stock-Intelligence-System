
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc,
    precision_recall_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from utils import (
    create_features, save_artifacts, backtest_strategy,
    walk_forward_validate, FEATURE_COLS, MODEL_DIR
)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#    Palette       
sns.set_theme(style="darkgrid", palette="muted")
DARK_BG   = "#0d0d14"
GRID_COL  = "#1e1e2a"
TEXT_COL  = "#e8e6e0"
GOLD      = "#e8c97a"
ACCENT    = "#f0a050"
RED_COL   = "#e05050"
GREEN_COL = "#60c870"
BLUE_COL  = "#6090e8"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(GOLD)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.6)
    if title:   ax.set_title(title, fontsize=11, pad=8)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"      Saved → {path}")


                   
# 1. DATA
                   

def fetch_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  STEP 1 — Downloading {ticker} ({period})")
    print(f"{'='*60}")
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        sys.exit(f"ERROR: No data for ticker '{ticker}'")
    print(f"  Rows downloaded : {len(df)}")
    print(f"  Date range      : {df.index[0].date()} → {df.index[-1].date()}")
    return df


                   
# 2. EDA PRINTOUT
                   

def print_eda(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  STEP 2 — Exploratory Data Analysis")
    print(f"{'='*60}")

    print("\n── HEAD (first 5 rows)             ─")
    print(df[['Date','Close','Return','MA10','RSI','Volatility','Target']].head().to_string(index=False))

    print("\n── TAIL (last 5 rows)             ──")
    print(df[['Date','Close','Return','MA10','RSI','Volatility','Target']].tail().to_string(index=False))

    print("\n── SHAPE                  ────")
    print(f"   Rows: {df.shape[0]}   Columns: {df.shape[1]}")

    print("\n── DESCRIBE (features)             ─")
    print(df[FEATURE_COLS].describe().round(4).to_string())

    print("\n── NULL CHECK             ──────────")
    nulls = df[FEATURE_COLS + ['Target']].isnull().sum()
    if nulls.sum() == 0:
        print("   No null values ✓")
    else:
        print(nulls[nulls > 0])

    print("\n── CLASS BALANCE             ───────")
    counts = df['Target'].value_counts()
    total  = len(df)
    print(f"   UP days   (1): {counts.get(1, 0):4d}  ({counts.get(1,0)/total:.1%})")
    print(f"   DOWN days (0): {counts.get(0, 0):4d}  ({counts.get(0,0)/total:.1%})")


                   
# 3. EDA PLOTS
                   

def plot_price_overview(df: pd.DataFrame, ticker: str):
    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['Date'], df['Close'],    color=TEXT_COL, lw=0.9, label='Close', alpha=0.9)
    ax1.plot(df['Date'], df['MA10'],     color=GOLD,     lw=1.2, label='MA10')
    ax1.plot(df['Date'], df['MA50'],     color=ACCENT,   lw=1.2, label='MA50')
    ax1.fill_between(df['Date'], df['Lower_Band'], df['Upper_Band'],
                     alpha=0.12, color=BLUE_COL, label='Bollinger Bands')
    ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.6)
    _style_ax(ax1, f"{ticker} — Closing Price + Moving Averages + Bollinger Bands", ylabel="Price (USD)")

    ax2 = fig.add_subplot(gs[1])
    colors = [GREEN_COL if r >= 0 else RED_COL for r in df['Return']]
    ax2.bar(df['Date'], df['Return'], color=colors, width=1.0, alpha=0.85)
    ax2.axhline(0, color=TEXT_COL, lw=0.6, ls='--')
    _style_ax(ax2, "Daily Return (%)", ylabel="Return")

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(df['Date'], df['RSI'], color=BLUE_COL, lw=1.1)
    ax3.axhline(70, color=RED_COL,   lw=0.9, ls='--', label='Overbought (70)')
    ax3.axhline(30, color=GREEN_COL, lw=0.9, ls='--', label='Oversold (30)')
    ax3.fill_between(df['Date'], df['RSI'], 70, where=(df['RSI'] >= 70), alpha=0.2, color=RED_COL)
    ax3.fill_between(df['Date'], df['RSI'], 30, where=(df['RSI'] <= 30), alpha=0.2, color=GREEN_COL)
    ax3.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.6)
    _style_ax(ax3, "RSI (14)", ylabel="RSI")

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(df['Date'], df['MACD'],        color=GOLD,   lw=1.1, label='MACD')
    ax4.plot(df['Date'], df['Signal_Line'], color=ACCENT, lw=1.1, label='Signal')
    macd_hist = df['MACD'] - df['Signal_Line']
    hist_colors = [GREEN_COL if v >= 0 else RED_COL for v in macd_hist]
    ax4.bar(df['Date'], macd_hist, color=hist_colors, width=1.0, alpha=0.6)
    ax4.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.6)
    _style_ax(ax4, "MACD", ylabel="Value")

    _save(fig, "01_price_overview.png")


def plot_feature_distributions(df: pd.DataFrame):
    n = len(FEATURE_COLS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.2), facecolor=DARK_BG)
    fig.suptitle("Feature Distributions", color=GOLD, fontsize=13, y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        ax.set_facecolor(DARK_BG)
        data = df[col].dropna()
        ax.hist(data, bins=50, color=BLUE_COL, alpha=0.75, edgecolor='none')
        ax.axvline(data.mean(),   color=GOLD,   lw=1.2, ls='--', label='mean')
        ax.axvline(data.median(), color=ACCENT, lw=1.2, ls=':',  label='median')
        _style_ax(ax, col)
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.4)

    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "02_feature_distributions.png")


def plot_correlation_matrix(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    corr = df[FEATURE_COLS + ['Target']].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", linewidths=0.4,
                linecolor=DARK_BG, square=True, ax=ax,
                annot_kws={"size": 7},
                cbar_kws={"shrink": 0.7})

    ax.set_title("Feature Correlation Matrix", color=GOLD, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    _save(fig, "03_correlation_matrix.png")


                   
# 4. WALK-FORWARD PLOT
                   

def plot_walk_forward(fold_accs: list, mean_acc: float, std_acc: float):
    """Plots per-fold accuracy to show how the model performs across time."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    folds = list(range(1, len(fold_accs) + 1))
    colors = [GREEN_COL if a > 0.55 else (GOLD if a > 0.50 else RED_COL) for a in fold_accs]
    ax.bar(folds, fold_accs, color=colors, alpha=0.85, width=0.6)
    ax.axhline(mean_acc, color=GOLD, lw=1.5, ls='--', label=f'Mean {mean_acc:.1%}')
    ax.axhline(0.50, color=TEXT_COL, lw=0.8, ls=':', alpha=0.5, label='Random baseline (50%)')
    ax.fill_between([0.5, len(folds) + 0.5], [mean_acc - std_acc]*2, [mean_acc + std_acc]*2,
                    alpha=0.12, color=GOLD, label=f'±1σ ({std_acc:.1%})')

    for fold, acc in zip(folds, fold_accs):
        ax.text(fold, acc + 0.005, f"{acc:.1%}", ha='center', va='bottom',
                color=TEXT_COL, fontsize=8, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax,
              f"Walk-Forward Validation — {mean_acc:.1%} ± {std_acc:.1%}\n"
              f"(honest accuracy range: {mean_acc-std_acc:.1%}–{mean_acc+std_acc:.1%})",
              ylabel="Accuracy")
    plt.tight_layout()
    _save(fig, "04_walk_forward_validation.png")


                   
# 5. MODEL
                   

def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ], name="StockANN")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def train_model(df: pd.DataFrame, epochs: int = 40, batch_size: int = 32):
    print(f"\n{'='*60}")
    print("  STEP 3 — Preparing Data & Training Final ANN")
    print(f"{'='*60}")

    X = df[FEATURE_COLS].values
    y = df['Target'].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series split — NO shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Class balance (train) — UP: {y_train.mean():.1%}  DOWN: {1-y_train.mean():.1%}")

    model = build_ann(X_train.shape[1])
    print(f"\n── Model Summary             ────────")
    model.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True,
                      monitor='val_auc', mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-5, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_DIR, "best_checkpoint.h5"),
                        monitor='val_auc', mode='max',
                        save_best_only=True, verbose=0),
    ]

    print(f"\n── Training ({epochs} epochs max, early stop on val_auc)   ───")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, scaler, history, X_test, y_test


                   
# 6. EVALUATION
                   

def evaluate(model, X_test, y_test):
    print(f"\n{'='*60}")
    print("  STEP 4 — Evaluation")
    print(f"{'='*60}")

    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)

    print(f"\n  Accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"\n── Classification Report             ")
    print(classification_report(y_test, preds,
                                target_names=["DOWN (0)", "UP (1)"],
                                digits=4))
    print(f"── Confusion Matrix             ─────")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return probs, preds


                   
# 7. TRAINING & EVALUATION PLOTS
                   

def plot_training_history(history):
    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=DARK_BG)
    fig.suptitle("Training History", color=GOLD, fontsize=13, y=1.01)

    pairs = [
        ('loss',      'val_loss',      "Loss (Binary Crossentropy)"),
        ('accuracy',  'val_accuracy',  "Accuracy"),
        ('auc',       'val_auc',       "AUC"),
        ('precision', 'val_precision', "Precision"),
    ]

    for ax, (train_key, val_key, title) in zip(axes.flatten(), pairs):
        ax.set_facecolor(DARK_BG)
        ax.plot(epochs, h[train_key], color=GOLD,     lw=1.8, label='Train',      marker='o', markersize=3)
        ax.plot(epochs, h[val_key],   color=BLUE_COL, lw=1.8, label='Validation', marker='s', markersize=3)
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
        best_ep = int(np.argmin(h['val_loss'])) if 'loss' in train_key else int(np.argmax(h[val_key]))
        ax.axvline(best_ep + 1, color=ACCENT, lw=1.0, ls='--', alpha=0.7, label=f'Best@ep{best_ep+1}')
        _style_ax(ax, title, xlabel="Epoch")

    plt.tight_layout()
    _save(fig, "05_training_history.png")


def plot_confusion_matrix_chart(y_test, preds):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    cm = confusion_matrix(y_test, preds)
    cmap = sns.light_palette(GOLD, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                linewidths=1, linecolor=DARK_BG,
                xticklabels=["Pred DOWN", "Pred UP"],
                yticklabels=["Actual DOWN", "Actual UP"],
                annot_kws={"size": 14, "color": DARK_BG})
    ax.set_title("Confusion Matrix", color=GOLD, fontsize=12)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    _save(fig, "06_confusion_matrix.png")


def plot_roc_and_pr(y_test, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc      = auc(fpr, tpr)
    ax1.set_facecolor(DARK_BG)
    ax1.plot(fpr, tpr, color=GOLD, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], color=GRID_COL, lw=1, ls='--', label="Random")
    ax1.fill_between(fpr, tpr, alpha=0.12, color=GOLD)
    ax1.legend(fontsize=9, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax1, "ROC Curve", "False Positive Rate", "True Positive Rate")

    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc        = auc(rec, prec)
    ax2.set_facecolor(DARK_BG)
    ax2.plot(rec, prec, color=ACCENT, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax2.axhline(y_test.mean(), color=GRID_COL, lw=1, ls='--', label="Baseline")
    ax2.fill_between(rec, prec, alpha=0.12, color=ACCENT)
    ax2.legend(fontsize=9, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax2, "Precision-Recall Curve", "Recall", "Precision")

    plt.tight_layout()
    _save(fig, "07_roc_pr_curves.png")


def plot_prediction_vs_actual(y_test, probs, n=200):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), facecolor=DARK_BG)
    x = np.arange(min(n, len(y_test)))

    ax1.set_facecolor(DARK_BG)
    ax1.plot(x, probs[:n],     color=GOLD,     lw=1.2, label='Pred probability', alpha=0.9)
    ax1.scatter(x, y_test[:n], color=BLUE_COL, s=8,    label='Actual (0/1)',     alpha=0.7, zorder=3)
    ax1.axhline(0.5, color=TEXT_COL, lw=0.8, ls='--', alpha=0.6, label='Decision boundary (0.5)')
    ax1.fill_between(x, 0.5, probs[:n], where=(probs[:n] > 0.5), alpha=0.12, color=GREEN_COL, label='BUY zone')
    ax1.fill_between(x, 0.5, probs[:n], where=(probs[:n] < 0.5), alpha=0.12, color=RED_COL, label='SELL zone')
    ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax1, f"Prediction Probability vs Actual (first {n} test samples)", ylabel="Prob / Label")

    correct = (probs[:n] > 0.5).astype(int) == y_test[:n]
    ax2.set_facecolor(DARK_BG)
    ax2.bar(x[correct],  [1]*correct.sum(),  color=GREEN_COL, width=1.0, alpha=0.8, label='Correct')
    ax2.bar(x[~correct], [1]*sum(~correct),  color=RED_COL,   width=1.0, alpha=0.8, label='Wrong')
    ax2.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax2, "Prediction Errors", xlabel="Sample index")

    plt.tight_layout()
    _save(fig, "08_predictions_vs_actual.png")


def plot_probability_distribution(probs, y_test):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.hist(probs[y_test == 0], bins=50, color=RED_COL,   alpha=0.6, label="Actual DOWN (0)", edgecolor='none', density=True)
    ax.hist(probs[y_test == 1], bins=50, color=GREEN_COL, alpha=0.6, label="Actual UP (1)",   edgecolor='none', density=True)
    ax.axvline(0.5, color=GOLD, lw=1.5, ls='--', label="Threshold (0.5)")
    ax.legend(fontsize=9, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax, "Predicted Probability Distribution by True Class", "Predicted Probability", "Density")
    _save(fig, "09_probability_distribution.png")


                   
# 8. BACKTEST PLOT  
                   

def plot_backtest(bt: dict, ticker: str):
    """
    Equity curve: ANN strategy vs buy-and-hold.
    Honest display: includes a disclaimer that this is in-sample test data.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), facecolor=DARK_BG,
                                   gridspec_kw={'height_ratios': [3, 1]})

    dates = bt['dates']
    ax1.set_facecolor(DARK_BG)
    ax1.plot(dates, bt['strat_equity'], color=GOLD,     lw=2.0, label=f"ANN Strategy  {bt['strat_return']:+.1%}")
    ax1.plot(dates, bt['bh_equity'],    color=BLUE_COL, lw=1.8, label=f"Buy & Hold    {bt['bh_return']:+.1%}", ls='--')
    ax1.fill_between(dates, bt['strat_equity'], bt['bh_equity'],
                     where=(bt['strat_equity'] >= bt['bh_equity']),
                     alpha=0.12, color=GREEN_COL, label='Outperforming')
    ax1.fill_between(dates, bt['strat_equity'], bt['bh_equity'],
                     where=(bt['strat_equity'] < bt['bh_equity']),
                     alpha=0.12, color=RED_COL, label='Underperforming')

    ax1.legend(fontsize=9, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax1,
              f"{ticker} Backtest — Strategy vs Buy & Hold\n"
              f"      In-sample test period only — NOT a forward prediction",
              ylabel=f"Portfolio Value (started ${bt['starting_cap']:,.0f})")

    # Drawdown
    roll_max  = np.maximum.accumulate(bt['strat_equity'])
    drawdown  = (bt['strat_equity'] - roll_max) / roll_max * 100
    ax2.set_facecolor(DARK_BG)
    ax2.fill_between(dates, drawdown, 0, color=RED_COL, alpha=0.6)
    ax2.axhline(bt['max_drawdown'] * 100, color=ACCENT, lw=0.8, ls='--',
                label=f"Max DD {bt['max_drawdown']:.1%}")
    ax2.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
    _style_ax(ax2, "Drawdown (%)", ylabel="%")

    # Stats text box
    stats_text = (
        f"Sharpe: {bt['sharpe']:.2f}   "
        f"Max DD: {bt['max_drawdown']:.1%}   "
        f"Trades: {bt['n_trades']}   "
        f"Final: ${bt['strat_final']:,.0f}"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', va='bottom',
             color=GOLD, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=GRID_COL, alpha=0.7))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "10_backtest_equity_curve.png")


                   
# MAIN
                   

def main():
    parser = argparse.ArgumentParser(description="Train Stock ANN v3")
    parser.add_argument("--ticker",     default="AAPL",  type=str)
    parser.add_argument("--period",     default="3y",    type=str)
    parser.add_argument("--epochs",     default=40,      type=int)
    parser.add_argument("--batch_size", default=32,      type=int)
    parser.add_argument("--skip_wfv",  action="store_true",
                        help="Skip walk-forward validation (faster, less honest)")
    args = parser.parse_args()

    # 1. Data
    raw_df = fetch_data(args.ticker, args.period)

    # 2. Features + EDA
    print(f"\n{'='*60}")
    print("  STEP 2 — Feature Engineering")
    print(f"{'='*60}")
    df = create_features(raw_df)
    print_eda(df)

    # 3. EDA plots
    print(f"\n{'='*60}")
    print("  Generating EDA plots...")
    print(f"{'='*60}")
    plot_price_overview(df, args.ticker)
    plot_feature_distributions(df)
    plot_correlation_matrix(df)

    # 4. Walk-forward validation (honest accuracy)
    fold_accs, mean_acc, std_acc = [], 0.0, 0.0
    if not args.skip_wfv:
        print(f"\n{'='*60}")
        print("  STEP 3a — Walk-Forward Validation (5 folds)")
        print(f"{'='*60}")
        print("  (This takes ~3–5 min. Use --skip_wfv to bypass.)")
        fold_accs, mean_acc, std_acc = walk_forward_validate(df, build_ann, n_splits=5)
        plot_walk_forward(fold_accs, mean_acc, std_acc)
    else:
        print("\n  ⚡ Skipping walk-forward validation (--skip_wfv set)")

    # 5. Train final model
    model, scaler, history, X_test, y_test = train_model(
        df, args.epochs, args.batch_size
    )

    # 6. Evaluate
    probs, preds = evaluate(model, X_test, y_test)

    # 7. Walk-forward accuracy context
    if fold_accs:
        single_split_acc = accuracy_score(y_test, preds)
        print(f"\n── Accuracy Context             ─────")
        print(f"   Single split accuracy  : {single_split_acc:.1%}")
        print(f"   Walk-forward mean ± std: {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"   Honest expected range  : {mean_acc-std_acc:.1%} – {mean_acc+std_acc:.1%}")
        if single_split_acc > mean_acc + std_acc:
            print(f"         Single-split acc is ABOVE walk-forward range — may be a lucky split")

    # 8. Backtest
    print(f"\n{'='*60}")
    print("  STEP 4b — Backtesting ANN Signal on Test Period")
    print(f"{'='*60}")
    test_start_idx = int(len(df) * 0.8)
    bt = backtest_strategy(df, probs, threshold=0.55,
                           starting_capital=10000.0,
                           test_start_idx=test_start_idx)
    print(f"  Starting capital : ${bt['starting_cap']:,.0f}")
    print(f"  ANN strategy     : ${bt['strat_final']:,.0f}  ({bt['strat_return']:+.1%})")
    print(f"  Buy & hold       : ${bt['bh_final']:,.0f}  ({bt['bh_return']:+.1%})")
    print(f"  Sharpe ratio     : {bt['sharpe']:.2f}")
    print(f"  Max drawdown     : {bt['max_drawdown']:.1%}")
    print(f"  Number of trades : {bt['n_trades']}")

    # 9. Evaluation + backtest plots
    print(f"\n{'='*60}")
    print("  Generating evaluation & backtest plots...")
    print(f"{'='*60}")
    plot_training_history(history)
    plot_confusion_matrix_chart(y_test, preds)
    plot_roc_and_pr(y_test, probs)
    plot_prediction_vs_actual(y_test, probs)
    plot_probability_distribution(probs, y_test)
    plot_backtest(bt, args.ticker)

    # 10. Save
    print(f"\n{'='*60}")
    print("  STEP 5 — Saving artifacts")
    print(f"{'='*60}")
    save_artifacts(model, scaler)
    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

    print(f"\n{'='*60}")
    print("  ✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Plots saved → {PLOTS_DIR}/  (10 plots)")
    print(f"  Model saved → {MODEL_DIR}/")
    print(f"\n  Now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
