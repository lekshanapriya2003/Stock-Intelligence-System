# Stock Intelligence System

An end-to-end machine learning system for stock trend prediction and decision support using technical indicators, neural networks, and sentiment analysis.

---

## Table of Contents

- Overview
- Features
- System Architecture
- Project Structure
- Installation
- Usage
- Model Details
- Evaluation
- Dashboard
- Generated Outputs
- Limitations
- Future Improvements
- Troubleshooting
- License

---

## Overview

This project builds a complete pipeline that:

1. Collects historical stock data
2. Generates technical indicators
3. Trains a machine learning model (ANN)
4. Evaluates performance with proper metrics
5. Provides an interactive dashboard for analysis

The model predicts short-term price direction:

- 1 → UP
- 0 → DOWN


## Features

- Automated data collection using Yahoo Finance
- Feature engineering (RSI, MACD, Moving Averages, Volatility)
- Artificial Neural Network (ANN) for prediction
- Sentiment analysis using Transformers
- Time-series aware train/test split (no data leakage)
- 8 diagnostic plots for model validation
- Interactive dashboard built with Streamlit


## System Architecture

Raw Stock Data (yfinance)
↓
Feature Engineering (utils.py)
↓
Data Scaling + Split
↓
ANN Model Training (train.py)
↓
Model + Scaler Saved
↓
Streamlit Dashboard (app.py)

```

---

## Project Structure

```

stock_intelligence/
├── utils.py
├── train.py
├── app.py
├── requirements.txt
├── model/
│   ├── ann_model.h5
│   ├── scaler.pkl
│   └── best_checkpoint.h5
└── plots/
├── 01_price_overview.png
├── 02_feature_distributions.png
├── 03_correlation_matrix.png
├── 04_training_history.png
├── 05_confusion_matrix.png
├── 06_roc_pr_curves.png
├── 07_predictions_vs_actual.png
└── 08_probability_distribution.png


## Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/stock_intelligence.git
cd stock_intelligence
```

---

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
transformers>=4.35.2
torch>=2.0.0
streamlit>=1.32.0
matplotlib>=3.7.0
seaborn>=0.13.0
joblib>=1.3.0
Pillow>=10.0.0
```

---

## Usage

### Step 1: Train the model

Run once:

```bash
python train.py
```

Custom runs:

```bash
python train.py --ticker TSLA
python train.py --ticker MSFT --epochs 60
python train.py --ticker GOOGL --period 2y --epochs 50 --batch_size 16
```

---

### Step 2: Launch dashboard

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Model Details

* Model: Artificial Neural Network (ANN)
* Input: Engineered tabular features
* Output: Binary classification
* Loss Function: Binary Crossentropy
* Activation: ReLU (hidden), Sigmoid (output)

### Why ANN?

ANN is appropriate for tabular feature-based prediction.
LSTM would be more suitable only if using raw sequential price data.

---

## Evaluation

Typical performance:

* Accuracy: 58% – 62%
* Balanced precision/recall
* ROC and PR curves included

### Important Reality Check

If a stock model shows:

* 80%+ accuracy → likely overfitting
* 90%+ accuracy → almost certainly wrong

---

## Dashboard

### Visual Components

* Price with Moving Averages and Bollinger Bands
* RSI indicator
* MACD and histogram
* Volatility and returns distribution

### Intelligence Panel

* Prediction probability
* BUY / SELL / HOLD decision
* Sentiment analysis
* Feature inspection

---

## Generated Outputs

| File                     | Purpose                      |
| ------------------------ | ---------------------------- |
| Price overview           | Validate feature correctness |
| Feature distributions    | Detect skew/outliers         |
| Correlation matrix       | Remove redundant features    |
| Training history         | Detect overfitting           |
| Confusion matrix         | Class performance            |
| ROC/PR curves            | Model quality                |
| Predictions vs actual    | Visual validation            |
| Probability distribution | Calibration                  |

---

## Limitations

* Financial markets are non-stationary
* ANN ignores sequential dependencies
* Sentiment is simplified (not real-time news)
* Model accuracy ceiling is inherently low

---

## Future Improvements

* Replace ANN with LSTM/Transformer models
* Integrate real-time news APIs
* Add backtesting engine
* Hyperparameter optimization
* Deploy as a web service (Docker + Cloud)

---

## Troubleshooting

### Missing modules

```bash
pip install yfinance transformers torch
```

---

### Model not found

```bash
python train.py
```

---

### Port already in use

```bash
streamlit run app.py --server.port 8502
```

---

## License

This project is for educational purposes only.

Do not use this system for real financial trading decisions.

---


