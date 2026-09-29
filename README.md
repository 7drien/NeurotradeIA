# NeurotradeIA: Deep Learning Model for Price Ratio Prediction

> **⚠️ Disclaimer**: This project is intended purely for educational purposes, quantitative research, and experimentation. It does not constitute financial advice, and the models or strategies developed here should not be used for live trading with real capital.

NeurotradeIA is a quantitative trading and deep learning framework designed to predict future asset price movements using a multi-layer perceptron (MLP) with sequence-normalized market features. The model processes 128 consecutive candles and forecasts the price ratio **16 candles ahead** relative to the current candle.

---

## 🚀 Technical Architecture

### 🧠 Model Architecture & Specifications
* **Input Layer**: 1024-dimensional feature vector ($128 \text{ consecutive candles} \times 8 \text{ features}$).
* **Hidden Layers**: 4 dense layers with **LeakyReLU** activations:
  * Dense(256) + LeakyReLU($\alpha=0.01$)
  * Dense(64) + LeakyReLU($\alpha=0.01$)
  * Dense(16) + LeakyReLU($\alpha=0.01$)
  * Dense(8) + LeakyReLU($\alpha=0.01$)
* **Output Layer**: 1-dimensional output with **$2 \times \text{sigmoid}$** activation:
  $$\hat{y} = 2 \cdot \sigma(x) = \frac{2}{1 + e^{-x}} \in (0, 2)$$
* **Loss Function**: Mean Squared Error (MSE), evaluated with Mean Absolute Error (MAE) and Directional Accuracy.

```text
[Input: 1024 Features]
        │
   Dense(256) ──> LeakyReLU(0.01)
        │
   Dense(64)  ──> LeakyReLU(0.01)
        │
   Dense(16)  ──> LeakyReLU(0.01)
        │
   Dense(8)   ──> LeakyReLU(0.01)
        │
   Dense(1)   ──> 2 * Sigmoid Activation
        │
[Output: Future Price Ratio in (0, 2)]
```

---

### 📊 Input Features (8 Features × 128 Candles = 1024 Values)

Each candle within the 128-candle sliding window includes 8 quantitative features:
1. **Open**: Opening price.
2. **Close**: Closing price.
3. **Low**: Minimum price of the candle.
4. **High**: Maximum price of the candle.
5. **Volume**: Traded volume.
6. **RSI**: 14-period Relative Strength Index.
7. **Z-Score**: Rolling 100-period price z-score: $\frac{Close - \mu_{100}}{\sigma_{100}}$.
8. **Williams %R**: 14-period Williams %R oscillator: $\frac{High_{14} - Close}{High_{14} - Low_{14}} \times (-100)$.

---

### 📐 Sequence-Wise Normalization

To preserve physical market relationships and avoid lookahead bias, each 128-candle segment is normalized independently:

* **Prices (Open, Close, Low, High)**:
  All four price series are normalized using the **exact same mean ($\mu_{price}$) and standard deviation ($\sigma_{price}$)** calculated across the entire 128-candle segment:
  $$\mu_{price} = \frac{1}{4 \times 128} \sum_{i=1}^{128} (Open_i + Close_i + Low_i + High_i)$$
  $$\sigma_{price} = \sqrt{\frac{1}{4 \times 128} \sum_{i=1}^{128} ((Open_i - \mu_{price})^2 + (Close_i - \mu_{price})^2 + \dots)}$$
  $$Open_{norm} = \frac{Open - \mu_{price}}{\sigma_{price}}, \quad Close_{norm} = \frac{Close - \mu_{price}}{\sigma_{price}}, \quad \dots$$
  *Benefit*: Candle structures, spreads ($High - Low$), bodies ($|Close - Open|$), and relative price trends are preserved identically.
* **Volume**: Standardized per segment using its 128-candle mean and standard deviation:
  $$Volume_{norm} = \frac{Volume - \mu_{vol}}{\sigma_{vol}}$$
* **RSI**: Scaled from $[0, 100]$ to $[-1, 1]$ centered at 50:
  $$RSI_{norm} = \frac{RSI - 50}{50}$$
* **Z-Score**: Bounded/clipped to $[-5, 5]$ to suppress extreme volatility spikes.
* **Williams %R**: Scaled from $[-100, 0]$ to $[-1, 1]$ centered at -50:
  $$Williams\%R_{norm} = \frac{Williams\%R + 50}{50}$$

---

### 🎯 Target & Prediction Horizon

* **Horizon**: $K = 16$ candles into the future.
* **Target Ratio**:
  $$y = \frac{Close_{t + 16}}{Close_t}$$
  where $t$ is the index of the 128th candle (the present moment).
* **Target Interpretation**:
  * $y = 1.0$: Price remains unchanged (100% of current price).
  * $y > 1.0$: Price increases over the next 16 candles (up to 200%).
  * $y < 1.0$: Price decreases over the next 16 candles (down to 0%).
* **Zero Data Leakage**: By normalizing each segment independently and anchoring the target ratio strictly to the 128th candle, the model has no prior exposure to future candles.

---

### ⚡ Walk-Forward Backtesting & GUI

1. **Trading Signals**:
   * **Long (Buy)**: Generated when $\hat{y} > 1.001$ (predicted upward movement $> 0.1\%$).
   * **Short (Sell)**: Generated when $\hat{y} < 0.999$ (predicted downward movement $> 0.1\%$).
2. **Trade Duration**: Positions are held for 16 candles, directly aligned with the prediction horizon.
3. **Execution Simulation**: Walk-forward backtesting powered by **VectorBT**, incorporating realistic transaction costs (0.1%).
4. **Desktop GUI (`main.py`)**: Built with Tkinter and Matplotlib to monitor:
   * Dynamic hyperparameters (Epochs, Batch Size, Sequence Length, Horizon, Days to Load).
   * Real-time training loss, directional accuracy, and validation curves.
   * Cumulative portfolio equity curve and trade markers.

---

## 🗂️ Project Structure

```text
project_root/
│
├── data/
│   ├── raw/                 # Downloaded raw financial data
│   ├── processed/           # Transformed datasets
│   ├── cached/              # Serialized pickle caches (e.g. BTC-USD_1h.pkl)
│
├── ui/
│   ├── main_ui.py           # Tkinter interface with dynamic hyperparameter controls
│   ├── model_tester.py      # Architecture testing utilities
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Global default parameters (N_STEPS=128, K_STEPS=16, etc.)
│   ├── config_loader.py     # Configuration helper
│   ├── data_loader.py       # Data fetching with retry logic and caching
│   ├── features.py          # RSI, Z-Score, Williams %R indicator implementations
│   ├── preprocessing.py     # 1024-dim sequence generation and segment-wise normalization
│   ├── model.py             # 4-layer MLP (256-64-16-8) with LeakyReLU & 2*sigmoid
│   ├── train.py             # Training loop, callbacks, and validation
│   ├── callbacks.py         # Real-time UI progress logger and backtest runner
│   ├── backtesting.py       # VectorBT walk-forward backtest engine
│   ├── cython_extensions/   # Optional Cython optimizations
│
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
├── test_train_fast.py       # Fast single-epoch pipeline integration test
└── main.py                  # Main entry point launching GUI
```

---

## 📦 Installation & Usage

### 1. Prerequisites
* **Python 3.10+** (tested on Linux/macOS/Windows).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Fast Integration Test
Run a quick test training verifying data ingestion, feature extraction, sequence generation, model compilation, and backtesting:
```bash
python test_train_fast.py
```

### 4. Launch Desktop Interface
```bash
python main.py
```

---

## 📍 Key Methodological Highlights

* **Intra-Sequence Invariance**: Prices are normalized per 128-candle sequence with shared mean and standard deviation, avoiding data leakage across sliding windows and ensuring high generalization.
* **Bounded Target Space**: The $2 \cdot \sigma(x)$ output naturally covers $(0, 2)$, preventing extreme gradient explosions common in unbounded price regression.
* **Strict Chronological Splitting**: Training and validation sets are strictly ordered in time ($80\%$ train, $20\%$ validation) without random shuffling.
