# NeurotradeIA: Deep Learning Model for Trading

NeurotradeIA is a comprehensive pipeline designed to train deep learning models that predict the direction or percentage return of upcoming financial market candles based on historical data. The project seamlessly integrates data preparation, advanced preprocessing, model training, robust backtesting, and a graphical user interface (GUI) into a single cohesive ecosystem.

---

## 🚀 Technical Characteristics

### 🧠 Deep Learning Architecture & Strategy
* **Supported Models**: The architecture supports various sequential neural networks, including **CNNs (Convolutional Neural Networks)**, **LSTMs (Long Short-Term Memory)**, **GRUs**, and **Transformers**.
* **Target Prediction**: 
  * **Regression**: Predicting the exact percentage return over the next *k* candles (MSE Loss).
  * **Classification**: Predicting the binary probability of an upward or downward movement (BCE Loss).
  * **Probabilistic**: Quantile loss implementation for models evaluating uncertainty.
* **Strict Chronological Validation**: To prevent data leakage and look-ahead bias, the dataset is split chronologically into **Train -> Validation -> Test**. 
  * *Train* is strictly the past.
  * *Validation* is the period immediately following the train set.
  * *Test* is completely unseen future data.
  * No global shuffling is permitted, and normalization is exclusively fitted on the training segment.

### ⚡ Performance Optimization
* **Cython Extensions**: Critical, repetitive, and computationally heavy time-series operations (such as generating sliding windows and rolling statistics) are implemented in C using **Cython**. This significantly accelerates data preprocessing and training times.
* **Pickle Data Caching**: Preprocessed DataFrames are serialized into `.pkl` files (stored in `data/cached/`). This allows for nearly instant data loading upon startup, avoiding the redundant recalculation of features or re-reading of heavy CSV files on every run.

### 🔄 End-to-End Pipeline
1. **Data Ingestion**: Automated financial data downloading via `yfinance` directly into `data/raw/`.
2. **Preprocessing**: Feature engineering, normalization, and sequence generation (sliding windows of size *n* context candles targeting *k* future candles).
3. **Training & Validation**: 
   * Custom DataLoaders designed to preserve sequentiality.
   * Incorporation of early stopping and learning rate schedulers based on real-time validation metrics.
   * Auto-saving the best-performing model weights (`.h5` or `.pt`).
4. **Walk-Forward Backtesting**: Realistic simulation of trading strategies based on the model's signals.
   * Incorporates real-world constraints: transaction costs, spread, and slippage.
   * Generates actionable trading metrics: Sharpe Ratio, Max Drawdown, Win/Loss Ratio, and Compound Annual Growth Rate (CAGR).

### 🖥️ User Interface (Tkinter)
* **Training Hub**: A dedicated Tkinter GUI (`main.py` and `ui/main_ui.py`) allows users to:
  * Select financial assets and define sequence lengths (*n* and *k*).
  * Choose and configure the model architecture.
  * Monitor loss and validation metric curves in real-time.
* **Threaded Execution**: The training loop runs in an isolated background thread, ensuring the Tkinter UI remains responsive and does not freeze during intensive epochs.

---

## 🗂️ Project Structure

```text
project_root/
│
├── data/
│   ├── raw/                 # Raw data downloaded via yfinance
│   ├── processed/           # Transformed and normalized data
│   ├── cached/              # Pickle (.pkl) files for accelerated loading
│
├── ui/
│   ├── main_ui.py           # Main Tkinter interface
│   ├── model_tester.py      # Advanced window for architecture and parameter testing
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Global parameters (n, k, ticker, etc.)
│   ├── data_loader.py       # Downloading and pickle caching logic
│   ├── preprocessing.py     # Normalization and sequence creation
│   ├── model.py             # Neural network definitions (LSTM, CNN, etc.)
│   ├── train.py             # Training loop and validation logic
│   ├── backtesting.py       # Backtest engine with costs, slippage, walk-forward
│   ├── cython_extensions/   # Compiled Cython code for calculations
│   │   ├── fast_ops.pyx     # Optimized functions (rolling windows, etc.)
│   │   ├── setup.py         # Cython compilation script
│
├── README.md                # Project documentation
├── requirements.txt         # Required Python libraries
└── .gitignore               # Ignored files
```

---

## 📦 Installation Guide

Follow these steps to install and run the project on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system. 

### 2. Install Dependencies
Navigate to the project root directory and install the required Python packages using pip:

```bash
pip install -r requirements.txt
```
*Key libraries include: `tensorflow` (or `pytorch`), `pandas`, `numpy`, `yfinance`, `scikit-learn`, `gymnasium`, `stable-baselines3`, and `cython`.*

### 3. Compile Cython Extensions
To benefit from the performance optimizations, you must compile the Cython C extensions before running the application for the first time.

```bash
cd src/cython_extensions/
python setup.py build_ext --inplace
cd ../../
```
*Note: This will generate compiled shared objects (`.so` on Linux/Mac, `.pyd` on Windows) that Python can import natively.*

### 4. Run the Application
Once dependencies are installed and the Cython extensions are compiled, launch the main Tkinter GUI:

```bash
python main.py
```

---

## 📍 Important Notes & Best Practices

* **Zero Data Leakage**: Never mix future and present data. Normalization parameters (mean, standard deviation) must strictly be calculated on the training set and applied to the validation/test sets.
* **Testing Horizons**: Experiment with multiple horizons for `n` (context size) and `k` (target size).
* **Baselines**: Always compare complex neural networks against simple baseline models (e.g., Buy & Hold or simple moving average crossovers).
* **Market Noise**: Financial markets are mostly noise with slight drift. Beware of overfitting; if training loss approaches zero, the model is likely memorizing noise.

---

## 🔮 Future Roadmap

* **Reinforcement Learning Integration**: Implement a Reinforcement Learning module (using `gymnasium` and `stable-baselines3`) to optimize dynamic position sizing and portfolio management.
* **Advanced Ensembling**: Test probabilistic models and ensemble methods to gauge signal uncertainty.
* **Rolling Retrain**: Automate periodic model retraining to adapt to evolving market regimes (drift monitoring).
