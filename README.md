# Apple Share Price Prediction Using Deep Learning

Forecasting the closing price of Apple Inc. (AAPL) stock from historical market data,
using two Multi-Layer Perceptron (MLP) implementations built independently in
TensorFlow/Keras and PyTorch.

The project covers the full workflow: data collection, preprocessing, feature
engineering, hyperparameter optimisation, training, evaluation, and model checkpointing.

## Dataset

Daily market data is pulled via the [`yfinance`](https://pypi.org/project/yfinance/)
library and includes:

| Feature | Description |
|---|---|
| AAPL Open / High / Low / Close | Apple's daily price action |
| AAPL Volume | Apple's daily trading volume |
| NASDAQ Composite | Broad tech-market index |
| S&P 500 | Broad market index |
| VIX | CBOE Volatility Index |
| DXY | US Dollar Index |
| US Treasury Yield | Interest-rate benchmark |
| SOX | PHLX Semiconductor Index |

**Target:** AAPL closing price.

## Preprocessing

1. Drop rows with missing values.
2. Normalise closing prices with Min-Max scaling.
3. Convert the series into supervised samples with a **30-day sliding window**.

The model predicts a sequence of future closes from a window of past closes, rather than
a single next-day value:

| Input (days) | Target (days) |
|---|---|
| 1–30 | 31–40 |
| 2–31 | 32–41 |
| 3–32 | 33–42 |

This lets the model do multi-step forecasting instead of single-step prediction.

## Models

**TensorFlow / Keras** — Sequential MLP, ReLU activations, dropout regularisation, Adam
optimiser.

**PyTorch** — Custom MLP, ReLU activations, Adam optimiser, checkpoint saving.

## Hyperparameter Optimisation

The TensorFlow model's hyperparameters were tuned with [Optuna](https://optuna.org/),
searching over the number of hidden layers, neurons per layer, learning rate, dropout
rate, and batch size. The best configuration is saved in
[`best_hyperparams.json`](best_hyperparams.json).

## Repository Structure

```
Apple_Share_Prediction/
├── MLP_TensorFlow.ipynb        # TensorFlow/Keras implementation
├── MLP_Torch.ipynb             # PyTorch implementation
├── best_hyperparams.json       # Optuna-tuned hyperparameters
├── mlp_model_tensorflow.keras  # Trained TensorFlow model
├── mlp_torch_checkpoint.pth    # Trained PyTorch checkpoint
├── requirements.txt
└── README.md
```

## Results

Both implementations learn the general trend of Apple's stock price, but predictions
can diverge noticeably from actual prices. This is expected:

- Stock prices are driven by news, earnings, macroeconomic events, and sentiment — none
  of which this model sees.
- The model uses only historical price data, with no technical indicators or economic
  features.
- An MLP treats each 30-day window as a fixed feature vector, with no explicit modelling
  of long-term temporal dependencies (unlike recurrent or attention-based architectures).
- Financial markets are noisy and non-stationary, so exact price prediction is inherently
  unreliable.

The model is a useful demonstration of applying deep learning to financial time series,
capturing overall trend rather than precise future prices.

## Future Improvements

- LSTM-based forecasting
- Transformer-based forecasting
- Technical indicators (Moving Average, RSI, MACD, Bollinger Bands)
- Additional market/economic features

## License

Developed for educational purposes as part of a machine learning portfolio.
