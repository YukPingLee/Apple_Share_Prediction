------------------------------------------------------------------------------------------------
Project: Apple Share Price Prediction Using Deep Learning
------------------------------------------------------------------------------------------------
Overview
------------------------------------------------------------------------------------------------
This project investigates the use of deep learning models for forecasting the closing price of Apple Inc. (AAPL) stock using historical market data.

Two Multi-Layer Perceptron (MLP) models were implemented and compared:

• TensorFlow/Keras MLP

• PyTorch MLP

The project covers the complete machine learning workflow, including data collection, preprocessing, feature engineering, hyperparameter optimisation, model training, evaluation, and saving trained models for future inference.

------------------------------------------------------------------------------------------------
Dataset
------------------------------------------------------------------------------------------------
Historical stock market data was downloaded using the Yahoo Finance API through the yfinance Python library.
The dataset contains daily trading information, including:

- Apple Open Price

- Apple High Price

- Apple Low Price

- Apple Close Price

- Apple Trading Volume

- NASDAQ Composite Index

- S&P 500 Index

- CBOE Volatility Index (VIX)

- US Dollar Index (DXY)

- US Treasury Yield

- PHLX Semiconductor Index (SOX)

The closing price was used as the prediction target.
 
------------------------------------------------------------------------------------------------
Data Preprocessing
------------------------------------------------------------------------------------------------
Before training the models:

•	Missing values were removed.

•	Closing prices were normalised using Min-Max Scaling.

•	The data was transformed into supervised learning samples using a 30-day sliding window.

------------------------------------------------------------------------------------------------
Sliding Window
------------------------------------------------------------------------------------------------
Instead of predicting tomorrow's price from only today's price, the model uses the previous 30 trading days as input.
For example:

Input	Target
Days 1–30	Closing Prices for Days 31-40
Days 2–31	Closing Prices for Days 32-41
Days 3–32	Closing Prices for Days 33-42

This approach enables the model to perform multi-step forecasting, predicting an entire sequence of future prices instead of only the next trading day.

------------------------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------------------------
Two implementations were developed:

TensorFlow / Keras

•	Sequential Multi-Layer Perceptron

•	ReLU activation

•	Dropout regularisation

•	Adam optimiser

PyTorch

•	Custom Multi-Layer Perceptron

•	ReLU activation

•	Adam optimiser

•	Model checkpoint saving
 
------------------------------------------------------------------------------------------------
Hyperparameter Optimisation
------------------------------------------------------------------------------------------------
Hyperparameters for the TensorFlow model were optimised using Optuna.

The search included:

•	Number of hidden layers

•	Number of neurons

•	Learning rate

•	Dropout rate

•	Batch size

The best configuration is stored in:

best_hyperparams.json

------------------------------------------------------------------------------------------------
Repository Structure
------------------------------------------------------------------------------------------------
Apple_Share_Prediction/
│
├── MLP_TensorFlow.ipynb
├── MLP_Torch.ipynb
├── best_hyperparams.json
├── mlp_model_tensorflow.keras
├── mlp_torch_checkpoint.pth
├── requirements.txt
└── README.md

------------------------------------------------------------------------------------------------
Results and Discussion
------------------------------------------------------------------------------------------------
Both TensorFlow and PyTorch implementations successfully learned the general trend of Apple's stock price.

However, the predicted prices can differ noticeably from the actual market prices. This is expected for several reasons:

• Stock prices are highly volatile and influenced by many external factors such as company news, earnings reports, macroeconomic events, and investor sentiment.

• The model only uses historical price information and does not incorporate news, technical indicators, market indices, or economic data.

• A Multi-Layer Perceptron treats each 30-day window as a fixed feature vector and does not explicitly model long-term temporal dependencies like recurrent or attention-based architectures.

• Predicting future stock prices is inherently uncertain because financial markets are noisy and non-stationary.

Although the model may not accurately predict the exact future price, it is able to capture the overall market trend and demonstrates the application of deep learning techniques to financial time series forecasting.

------------------------------------------------------------------------------------------------
Future Improvements
------------------------------------------------------------------------------------------------
Potential extensions include:

•	Implement Long Short-Term Memory (LSTM) networks

•	Implement Transformer-based forecasting models

•	Add technical indicators (e.g. Moving Average, RSI, MACD, Bollinger Bands)

•	Include additional market and economic features

------------------------------------------------------------------------------------------------
License
------------------------------------------------------------------------------------------------
This project was developed for educational purposes and as part of a machine learning portfolio.

