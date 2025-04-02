# Time-series-analysys-using-LSTM
This repository focuses on LSTM models and technical indicators like RSI, MACD, Bollinger Bands, and ROC. It includes data preprocessing, EDA, hyperparameter tuning, and stock price prediction to enhance investment strategies.

**Features**
1. Data Collection: Downloads historical stock data for portfolio tickers using Yahoo Finance.
2. Technical Indicator Calculation: Includes RSI, MACD, Bollinger Bands, and ROC for detailed analysis.
3. Exploratory Data Analysis (EDA): Performs statistical analysis and detects outliers.
4. Preprocessing: Handles missing values, outliers, and scales data for model input.
5. LSTM Model Implementation: Builds and trains LSTM models with hyperparameter tuning.
6. Hyperparameter Optimization: Utilizes Bayesian Optimization for optimal model configuration.
7. Performance Metrics: Calculates MAPE (Mean Absolute Percentage Error) for model evaluation.
8. Visualization: Generates plots comparing actual vs predicted stock prices.


├── stock_data_LSTM_Future_1/       # Directory for storing downloaded stock data
├── keras_tuner/                    # Directory for storing hyperparameter tuning results
├── stock_predictions_plot_hyperparameter_1/ # Directory for saving prediction plots
├── holdings.xlsx                   # Input file containing portfolio tickers
├── LSTM.py                         # Main Python script
└── README.md                       # Project documentation

**Dependencies**
This project uses the following libraries:

pandas - For data manipulation
numpy - For numerical computations
matplotlib & seaborn - For visualization
yfinance - For downloading stock data
tensorflow & keras_tuner - For building and optimizing LSTM models
scipy & statsmodels - For statistical analysis
sklearn - For preprocessing and feature selection

**Visualization**

