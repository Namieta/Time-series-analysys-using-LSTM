
# Time-series-analysys-using-LSTM
This repository focuses on LSTM models and technical indicators like RSI, MACD, Bollinger Bands, and ROC. It includes data preprocessing, EDA, hyperparameter tuning, and stock price prediction to enhance investment strategies.

### Evaluation Metric
**Mean Absolute Percentage Error (MAPE):**


![MAPE Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BMAPE%7D%20%3D%20%5Cfrac%7B100%5C%25%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20%5Cleft%7C%20%5Cfrac%7By_i%20-%20%5Chat%7By%7D_i%7D%7By_i%7D%20%5Cright%7C)

Feature Selection 
**Random Forest Feature Importance:**  
Calculated using Mean Decrease in Impurity (MDI):


![Feature Importance Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BImportance%7D_j%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bt%3D1%7D%5EN%20%5CDelta%5Ctext%7BImpurity%7D_t%5E%7B(j)%7D)




**Features**
1. Data Collection: Downloads historical stock data for portfolio tickers using Yahoo Finance.
2. Technical Indicator Calculation: Includes RSI, MACD, Bollinger Bands, and ROC for detailed analysis.
3. Exploratory Data Analysis (EDA): Performs statistical analysis and detects outliers.
4. Preprocessing: Handles missing values, outliers, and scales data for model input.
5. LSTM Model Implementation: Builds and trains LSTM models with hyperparameter tuning.
6. Hyperparameter Optimization: Utilizes Bayesian Optimization for optimal model configuration.
7. Performance Metrics: Calculates MAPE (Mean Absolute Percentage Error) for model evaluation.
8. Visualization: Generates plots comparing actual vs predicted stock prices.

