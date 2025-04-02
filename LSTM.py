import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from scipy.optimize import minimize
import statsmodels.api as sm
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from tensorflow.keras.regularizers import l1_l2 # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
import traceback
from sklearn.metrics import mean_squared_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # type: ignore
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor # type: ignore
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
from tensorflow import keras # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import CuDNNLSTM # type: ignore

# Time period for historical data
end_date = datetime(2025, 3, 13)
start_date = end_date - timedelta(days=9131)  

data_dir = "stock_data_LSTM_Future_1"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

portfolio_df = pd.read_excel('holdings.xlsx', sheet_name='Equity', skiprows=22, header=0)

print("DataFrame head:")
print(portfolio_df.head())
print("\nColumns:", portfolio_df.columns.tolist())

tickers = portfolio_df['Symbol'].tolist()
print("\nExtracted tickers:", tickers)

def fetch_historical_data(ticker, start_date, end_date):
    suffixes = [f"{ticker}.NS", f"{ticker}.BO", ticker, f"{ticker.replace('-E', '')}.NS"]
    for suffix in suffixes:
        try:
            data = yf.download(suffix, start=start_date, end=end_date)
            if not data.empty:
                print(f"Successfully downloaded data for {suffix}")
                return data[['Open', 'High', 'Low', 'Close']].dropna()
        except Exception as e:
            print(f"Error with {suffix}: {e}")
    return pd.DataFrame()

# Download historical data for each stock
stock_price_data = {}
for ticker in tickers:
    print(f"\nProcessing {ticker}:")
    stock_data = fetch_historical_data(ticker, start_date, end_date)
    if not stock_data.empty:
        stock_price_data[ticker] = stock_data
        csv_filename = os.path.join(data_dir, f"{ticker}_data.csv")
        stock_data.to_csv(csv_filename)
        print(f"Saved data for {ticker} to {csv_filename}")
    else:
        print(f"Could not retrieve data for {ticker}")

def calculate_rsi(close_series, period=14):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # avg_gain = gain.rolling(period).mean()
    # avg_loss = loss.rolling(period).mean()

    # Use Wilder's EMA smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    # rs = avg_gain / avg_loss
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(close_series, fast=12, slow=26, signal=9):
    #newadd
    if fast >= slow:
        raise ValueError("Fast period must be smaller than slow period")
    #newadd
    exp1 = close_series.ewm(span=fast, adjust=False).mean()
    exp2 = close_series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    # Input validation
    if window <= 1:
        raise ValueError("Window size must be greater than 1")
    if num_std <= 0:
        raise ValueError("Number of standard deviations must be positive")

    # Calculate rolling metrics
    rolling_mean = data.rolling(window=window, min_periods=1).mean()  
    rolling_std = data.rolling(window=window, min_periods=2).std(ddof=0)  

    # Calculate bands
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, rolling_mean, lower_band    

def calculate_roc(close_series, period=12):
    return close_series.pct_change(periods=period) * 100

def calculate_technical_indicators(df, 
                                 rsi_period=14,
                                 macd_fast=12,
                                 macd_slow=26,
                                 macd_signal=9,
                                 bb_period=20,
                                 roc_period=12):
                            
    if 'High' not in df or 'Low' not in df:
        raise ValueError("DataFrame must contain 'High' and 'Low' columns for full technical analysis")

    # Indicators
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
    
    # MACD
    macd, signal = calculate_macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['MACD'] = macd
    df['Signal'] = signal
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['Close'], 
                                                                               window=bb_period)

    # Rate of Change
    df['ROC'] = calculate_roc(df['Close'], period=roc_period)
    
    if df is not None:
        df = df.dropna()
    else:
        print(f"Error: DataFrame for {ticker} is None")
    
    return df

for ticker in stock_price_data:
    df = stock_price_data[ticker]
    
    try:
        df = calculate_technical_indicators(df)

        features = [
            'Close', 'RSI', 'MACD', 'Signal', 
            'BB_upper', 'BB_middle', 'BB_lower',
            'ROC'
        ]

        required_columns = features + ['High', 'Low']
        df = df[required_columns]
        
        csv_filename = os.path.join(data_dir, f"{ticker}_data_with_indicators.csv")
        df.to_csv(csv_filename)
        
        print(f"Saved data for {ticker} to {csv_filename}")
    except ValueError as ve:
        print(f"Error processing {ticker}: {str(ve)}")
        traceback.print_exc()
        continue
    except Exception as e:
        print(f"Unexpected error processing {ticker}: {str(e)}")
        
        continue

def perform_eda(stock_data_dict):
    combined_df = pd.concat(stock_data_dict.values(), keys=stock_data_dict.keys(), names=['Ticker', 'Date'])

    print('NamitaVilasSawant')
    
    summary_stats = combined_df.groupby('Ticker')['Close'].describe()
    print(summary_stats)
    print(combined_df.groupby('Ticker')['Close'].describe())

    reshaped_summary_stats = summary_stats.reset_index()
    summary_stats_file = os.path.join(data_dir, "summary_statistics.csv")
    reshaped_summary_stats.to_csv(summary_stats_file, index=False)
    print(f"Summary statistics saved to {summary_stats_file}")

    for ticker in combined_df.index.get_level_values('Ticker').unique():
        combined_df.loc[ticker, 'Returns'] = combined_df.loc[ticker, 'Close'].pct_change(fill_method=None)

    z_scores = stats.zscore(combined_df['Returns'].dropna())
    outliers = (abs(z_scores) > 3).sum()
    print(f"Number of outliers detected: {outliers}")
        
perform_eda(stock_price_data)


def preprocess_data(df):
    numeric_columns = ['Close', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_middle', 'BB_lower', 'ROC', 'High', 'Low']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # IQR method
    def handle_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = np.where(
                (df[column] < lower_bound) | (df[column] > upper_bound),
                np.nan,  # Replace outliers with NaN
                df[column]
            )

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound) 

        return df

def load_and_preprocess_data(csv_filename):
    df = pd.read_csv(csv_filename, skiprows=2, index_col=0, parse_dates=True)
    
    if len(df.columns) != 10:  # We expect 4 columns excluding the index
        print(f"Unexpected number of columns in {csv_filename}")
        return None
    
    df.columns = ['Close', 'RSI', 'MACD', 'Signal', 
            'BB_upper', 'BB_middle', 'BB_lower',
            'ROC', 'High', 'Low']

    return preprocess_data(df) 

sequence_length = 180

def prepare_sequences(data, sequence_length):  
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length][0]) 
    return np.array(X), np.array(y)

def create_lstm_model(input_file, look_back=126, train_split=0.7):
    # Load and preprocess the data
    df = load_and_preprocess_data(input_file)
    
    if df is None:
        print(f"Error loading data from {input_file}")
        return None, None, None, None

    features = ['Close', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_middle', 'BB_lower', 'ROC']

    rf = RandomForestRegressor()
    rf.fit(df[features], df['Close'])

    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("Feature Importance:\n", importance)

    selected_features = importance.head(5).index.tolist()
    print("Selected features:", selected_features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = prepare_sequences(scaled_data, look_back)
    
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    

    def build_model(hp): ##bayesian
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=256, step=32),
            input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=False  # Add this for single LSTM layer
        ))
        model.add(Dense(1))  
        
        learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']  # validation metrics
        ) 
        
        return model
    
    overwrite=True

    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=20,  # Oracle param 
        num_initial_points=5,  # Not inside Oracle()
        directory='keras_tuner',
        project_name=f'lstm_tuning_{ticker}',
        # overwrite=True,
        # Remove explicit oracle= parameter
        # Add other Bayesian params directly:
        alpha=1e-4,
        beta=2.6,
        executions_per_trial=3
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    tuner.search(
        X_train, y_train,
        epochs=100,  # Set max epochs here
        validation_data=(X_test, y_test),
        batch_size=64,
        callbacks=[early_stopping]
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    
    return best_model, tuner.get_best_hyperparameters()[0], scaler, df, X_train, X_test, y_train, y_test, train_size, look_back

for ticker in tickers:
    input_file = os.path.join(data_dir, f"{ticker}_data_with_indicators.csv")
    if os.path.exists(input_file):
        print(f"\nProcessing LSTM for {ticker}")
        best_model, best_hps, scaler, df, X_train, X_test, y_train, y_test, train_size, look_back = create_lstm_model(input_file)
        
        if best_model is not None:
            train_predict = best_model.predict(X_train)
            test_predict = best_model.predict(X_test)
            
            train_predict_full = np.zeros((len(train_predict), len(features))) 
            train_predict_full[:, 0] = train_predict[:, 0]  

            test_predict_full = np.zeros((len(test_predict), len(features)))
            test_predict_full[:, 0] = test_predict[:, 0]

            train_original_scale = scaler.inverse_transform(train_predict_full)[:, 0]
            test_original_scale = scaler.inverse_transform(test_predict_full)[:, 0]

            y_train_full = np.zeros((len(y_train), len(features)))
            y_train_full[:, 0] = y_train.reshape(-1)
            y_train_inv = scaler.inverse_transform(y_train_full)[:, 0]

            y_test_full = np.zeros((len(y_test), len(features)))
            y_test_full[:, 0] = y_test.reshape(-1)
            y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

            print(f"X_train shape for {ticker}: {X_train.shape}, NaNs: {np.any(np.isnan(X_train))}")
            print(f"y_train shape for {ticker}: {y_train.shape}, NaNs: {np.any(np.isnan(y_train))}")

            def mean_absolute_percentage_error(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            train_mape = mean_absolute_percentage_error(y_train_inv, train_original_scale)
            test_mape = mean_absolute_percentage_error(y_test_inv, test_original_scale)
            print(f'Train MAPE: {train_mape:.2f}%')
            print(f'Test MAPE: {test_mape:.2f}%')

            plot_dir = "stock_predictions_plot_hyperparameter_1"
            if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

            # Plot results
            plt.figure(figsize=(16, 8))
            plt.plot(df.index[look_back:], df['Close'][look_back:], label='Actual')
            plt.plot(df.index[look_back:look_back + len(train_original_scale)], train_original_scale, label='Train Predict')
            plt.plot(df.index[train_size + look_back:], test_original_scale, label='Test Predict')
            plt.legend()
            plt.title(f'Stock Price Prediction - {os.path.basename(input_file)}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.savefig(os.path.join(plot_dir, f'{ticker}_predictions.png'))
            plt.close()
            
            print(f"Best hyperparameters: {best_hps.values}")
    else:
        print(f"No processed data file found for {ticker}")




