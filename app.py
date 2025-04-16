import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dictionary to store trained models
trained_models = {}
scalers = {}

# Supported cryptocurrencies (matching the JS frontend)
SUPPORTED_CRYPTOS = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT', 'DOGE', 'AVAX']

def fetch_crypto_data(symbol, period='1y'):
    """
    Fetch historical cryptocurrency data from Yahoo Finance
    """
    ticker = f"{symbol}-USD"
    data = yf.download(ticker, period=period)
    return data

def preprocess_data(data, sequence_length=60):
    """
    Preprocess the data for LSTM model
    """
    # Select only the Close price
    df = data[['Close']].copy()
    
    # Feature engineering
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=7).std()
    
    # Drop NA values
    df = df.dropna()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 0])  # Only predict the Close price
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler, df

def build_lstm_model(input_shape):
    """
    Build an LSTM model for time series prediction
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(symbol):
    """
    Train an LSTM model for a specific cryptocurrency
    """
    print(f"Training model for {symbol}...")
    
    # Fetch data
    data = fetch_crypto_data(symbol)
    
    # Preprocess data
    X, y, scaler, df = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model for {symbol} trained. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Store model and scaler
    trained_models[symbol] = model
    scalers[symbol] = scaler
    
    # Return model metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def generate_prediction(symbol, timeframe_days):
    """
    Generate price predictions for the specified cryptocurrency and timeframe
    """
    # Check if we have a trained model, if not, train one
    if symbol not in trained_models:
        train_model(symbol)
    
    # Get the model and scaler
    model = trained_models[symbol]
    scaler = scalers[symbol]
    
    # Fetch recent data
    data = fetch_crypto_data(symbol, period='60d')
    
    # Preprocess data (without splitting)
    X, _, _, df = preprocess_data(data)
    
    # Use the last sequence for prediction
    last_sequence = X[-1:]
    
    # Current price
    current_price = df['Close'].iloc[-1]
    
    # Make predictions for future days
    predictions = []
    prediction_upper = []
    prediction_lower = []
    
    # Start with the last known sequence
    curr_seq = last_sequence[0]
    
    # Generate predictions for each day in the timeframe
    for i in range(timeframe_days):
        # Predict next day
        pred = model.predict(np.array([curr_seq]), verbose=0)[0][0]
        
        # Calculate upper and lower bounds (uncertainty increases with time)
        uncertainty = 0.01 + (i / timeframe_days * 0.04)
        upper_bound = pred * (1 + uncertainty)
        lower_bound = pred * (1 - uncertainty)
        
        # Store predictions with bounds
        predictions.append(pred)
        prediction_upper.append(upper_bound)
        prediction_lower.append(lower_bound)
        
        # Update sequence for next prediction
        # Create a new row with the same features as the last one, but update the Close price
        new_row = np.copy(curr_seq[-1])
        new_row[0] = pred  # Update Close price
        
        # Shift sequence and add new prediction
        curr_seq = np.vstack([curr_seq[1:], new_row])
    
    # Inverse transform to get actual prices
    # We need to create a proper shaped array that matches the training data
    pred_array = np.zeros((len(predictions), scaler.scale_.shape[0]))
    upper_array = np.zeros((len(predictions), scaler.scale_.shape[0]))
    lower_array = np.zeros((len(predictions), scaler.scale_.shape[0]))
    
    # Fill in the Close price predictions
    pred_array[:, 0] = predictions
    upper_array[:, 0] = prediction_upper
    lower_array[:, 0] = prediction_lower
    
    # Fill in other features with the last known values (simplified)
    for i in range(1, scaler.scale_.shape[0]):
        pred_array[:, i] = curr_seq[-1, i]
        upper_array[:, i] = curr_seq[-1, i]
        lower_array[:, i] = curr_seq[-1, i]
    
    # Inverse transform
    pred_prices = scaler.inverse_transform(pred_array)[:, 0]
    upper_prices = scaler.inverse_transform(upper_array)[:, 0]
    lower_prices = scaler.inverse_transform(lower_array)[:, 0]
    
    # Calculate price changes
    target_price = pred_prices[-1]
    high_price = upper_prices[-1]
    low_price = lower_prices[-1]
    
    target_change = ((target_price / current_price) - 1) * 100
    high_change = ((high_price / current_price) - 1) * 100
    low_change = ((low_price / current_price) - 1) * 100
    
    # Generate dates for the prediction
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%b %d') for i in range(1, timeframe_days + 1)]
    
    # Get historical data for chart
    historical_data = fetch_crypto_data(symbol, period='30d')
    historical_dates = [d.strftime('%b %d') for d in historical_data.index.date]
    historical_prices = historical_data['Close'].tolist()
    
    return {
        'current_price': current_price,
        'prediction': {
            'target': {
                'price': target_price,
                'change': target_change
            },
            'high': {
                'price': high_price,
                'change': high_change
            },
            'low': {
                'price': low_price,
                'change': low_change
            }
        },
        'chart_data': {
            'dates': dates,
            'predicted_prices': pred_prices.tolist(),
            'upper_bound': upper_prices.tolist(),
            'lower_bound': lower_prices.tolist(),
            'historical_dates': historical_dates,
            'historical_prices': historical_prices
        }
    }

def generate_market_data(symbol):
    """
    Generate additional market data for the specified cryptocurrency
    """
    data = fetch_crypto_data(symbol, period='30d')
    
    # Calculate market cap and volume (approximate values for demonstration)
    market_caps = {
        'BTC': 918.4e9,
        'ETH': 372.5e9,
        'BNB': 71.2e9,
        'SOL': 42.8e9,
        'ADA': 18.3e9,
        'DOT': 14.7e9,
        'DOGE': 16.8e9,
        'AVAX': 11.2e9
    }
    
    # Calculate 24h change
    if len(data) >= 2:
        price_24h_ago = data['Close'].iloc[-2]
        current_price = data['Close'].iloc[-1]
        change_24h = ((current_price / price_24h_ago) - 1) * 100
    else:
        change_24h = 0
    
    # Calculate 7d change
    if len(data) >= 8:
        price_7d_ago = data['Close'].iloc[-8]
        current_price = data['Close'].iloc[-1]
        change_7d = ((current_price / price_7d_ago) - 1) * 100
    else:
        change_7d = 0
    
    # Calculate approximate volume
    volume = data['Volume'].mean() / 1e9  # Convert to billions
    
    return {
        'current_price': data['Close'].iloc[-1],
        'change_24h': change_24h,
        'change_7d': change_7d,
        'market_cap': market_caps.get(symbol, 10e9) / 1e9,  # In billions
        'volume': volume
    }

# API endpoints
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for cryptocurrency price prediction
    """
    data = request.json
    symbol = data.get('symbol', 'BTC')
    timeframe = int(data.get('timeframe', 30))
    
    if symbol not in SUPPORTED_CRYPTOS:
        return jsonify({
            'success': False,
            'error': f'Unsupported cryptocurrency: {symbol}'
        })
    
    try:
        # Generate prediction
        prediction_data = generate_prediction(symbol, timeframe)
        market_data = generate_market_data(symbol)
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': prediction_data,
                'market': market_data
            }
        })
    except Exception as e:
        print(f"Error generating prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/market-data', methods=['GET'])
def market_data():
    """
    API endpoint for current market data
    """
    symbol = request.args.get('symbol', 'BTC')
    
    if symbol not in SUPPORTED_CRYPTOS:
        return jsonify({
            'success': False,
            'error': f'Unsupported cryptocurrency: {symbol}'
        })
    
    try:
        data = generate_market_data(symbol)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """
    API endpoint to check which models are trained
    """
    return jsonify({
        'success': True,
        'data': {
            'trained_models': list(trained_models.keys()),
            'supported_cryptos': SUPPORTED_CRYPTOS
        }
    })

# Generate visualization of model training and prediction results
@app.route('/api/visualization', methods=['GET'])
def visualization():
    """
    API endpoint for generating model performance visualizations
    """
    symbol = request.args.get('symbol', 'BTC')
    
    if symbol not in trained_models:
        train_model(symbol)
    
    # Fetch data for visualization
    data = fetch_crypto_data(symbol, period='60d')
    X, y, scaler, df = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    model = trained_models[symbol]
    y_pred = model.predict(X_test)
    
    # Inverse transform
    test_indices = np.arange(len(X_train), len(X_train) + len(X_test))
    
    # Create arrays with shape matching original data
    y_test_array = np.zeros((len(y_test), scaler.scale_.shape[0]))
    y_test_array[:, 0] = y_test
    
    y_pred_array = np.zeros((len(y_pred), scaler.scale_.shape[0]))
    y_pred_array[:, 0] = y_pred.flatten()
    
    # Fill in other features with zeros
    for i in range(1, scaler.scale_.shape[0]):
        last_val = df.iloc[-1].values[i] / df.iloc[-1].values[0]  # normalize by Close price
        y_test_array[:, i] = y_test * last_val
        y_pred_array[:, i] = y_pred.flatten() * last_val
    
    # Inverse transform
    y_test_real = scaler.inverse_transform(y_test_array)[:, 0]
    y_pred_real = scaler.inverse_transform(y_pred_array)[:, 0]
    
    # Create visualization directory if it doesn't exist
    os.makedirs('static/visualizations', exist_ok=True)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label='Actual Prices')
    plt.plot(y_pred_real, label='Predicted Prices')
    plt.title(f'{symbol} Price Prediction Model Performance')
    plt.xlabel('Test Sample')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    fig_path = f'static/visualizations/{symbol}_prediction.png'
    plt.savefig(fig_path)
    plt.close()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)
    
    return jsonify({
        'success': True,
        'data': {
            'visualization_path': fig_path,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }
    })

@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """
    API endpoint for generating feature importance visualizations
    """
    symbol = request.args.get('symbol', 'BTC')
    
    # Fetch data
    data = fetch_crypto_data(symbol, period='1y')
    
    # Create correlation matrix
    corr_matrix = data.corr()
    
    # Create visualization directory if it doesn't exist
    os.makedirs('static/visualizations', exist_ok=True)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'{symbol} Feature Correlation Matrix')
    
    # Save the figure
    fig_path = f'static/visualizations/{symbol}_correlation.png'
    plt.savefig(fig_path)
    plt.close()
    
    return jsonify({
        'success': True,
        'data': {
            'visualization_path': fig_path,
            'correlation_matrix': corr_matrix.to_dict()
        }
    })

@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    """
    API endpoint for training a model
    """
    data = request.json
    symbol = data.get('symbol', 'BTC')
    
    if symbol not in SUPPORTED_CRYPTOS:
        return jsonify({
            'success': False,
            'error': f'Unsupported cryptocurrency: {symbol}'
        })
    
    try:
        metrics = train_model(symbol)
        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'metrics': metrics
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Function to pre-train models for all supported cryptocurrencies
def pretrain_models():
    """
    Pre-train models for all supported cryptocurrencies
    """
    for symbol in SUPPORTED_CRYPTOS:
        try:
            train_model(symbol)
        except Exception as e:
            print(f"Error training model for {symbol}: {str(e)}")

if __name__ == '__main__':
    # Pre-train models in the background
    import threading
    pretraining_thread = threading.Thread(target=pretrain_models)
    pretraining_thread.start()
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)