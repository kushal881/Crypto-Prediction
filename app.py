from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import json
import io
import base64
from scipy import stats

app = Flask(__name__, static_folder='static')

# Create necessary directories if they don't exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

# Sample cryptocurrency data
# In a production environment, this would come from an API or database
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE']
CRYPTO_NAMES = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin',
    'XRP': 'Ripple',
    'ADA': 'Cardano',
    'SOL': 'Solana',
    'DOT': 'Polkadot',
    'DOGE': 'Dogecoin'
}

# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    """
    Get current market data for cryptocurrencies.
    In a real application, this would fetch data from an API.
    """
    # Simulate market data
    market_data = {
        'btc': {
            'price': '$63,245.78',
            'change': '+2.34%'
        },
        'eth': {
            'price': '$3,487.12',
            'change': '+1.76%'
        },
        'marketCap': {
            'value': '$2.34T',
            'change': '+0.89%'
        },
        'volume': {
            'value': '$142.6B',
            'change': '-1.24%'
        }
    }
    return jsonify(market_data)

@app.route('/api/price-chart', methods=['GET'])
def get_price_chart():
    """
    Generate price chart for a cryptocurrency over a specified timeframe.
    """
    symbol = request.args.get('symbol', 'BTC')
    timeframe = request.args.get('timeframe', '7d')
    
    # Generate or load historical data
    df = generate_historical_data(symbol, timeframe)
    
    # Create the chart
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['price'], color='#6c5ce7', linewidth=2)
    plt.fill_between(df.index, df['price'], color='#6c5ce7', alpha=0.1)
    plt.title(f'{CRYPTO_NAMES.get(symbol, symbol)} Price Chart ({timeframe})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the chart to a buffer and return as base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'chart': chart_data})

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Generate cryptocurrency recommendations based on machine learning.
    """
    data = request.get_json()
    risk_tolerance = data.get('riskTolerance', 'medium')
    time_horizon = data.get('timeHorizon', 'medium')
    
    # Generate recommendations using ML
    recommendations = generate_recommendations(risk_tolerance, time_horizon)
    
    return jsonify({'recommendations': recommendations})

@app.route('/api/prediction', methods=['POST'])
def get_prediction():
    """
    Generate price prediction for a cryptocurrency.
    """
    data = request.get_json()
    symbol = data.get('symbol', 'BTC')
    days_ahead = int(data.get('daysAhead', 7))
    
    # Generate prediction using ML
    prediction_data = generate_prediction(symbol, days_ahead)
    
    return jsonify(prediction_data)

@app.route('/api/analysis/correlation', methods=['GET'])
def get_correlation_analysis():
    """
    Generate correlation heatmap for cryptocurrencies.
    """
    # Generate correlation data
    df = generate_correlation_data()
    
    # Create correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Cryptocurrency Price Correlation', fontsize=14)
    plt.tight_layout()
    
    # Save the chart to a buffer and return as base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'chart': chart_data})

@app.route('/api/analysis/volatility', methods=['GET'])
def get_volatility_analysis():
    """
    Generate volatility analysis for cryptocurrencies.
    """
    # Generate volatility data
    volatility_data = generate_volatility_data()
    
    # Create volatility chart
    plt.figure(figsize=(8, 6))
    plt.bar(volatility_data.keys(), volatility_data.values(), color='#6c5ce7')
    plt.title('30-Day Volatility by Cryptocurrency', fontsize=14)
    plt.ylabel('Volatility (Standard Deviation of Daily Returns)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save the chart to a buffer and return as base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'chart': chart_data})

@app.route('/api/analysis/sentiment', methods=['GET'])
def get_sentiment_analysis():
    """
    Generate sentiment analysis for cryptocurrencies.
    """
    # Generate sentiment data
    sentiment_data = generate_sentiment_data()
    
    # Create sentiment chart
    plt.figure(figsize=(8, 6))
    
    # Extract data
    symbols = list(sentiment_data.keys())
    positive = [data['positive'] for data in sentiment_data.values()]
    neutral = [data['neutral'] for data in sentiment_data.values()]
    negative = [data['negative'] for data in sentiment_data.values()]
    
    # Create stacked bar chart
    width = 0.6
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(symbols, positive, width, label='Positive', color='#00b894')
    ax.bar(symbols, neutral, width, bottom=positive, label='Neutral', color='#fdcb6e')
    ax.bar(symbols, negative, width, bottom=[p+n for p, n in zip(positive, neutral)], label='Negative', color='#ff7675')
    
    ax.set_ylabel('Sentiment Distribution')
    ax.set_title('Market Sentiment Analysis', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the chart to a buffer and return as base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({'chart': chart_data})

@app.route('/api/placeholder/<int:width>/<int:height>', methods=['GET'])
def get_placeholder_image(width, height):
    """
    Generate a placeholder image with specified dimensions.
    """
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.plot([0, 1], [0, 1], color='#6c5ce7', linewidth=2)
    plt.plot([0, 1], [1, 0], color='#6c5ce7', linewidth=2)
    plt.fill_between([0, 0.5, 1], [0, 0.7, 0], color='#6c5ce7', alpha=0.1)
    plt.grid(True, alpha=0.2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plt.close()
    
    return buffer.getvalue(), 200, {'Content-Type': 'image/png'}

# Helper functions for data generation and ML models
def generate_historical_data(symbol, timeframe):
    """
    Generate historical price data for a cryptocurrency.
    In a real application, this would fetch data from an API.
    """
    # Map timeframe to number of days
    days_map = {
        '1d': 1,
        '7d': 7,
        '30d': 30,
        '90d': 90
    }
    days = days_map.get(timeframe, 7)
    
    # Generate synthetic data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create a dataframe with random price data
    np.random.seed(hash(symbol) % 1000)  # Use symbol as seed for consistent randomness
    
    # Start with a base price depending on the symbol
    base_prices = {
        'BTC': 63000,
        'ETH': 3500,
        'BNB': 580,
        'XRP': 0.65,
        'ADA': 0.52,
        'SOL': 140,
        'DOT': 8.2,
        'DOGE': 0.14
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generate price with random walk and some trendy patterns
    trend = np.linspace(0, np.random.uniform(-0.15, 0.15), len(dates))
    volatility = 0.005 + np.random.uniform(-0.002, 0.002)
    
    prices = [base_price]
    for i in range(1, len(dates)):
        rnd = np.random.normal(0, volatility)
        change = rnd + trend[i]
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    df.set_index('date', inplace=True)
    return df

def generate_recommendations(risk_tolerance, time_horizon):
    """
    Generate cryptocurrency recommendations based on machine learning analysis.
    """
    # Generate historical data for all symbols
    all_data = {}
    for symbol in CRYPTO_SYMBOLS