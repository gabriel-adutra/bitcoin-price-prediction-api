# Bitcoin Price Prediction API

A Machine Learning model integrated via REST API to predict Bitcoin price using technical indicators and historical data.

## Features

- **Real-time Bitcoin data**: Downloads historical price data from Yahoo Finance
- **Technical Indicators**: Calculates 30+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Machine Learning**: Uses Random Forest model for price prediction
- **REST API**: FastAPI-based endpoint for easy integration
- **Data Normalization**: Automatic feature scaling for optimal model performance

## Prerequisites

- Python 3.12
- Conda (for environment management)

## Installation

### 1. Create and activate virtual environment

```bash
conda create --name apiBitcoinPrice python=3.12
conda activate apiBitcoinPrice
```

### 2. Install dependencies

```bash
conda install pip
pip install -r requirements.txt
```

## Usage

### 1. Start the API server

```bash
python app.py
```

The API will be available at `http://localhost:3000`

### 2. Make predictions

You can use the provided shell script:

```bash
./predict_bitcoin_price.sh
```

Or make direct API calls:

```bash
curl -X POST "http://localhost:3000/predict-bitcoin-price" \
     -H "Content-Type: application/json" \
     -d '{"Model": "Random Forest"}'
```

## API Endpoints

### POST /predict-bitcoin-price

Predicts Bitcoin price for the next day.

**Request Body:**
```json
{
  "Model": "Random Forest"
}
```

**Response:**
```json
{
  "ModelUsed": "Random Forest",
  "LastKnownBitcoinPrice": 45000.50,
  "PredictedBitcoinPriceForNextDay": 45230.75
}
```

## Project Structure

```
├── app.py                          # FastAPI application
├── predict_utils.py                # Prediction utilities
├── technical_indicators.py         # Technical analysis indicators
├── requirements.txt                # Python dependencies
├── predict_bitcoin_price.sh        # Client script
├── bitcoin_price_model_random_forest.joblib  # Trained model
└── btc_feature_scaler.bin          # Feature scaler
```

## Technical Indicators

The system calculates various technical indicators including:
- Williams %R
- Rate of Change (ROC)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Ichimoku Cloud
- Exponential Moving Averages (EMA)
- Average Directional Index (ADX)
- And many more...

## Cleanup

To deactivate the virtual environment and remove it (optional):

```bash
conda deactivate
conda remove --name apiBitcoinPrice --all
```

## License

This project is part of the Software Engineering for Machine Learning course. 