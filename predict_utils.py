import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from technical_indicators import calculate_technical_indicators
import yfinance as yf
from joblib import load
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


###########
def download_bitcoin_historical_data(period_duration: str = "500d") -> pd.DataFrame:
    """ Downloads historical Bitcoin price data, calculates financial indicators and sorts the data. """

    logger.info(f"download_bitcoin_historical_data() is called with period_duration: {period_duration}")
    bitcoin_ticker = yf.Ticker("BTC-USD")
    historical_price_data = bitcoin_ticker.history(period=period_duration, actions=False)
    historical_price_data_without_timezone = historical_price_data.tz_localize(None)
    historical_price_data_with_indicators = calculate_technical_indicators(historical_price_data_without_timezone)
    sorted_historical_data_desc = historical_price_data_with_indicators.sort_index(ascending=False)
    return sorted_historical_data_desc


###########
def extract_latest_record_and_prepare_input_array(historical_data_sorted_desc: pd.DataFrame) -> np.ndarray:
    """ Extracts the most recent record from historical data and prepares a features array for the model. """

    latest_day_record = historical_data_sorted_desc.iloc[0, :]
    latest_day_record_filled_na = latest_day_record.fillna(0)
    input_features_array_reshaped = latest_day_record_filled_na.array.reshape(1, -1)
    return input_features_array_reshaped


###########
def load_data_scaler_from_file(scaler_file_path: str = "btc_feature_scaler.bin") -> StandardScaler:
    """ Loads the scaler object used for input data normalization. """

    logger.info(f"load_data_scaler_from_file() is called with scaler_file_path: {scaler_file_path}")
    scaler_object = load(scaler_file_path)
    return scaler_object


###########
def apply_scaler_to_input_features(scaler_object: StandardScaler, input_features_array: np.ndarray) -> np.ndarray:
    """ Applies the scaler to input data for normalization. """
    
    scaled_input_features_array = scaler_object.transform(input_features_array)
    return scaled_input_features_array


###########
def load_machine_learning_model_by_name(model_name: str) -> BaseEstimator:
    """ Loads the machine learning model based on the specified name. """
    
    logger.info(f"load_machine_learning_model_by_name() is called with model_name: {model_name}")
    if model_name == "Random Forest":
        machine_learning_model_file = "bitcoin_price_model_random_forest.joblib"
        loaded_model = load(machine_learning_model_file)
        return loaded_model
    logger.error(f"load_machine_learning_model_by_name() received unknown model_name: {model_name}")
    raise ValueError(f"Unknown model name provided: {model_name}")


###########
def predict_next_day_price(model_object: BaseEstimator, scaled_input_features: np.ndarray) -> np.ndarray:
    """ Performs Bitcoin price prediction for the next day using the provided model. """

    predicted_price_array = model_object.predict(scaled_input_features)
    logger.info(f"predict_next_day_price() is returning {predicted_price_array}")
    return predicted_price_array


###########
def build_prediction_response(model_name_used: str, historical_data_sorted_desc: pd.DataFrame, predicted_price_array: np.ndarray) -> dict:
    """ Builds the API response dictionary containing the model used, last known price and prediction. """
    
    last_known_bitcoin_price = historical_data_sorted_desc.iloc[0, 3]
    response_dict = {
        "ModelUsed": model_name_used,
        "LastKnownBitcoinPrice": round(last_known_bitcoin_price, 2),
        "PredictedBitcoinPriceForNextDay": round(predicted_price_array.tolist()[0], 2),
    }
    return response_dict
