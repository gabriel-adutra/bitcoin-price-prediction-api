import uvicorn
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import predict_utils


# Logger configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Define metadata for API documentation
tags_metadata = [{"name": "Bitcoin-Price-Prediction", "description": "Predicting Bitcoin Price with Machine Learning"}]

# Create FastAPI application instance with metadata information
app = FastAPI(
    title="Bitcoin Price API – Price Prediction with Machine Learning",
        description=(
        "This API provides daily Bitcoin price predictions using Machine Learning models trained with historical data extracted from Yahoo Finance and custom technical indicators. Ideal for financial analysis and data-driven decision making."
    ),
    version="1.0",
    contact={"name": "Gabriel Lopes de Albuquerque Dutra", "email": "gabriel.adutra@ufpe.br"},
    openapi_tags=tags_metadata,
)


############
class Features(BaseModel):  # Define a Pydantic model to validate input data
    Model: str


############
@app.get("/")
def message():
    logger.info("Root endpoint '/' was called.")
    return "This is an API to predict Bitcoin price using Machine Learning. Use the appropriate method"


############
@app.post("/predict-bitcoin-price", tags=["Bitcoin Price Prediction"])
async def predict(features: Features) -> dict:
    """ Endpoint for predicting Bitcoin price for the next day. """
    
    logger.info(f" /predict-bitcoin-price endpoint is requested with model: {features.Model}")
    
    historical_bitcoin_data = predict_utils.download_bitcoin_historical_data()

    input_features_array = predict_utils.extract_latest_record_and_prepare_input_array(historical_bitcoin_data)

    scaler = predict_utils.load_data_scaler_from_file()

    scaled_input_features = predict_utils.apply_scaler_to_input_features(scaler, input_features_array)

    machine_learning_model = predict_utils.load_machine_learning_model_by_name(features.Model)

    predicted_price_array = predict_utils.predict_next_day_price(machine_learning_model, scaled_input_features)

    prediction_response = predict_utils.build_prediction_response(features.Model, historical_bitcoin_data, predicted_price_array)
    
    logger.info(f"/predict-bitcoin-price endpoint is returning: {prediction_response}")
    return prediction_response


#########################
# Start server on any machine IP and port 3000
if __name__ == "__main__":
    logger.info("Starting Uvicorn server on 0.0.0.0:3000")
    uvicorn.run(app, host="0.0.0.0", port=3000)
