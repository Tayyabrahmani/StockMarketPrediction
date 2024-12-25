from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle
import os
import pandas as pd
import numpy as np
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    train_test_split_time_series,
)
from machine_learning_models.evaluation import evaluate_predictions


class SVRStockModel:
    def __init__(self, file_path, stock_name):
        self.file_path = file_path
        self.stock_name = stock_name
        self.model = None
        self.scaler = None

        # Load and preprocess data
        self.data = load_data(self.file_path)
        self.data = create_lagged_features(self.data, target_col="Close")
        self.features, self.target, self.scaler = preprocess_data(self.data, target_col="Close")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )

    def train(self):
        """
        Trains the SVR model.
        """
        self.model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.5, gamma=1e-7))
        self.model.fit(self.X_train.reshape(self.X_train.shape[0], -1), self.y_train)

    def predict(self):
        """
        Generates predictions for the test data and maps them back to the original stock price range.

        Returns:
            np.array: Predicted values for the test data in the original stock price range.
        """
        predictions = self.model.predict(self.X_test.reshape(self.X_test.shape[0], -1))
        
        # Inverse transform to the original scale
        predictions_reshaped = predictions.reshape(-1, 1)
        dummy_features = np.zeros((len(predictions), self.X_test.shape[2]))
        original_scale = self.scaler.inverse_transform(
            np.hstack([dummy_features, predictions_reshaped])
        )
        return original_scale[:, -1]  # Extract the target column

    def evaluate(self):
        """
        Evaluates the SVR model.

        Returns:
            dict: Evaluation metrics.
        """
        predictions = self.predict()
        metrics = evaluate_predictions(self.y_test, predictions)
        return metrics

    def save_model(self):
        """
        Saves the trained SVR model.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_svr_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.

        Parameters:
            predictions (np.array): Predicted values for the test data.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"SVR_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions.flatten(),
        })
        prediction_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: trains, evaluates, and saves the model and predictions.
        """
        print(f"Training SVR model for {self.stock_name}...")
        self.train()
        metrics = self.evaluate()
        predictions = self.predict()
        print(f"Evaluation Metrics: {metrics}")
        self.save_model()
        self.save_predictions(predictions)
        return metrics
