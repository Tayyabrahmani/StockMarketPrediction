import os
import pickle
import numpy as np
import pandas as pd
import optuna
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data_svr,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import evaluate_predictions, plot_shap_feature_importance


class SVRStockModel:
    def __init__(self, file_path, stock_name):
        self.file_path = file_path
        self.stock_name = stock_name
        self.model = None
        self.scaler = None

        # Load and preprocess data
        self.data = load_data(self.file_path)

        # Create features and target dataframes
        self.target = self.data["Close"]
        self.features = create_lagged_features(self.data)
        self.features = fill_na_values(self.features)
        self.features = extract_date_features(self.features)
        self.features = self.features.drop(columns=['Close'], errors='ignore')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train
        )

        self.X_train, self.X_test, self.y_train, self.y_test, self.X_val, self.y_val, self.feature_scaler, self.target_scaler = preprocess_data_svr(self.X_train, self.X_test, self.y_train, self.y_test, self.X_val, self.y_val)

    def train(self):
        """
        Trains the SVR model.
        """
        X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], -1)
        self.model.fit(X_train_reshaped, self.y_train)

    def predict(self):
        """
        Generates predictions for the test data and maps them back to the original stock price range.

        Returns:
            np.array: Predicted values for the test data in the original stock price range.
        """
        # Reshape X_test for model input
        X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], -1)

        # Generate predictions
        predictions = self.model.predict(X_test_reshaped)
        
        # Inverse transform predictions to the original scale
        predictions_original_scale = self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
        
        return predictions_original_scale.flatten()

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
            "Predicted Close": predictions,
        })
        prediction_df.to_csv(prediction_path, index=False)

    def optimize_hyperparameters(self):
        """
        Optimizes SVR hyperparameters using Optuna.

        Returns:
            dict: Best hyperparameters found by Optuna.
        """
        def objective(trial):
            # Define the search space for hyperparameters
            C = trial.suggest_float("C", 1.0, 1000.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            epsilon = trial.suggest_float("epsilon", 0.001, 1.0, log=True)

            # Reshape X_train for SVR
            X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], -1)

            # Train SVR model with the suggested hyperparameters
            model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train_reshaped, self.y_train)

            # Reshape X_val and make predictions
            X_val_reshaped = self.X_val.reshape(self.X_val.shape[0], -1)
            predictions = model.predict(X_val_reshaped)

            # Calculate RMSE for the predictions
            rmse = np.sqrt(mean_squared_error(self.y_val, predictions))
            return rmse

        # Use Optuna to optimize the objective function
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        # Get the best hyperparameters
        best_params = study.best_params
        print(f"Best Hyperparameters: {best_params}")
        return best_params

    def run(self):
        """
        Runs the full pipeline: trains, evaluates, and saves the model and predictions.
        """
        print(f"Optimizing hyperparameters for {self.stock_name}...")
        best_params = self.optimize_hyperparameters()

        # Set the model with the best hyperparameters
        self.model = SVR(
            kernel="rbf",
            C=best_params["C"],
            gamma=best_params["gamma"],
            epsilon=best_params["epsilon"],
        )

        print(f"Training SVR model for {self.stock_name}...")
        self.train()
        metrics = self.evaluate()
        predictions = self.predict()
        print(f"Evaluation Metrics: {metrics}")
        self.save_model()
        self.save_predictions(predictions)

        plot_shap_feature_importance(
            model=self.model, 
            X_train=self.X_train.reshape(self.X_train.shape[0], -1),
            feature_names=self.features.columns,
            stock_name=self.stock_name
        )

        return metrics
