import os
import numpy as np
import pandas as pd
import pywt
from machine_learning_models.preprocessing import load_data, create_lagged_features, train_test_split_time_series, fill_na_values, extract_date_features
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import optuna
import pickle   
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class DWT_ARIMA_GSXGB:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        """
        Initializes the DWT-ARIMA-GSXGB model.

        Parameters:
            file_path (str): Path to the CSV file containing historical financial data.
        """
        self.file_path = file_path
        self.stock_name = stock_name

        # Load data and handle technical indicators
        self.data = load_data(self.file_path)

        # Create features and target dataframes
        self.target = self.data["Close"]
        self.features = create_lagged_features(self.data)
        self.features = fill_na_values(self.features)
        self.features = extract_date_features(self.features)
        self.features = self.features.drop(columns=['Close'], errors='ignore')
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(self.features, self.target)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train, test_size=0.2
        )
        
        self.adjust_wavelet_decomposition()

        self.LT = None
        self.NT_test = None
        self.best_params = None

    def adjust_wavelet_decomposition(self, level=1, wavelet='sym4'):
        """
        Adjusts the wavelet decomposition level and basis function.
        """
        # Perform decomposition
        self.Lt, self.Nt = pywt.wavedec(self.y_train, wavelet=wavelet, mode='sym', level=level)

        # Calculate the reduction factor using np.ceil
        reduction_factor = int(np.ceil(len(self.y_train) / len(self.Lt)))

        # Skip alternate rows
        self.X_train = self.X_train.iloc[::reduction_factor]
        min_length = min(len(self.X_train), len(self.Nt))
        self.X_train = self.X_train[:min_length]
        self.Nt = self.Nt[:min_length]

        # aggregated_features = self.X_train.iloc[:, :-1].groupby(np.arange(len(self.X_train)) // reduction_factor).mean()

        self.Lt_val, self.Nt_val = pywt.wavedec(self.y_val, wavelet=wavelet, mode='sym', level=level)

        # Calculate the reduction factor using np.ceil
        reduction_factor = int(np.ceil(len(self.y_val) / len(self.Lt_val)))

        # Skip alternate rows
        self.X_val = self.X_val.iloc[::reduction_factor]
        min_length = min(len(self.X_val), len(self.Nt_val))
        self.X_val = self.X_val[:min_length]
        self.Nt_val = self.Nt_val[:min_length]

        print(f"Wavelet decomposition adjusted to level {level} with wavelet {wavelet}")

    def tune_arima_hyperparameters(self, n_trials=50):
        """
        Tunes ARIMA hyperparameters (p, d, q) using Optuna on the approximation part (Lt_val).

        Parameters:
            n_trials (int): Number of trials for Optuna optimization.

        Returns:
            dict: Best ARIMA hyperparameters determined by Optuna.
        """
        def objective_arima(trial):
            # Suggest hyperparameters for ARIMA
            p = trial.suggest_int("p", 0, 5)
            d = trial.suggest_int("d", 1, 2)
            q = trial.suggest_int("q", 0, 5)

            try:
                # Train ARIMA on Lt (training approximation part)
                model = ARIMA(self.Lt, order=(p, d, q))
                fitted_model = model.fit()

                # Predict on validation approximation part
                Lt_val_pred = fitted_model.forecast(steps=len(self.Lt_val))

                # Calculate validation loss (MSE)
                val_loss = np.mean((self.Lt_val - Lt_val_pred) ** 2)
                return val_loss

            except Exception as e:
                # Return a high loss for failed trials
                print(f"Failed trial with (p, d, q): {(p, d, q)}. Error: {e}")
                return float("inf")

        # Perform Optuna optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_arima, n_trials=n_trials)

        print(f"Best ARIMA hyperparameters found by Optuna: {study.best_params}")
        return study.best_params

    def train_arima(self, order=(1, 1, 0)):
        """
        Trains the ARIMA model on the approximation part (Lt).

        Parameters:
            order (tuple): ARIMA model parameters (p, d, q).
        """
        model = ARIMA(self.Lt, order=order)
        fitted_model = model.fit()
        self.LT = fitted_model.forecast(steps=len(self.X_test))

    def tune_hyperparameters(self, n_trials=50):
        """
        Tunes XGBoost hyperparameters using Optuna on validation data.

        Parameters:
            n_trials (int): Number of trials for Optuna optimization.

        Returns:
            dict: Best hyperparameters determined by Optuna.
        """
        def objective_xgb(trial):
            """
            Objective function for Optuna to tune XGBoost hyperparameters.

            Parameters:
                trial (optuna.trial.Trial): An Optuna trial object.

            Returns:
                float: Validation loss (MSE) for the given trial's hyperparameters.
            """
            # Suggest a broader range of hyperparameters
            params = {
                "objective": "reg:squarederror",
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "early_stopping_rounds": 15,
            }

            # Align X_train and Nt
            min_length_train = min(len(self.X_train), len(self.Nt))
            X_train = self.X_train[:min_length_train]
            Nt_train = self.Nt[:min_length_train]

            # Align X_val and Nt_val
            min_length_val = min(len(self.X_val), len(self.Nt_val))
            X_val = self.X_val[:min_length_val]
            Nt_val = self.Nt_val[:min_length_val]

            # Train XGBoost with early stopping
            xgb = XGBRegressor(
                **params
            )
            xgb.fit(
                X_train, Nt_train,
                eval_set=[(X_val, Nt_val)],
                verbose=False,
            )

            # Predict and calculate validation loss
            preds = xgb.predict(X_val)
            loss = np.sqrt(mean_squared_error(Nt_val, preds))
            return loss

        # Perform Optuna optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_xgb, n_trials=n_trials)

        print(f"Best hyperparameters for XGBoost: {study.best_params}")
        return study.best_params

    def train_gsxgb(self, best_params):
        """
        Trains the XGBoost model on the training data and detail coefficients (Nt).
        Predicts the detail coefficients for the test set (NT_test).

        Parameters:
            best_params (dict): Best hyperparameters determined by Optuna.
        """
        # Train XGBoost model
        xgb = XGBRegressor(**best_params)
        xgb.fit(self.X_train, self.Nt)

        # Predict detail coefficients for the test set
        self.NT_test = xgb.predict(self.X_test)
    
    def reconstruct(self):
        """
        Reconstructs the final predictions by combining LT and NT.

        Returns:
            np.array: Reconstructed predictions (YT).
        """
        return pywt.waverec([self.LT*2, self.NT_test], wavelet='db4')

    def save_model(self, file_name="dwt_arima_gsxgb_model.pkl"):
        """
        Saves the trained model's parameters and data.

        Parameters:
            file_name (str): File name to save the model.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, file_name)

        model_data = {
            "LT": self.LT,
            "NT_test": self.NT_test,
            "best_params": self.best_params,
            "data": self.data,
        }
        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_path}")

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.

        Parameters:
            predictions (np.array): Predicted values.
            file_name (str): File name to save the predictions.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"ARIMA-XGB_{self.stock_name}_predictions.csv")

        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions,
        })
        prediction_df.to_csv(prediction_path, index=False)
        print(f"Predictions saved to {prediction_path}")

    def run(self, arima_order=(1, 1, 0), n_trials=50, n_trials_arima=50):
        """
        Executes the full pipeline: trains ARIMA and GSXGB, reconstructs predictions, and saves the model and predictions.

        Parameters:
            arima_order (tuple): ARIMA model parameters (p, d, q).
            n_trials (int): Number of trials for Optuna optimization.

        Returns:
            np.array: Reconstructed predictions (YT).
        """
        print("Tuning ARIMA hyperparameters with Optuna...")
        # self.best_arima_params = self.tune_arima_hyperparameters(n_trials=n_trials_arima)
        self.best_arima_params = {"p": 1, "d": 2, "q": 4}

        print("Training ARIMA model...")
        self.train_arima(order=(self.best_arima_params["p"], self.best_arima_params["d"], self.best_arima_params["q"]))

        print("Tuning hyperparameters with Optuna...")
        self.best_params = self.tune_hyperparameters(n_trials=n_trials)
        # self.best_params = {'max_depth': 1, 'n_estimators': 143, 'min_child_weight': 6, 'learning_rate': 0.024535275534155604}
        # self.best_params = {'n_estimators': 567, 'learning_rate': 0.09400230009881862, 'max_depth': 7,
        #                     'subsample': 0.6003806098649144, 'colsample_bytree': 0.6885322372677508, 'gamma': 0.00890935048519874,
        #                     'reg_alpha': 0.23731206646855155, 'reg_lambda': 0.6262788315842578}

        print("Training GSXGB model...")
        self.train_gsxgb(self.best_params)

        print("Reconstructing final predictions...")
        predictions = self.reconstruct()

        print("Saving model and predictions...")
        self.save_model()
        self.save_predictions(predictions)

        return predictions
