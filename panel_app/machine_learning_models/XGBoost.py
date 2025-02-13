import pandas as pd
import numpy as np
from machine_learning_models.preprocessing import load_data, create_lagged_features, train_test_split_time_series, fill_na_values, extract_date_features
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import os
import pickle
import optuna
import shap
from sklearn.inspection import permutation_importance

def rmse(y_true, y_pred):
    """
    Custom RMSE scoring function.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


class XGBoostStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        """
        Initializes the XGBoost model for stock prediction.

        Parameters:
            file_path (str): Path to the CSV file containing stock data.
            stock_name (str): Name of the stock for saving models and predictions.
            lags (int): Number of lag features to create.
            rolling_window (int): Size of the rolling window for feature engineering.
            test_size (float): Fraction of the data to use for testing.
            hyperparameters (dict): Hyperparameters for the XGBoost model.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.5,
            "colsample_bytree": 0.82,
            "gamma": 0.08,
            "reg_alpha": 0.5,
            "reg_lambda": 1,
            "random_state": 42,
        }

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

    def perform_time_series_cv(self):
        """
        Performs time series cross-validation and calculates RMSE scores.

        Returns:
            list: RMSE scores for each fold.
        """
        print("Performing time series cross-validation...")
        
        # Define time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define custom RMSE scorer
        rmse_scorer = make_scorer(rmse, greater_is_better=False)

        # Perform cross-validation
        xgb_regressor = xgb.XGBRegressor(**self.hyperparameters)

        # Convert training data to NumPy arrays for compatibility
        X_train_np = np.asarray(self.X_train)
        y_train_np = np.asarray(self.y_train)

        scores = cross_val_score(
            xgb_regressor,
            X_train_np,
            y_train_np,
            cv=tscv,
            scoring=rmse_scorer,
            n_jobs=-1  # Parallel processing
        )

        # Convert negative scores to positive RMSE values
        rmse_scores = -scores
        print(f"Time series cross-validation RMSE scores: {rmse_scores}")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

        return rmse_scores

    def select_relevant_features(self, num_features, method):
        """
        Selects the top `num_features` most relevant features using advanced techniques.

        Parameters:
            num_features (int): Number of top features to select.
            method (str): Feature selection method, either "shap" or "permutation".

        Returns:
            None: Updates the `self.features` and re-splits the data.
        """
        preliminary_model = xgb.XGBRegressor(**self.hyperparameters)
        preliminary_model.fit(self.X_train, self.y_train)

        if method == "shap":
            # Compute SHAP values for feature importance
            explainer = shap.TreeExplainer(preliminary_model)
            shap_values = explainer.shap_values(self.X_train)
            importance = np.abs(shap_values).mean(axis=0)
        elif method == "permutation":
            # Use permutation importance for feature ranking
            perm_importance = permutation_importance(preliminary_model, self.X_train, self.y_train, n_repeats=10, random_state=42)
            importance = perm_importance.importances_mean
        else:
            raise ValueError("Invalid method. Choose 'shap' or 'permutation'.")

        # Select the top `num_features` based on importance
        important_indices = np.argsort(importance)[-num_features:]
        selected_features = self.X_train.columns[important_indices]

        # Reduce features to top important ones
        self.features = self.features[selected_features]
        print(f"Top {num_features} selected features ({method} method): {selected_features.tolist()}")

        # Re-split the data with reduced features
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            self.features, self.target
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train
        )

    def optimize_hyperparameters(self, n_trials=50):
        """
        Uses Optuna to find the best hyperparameters for the XGBoost model.

        Parameters:
            n_trials (int): Number of trials for the optimization process.
        
        Returns:
            dict: The best hyperparameters found.
        """
        def objective(trial):
            # Define the hyperparameter search space
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

            # Train and validate the model
            model = xgb.XGBRegressor(**params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )
            predictions = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, predictions))
            return rmse

        # Create an Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Print the best hyperparameters and set them
        print("Best hyperparameters:", study.best_params)
        self.hyperparameters = study.best_params
        return study.best_params

    def train(self):
        """
        Trains the XGBoost model.

        Parameters:
            optimize (bool): Whether to run hyperparameter optimization before training.
            n_trials (int): Number of trials for hyperparameter optimization if `optimize` is True.
        """
        self.model = xgb.XGBRegressor(**self.hyperparameters)
        print("Training the model on full training data...")
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        """
        Evaluates the model's performance.

        Returns:
            dict: Evaluation metrics (MAE, RMSE, R2).
        """
        predictions = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def save_model(self):
        """
        Saves the trained model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_xgboost_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def save_predictions(self):
        """
        Saves the predictions as a CSV file.
        """
        predictions = self.model.predict(self.X_test)
        forecast_df = pd.DataFrame({
            "Date": self.X_test.index,
            "Predicted Close": predictions
        })
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"XGBoost_{self.stock_name}_predictions.csv")
        forecast_df.to_csv(prediction_path, index=False)

    def plot_shap_feature_importance(self):
        """
        Plots SHAP feature importance for the trained model.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not hasattr(self, 'model'):
            raise ValueError("The model must be trained before plotting SHAP feature importance.")
        
        # Explain the model's predictions using SHAP
        print("Generating SHAP feature importance plot...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)

        # Summary plot of SHAP values
        shap.summary_plot(shap_values, self.X_train, plot_type="bar")

    def save_hyperparameters(self):
        """
        Saves the hyperparameters as a CSV file.
        """
        hyperparam_dir = os.path.join("Output_Data", "Hyperparameters", "XGBoost")
        os.makedirs(hyperparam_dir, exist_ok=True)
        hyperparam_path = os.path.join(hyperparam_dir, f"{self.stock_name}_hyperparameter.csv")

        hyperparam_df = pd.DataFrame.from_dict(self.hyperparameters, orient="index", columns=["Value"])
        hyperparam_df.to_csv(hyperparam_path)
        print(f"Hyperparameters saved to {hyperparam_path}")

    def run(self):
        """
        Runs the full pipeline: trains the model, evaluates it, saves the model and predictions.
        """
        # Select relevant features before training
        print("Selecting relevant features...")
        self.select_relevant_features(num_features=30, method="shap")

        print("Optimizing hyperparameters...")
        self.optimize_hyperparameters(n_trials=50)
        print("Optimal hyperparameters found:", self.hyperparameters)

        print("Performing Time series CV...")
        self.perform_time_series_cv()
        print("Completed Time series CV...")

        print(f"Training XGBoost model for {self.stock_name}...")
        self.train()
        metrics = self.evaluate()
        print(f"Evaluation Metrics for {self.stock_name}: {metrics}")
        self.save_model()
        self.save_predictions()
        # self.plot_shap_feature_importance()
        self.save_hyperparameters()
        return metrics
