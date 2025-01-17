from tsai.all import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    create_sequences,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features,
)
from fastai.callback.tracker import EarlyStoppingCallback

class PatchTSTStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        """
        Initializes the PatchTSTStockModel with default hyperparameters.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "context_length": 64,
            "forecast_horizon": 3,
            "patch_len": 16,
            "stride": 8,
            "n_layers": 3,
            "n_heads": 4,
            "d_model": 64,
            "d_ff": 256,
            "dropout": 0.2,
            "batch_size": 32,
            "num_epochs": 30,
            "learning_rate": 0.0001,
        }

    def prepare_data(self):
        """
        Prepares the dataset using tsai's utilities for forecasting tasks.
        """
        # Load the dataset
        self.data = load_data(self.file_path)
        df = create_lagged_features(self.data)
        df = fill_na_values(df)
        df = extract_date_features(df)

        # Define splits
        splits = get_forecasting_splits(df, 
                                        fcst_history=self.hyperparameters["context_length"],
                                        fcst_horizon=self.hyperparameters["forecast_horizon"],
                                        valid_size=0.1,
                                        test_size=82,
                                        )

        self.splits = splits

        # Fit scaler on training data only
        train_df = df.iloc[self.splits[0]]  # Training data
        self.scaler = TSStandardScaler(columns=df.columns).fit(train_df)

        # Apply scaling to the entire dataset
        df = self.scaler.transform(df)

        self.df = df

        # Sliding window
        x_vars = df.columns[0:]
        y_vars = df.columns[0:]
        X, y = prepare_forecasting_data(df, 
                                        fcst_history=self.hyperparameters["context_length"], 
                                        fcst_horizon=self.hyperparameters["forecast_horizon"], 
                                        x_vars=x_vars, y_vars=y_vars)
        # self.X, self.y = X, y
        # Ensure numeric types
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)


    def build_and_train_model(self):
        """
        Builds and trains the PatchTST model using tsai's TSForecaster.
        """
        # Define architecture configuration
        arch_config = dict(
            n_layers=self.hyperparameters["n_layers"],
            n_heads=self.hyperparameters["n_heads"],
            d_model=self.hyperparameters["d_model"],
            d_ff=self.hyperparameters["d_ff"],
            dropout=self.hyperparameters["dropout"],
            patch_len=self.hyperparameters["patch_len"],
            stride=self.hyperparameters["stride"],
        )

        # TSForecaster
        learn = TSForecaster(self.X, self.y, splits=self.splits, path="models",
                            arch="PatchTST", arch_config=arch_config, 
                            metrics=[mae, mse], 
                            batch_size=self.hyperparameters["batch_size"])

        # Add EarlyStoppingCallback
        early_stopping = EarlyStoppingCallback(monitor='valid_loss', patience=5, min_delta=1e-4)

        # Training
        learn.fit_one_cycle(self.hyperparameters["num_epochs"], 
                            lr_max=self.hyperparameters["learning_rate"],
                            cbs=[early_stopping])
        self.learn = learn

    def predict(self):
        """
        Generates predictions for the test dataset and applies inverse transform if necessary.
        """
        # Get predictions from the model
        preds, *_ = self.learn.get_X_preds(self.X[self.splits[2]])
        preds = preds.squeeze(-1).numpy()  # Ensure predictions are 2D

        # Extract the target variable (predicted Close price)
        preds_df = pd.DataFrame(preds[:, :, 0], columns=self.df.columns)
        # preds_df = pd.DataFrame(preds.mean(axis=2), columns=self.df.columns)

        preds_df = self.scaler.inverse_transform(preds_df)
        preds_df = preds_df["Close"].values.flatten()
        return preds_df

    def save_predictions(self, preds):
        """
        Saves predictions to a CSV file.
        """
        # Save predictions to CSV
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"PatchTST_{self.stock_name}_predictions.csv")

        # Create a DataFrame for predictions and save
        prediction_dates = self.df.index[-len(preds):]
        preds_df = pd.DataFrame({"Date": prediction_dates, "Predicted Close": preds})
        preds_df.to_csv(prediction_path, index=False)
        print(f"Predictions saved to: {prediction_path}")

    def save_model(self):
        """
        Saves the trained model for future use.
        """
        self.learn.export(f"{self.stock_name}_patchTST_model.pt")

    def run(self):
        """
        Runs the entire pipeline: prepare data, train model, and save predictions.
        """
        self.prepare_data()
        self.build_and_train_model()

        predictions = self.predict()
        self.save_predictions(predictions)
        self.save_model()
