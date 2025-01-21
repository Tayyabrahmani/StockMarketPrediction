import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import pickle
import pandas as pd
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    create_sequences,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import predict_and_inverse_transform
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback

class RNN(Model):
    def __init__(self, input_dim, hidden_dim=50, num_layers=3, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn_layers = [
            layers.SimpleRNN(
                hidden_dim, return_sequences=(i < num_layers - 1), dropout=dropout
            )
            for i in range(num_layers)
        ]
        self.fc = layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        return self.fc(x)

class RNNStockModel:
    def __init__(self, file_path, stock_name):
        """
        Initializes the RNNStockModel.

        Parameters:
            file_path (str): Path to the data file.
            stock_name (str): Name of the stock.
            hyperparameters (dict): Hyperparameters for training the RNN model.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.model = None
        self.sequence_length = 30

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
            self.X_train, self.y_train, test_size=0.1
        )

        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, self.feature_scaler, self.target_scaler = preprocess_data(self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, add_feature_dim=False)

        # Add the last 29 rows (sequence length) from the train data to create sequences
        self.X_test = np.vstack([self.X_train[-self.sequence_length:], self.X_test])
        self.y_test = np.concatenate([self.y_train[-self.sequence_length:], self.y_test])

        # Concatenate features and targets for sequence creation (train)
        data_train = np.hstack([self.X_train, self.y_train.reshape(-1, 1)])
        self.X_train, self.y_train = create_sequences(
            data_train, sequence_length=self.sequence_length, target_col="Close", is_df=False
        )

        # Concatenate features and targets for sequence creation (test)
        data_test = np.hstack([self.X_test, self.y_test.reshape(-1, 1)])
        self.X_test, self.y_test = create_sequences(
            data_test, sequence_length=self.sequence_length, target_col="Close", is_df=False
        )

        data_val = np.hstack([self.X_val, self.y_val.reshape(-1, 1)])
        self.X_val, self.y_val = create_sequences(
            data_val, sequence_length=self.sequence_length, target_col="Close", is_df=False
        )

    def build_model(self, input_dim, hidden_dim, num_layers, dropout, learning_rate):
        self.model = RNN(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
        return self.model

    def train(self, batch_size, epochs, patience=5):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
        )

    def predict(self):
        predictions = self.model.predict(self.X_test)
        predictions = self.target_scaler.inverse_transform(predictions)
        return predictions.flatten()

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_rnn_model.keras")
        self.model.save(model_path)

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.

        Parameters:
            predictions (np.array): Predicted values for the test data.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"RNN_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions,
        })
        prediction_df.to_csv(prediction_path, index=False)

    def tune_hyperparameters(self, n_trials=50):
        def objective(trial):
            # Suggest hyperparameters
            hidden_dim = trial.suggest_int("hidden_dim", 50, 300, step=50)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int("batch_size", 16, 64, step=16)

            # Build and compile model
            self.build_model(input_dim=self.X_train.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate)

            # Train the model
            early_stopping = TFKerasPruningCallback(trial, monitor="val_loss")
            history = self.model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_val, self.y_val),
                batch_size=batch_size,
                epochs=30,
                callbacks=[early_stopping],
                verbose=0
            )

            # Return the validation loss for the last epoch
            return history.history["val_loss"][-1]

        # Create an Optuna study and optimize
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("Best parameters:", study.best_params)
        print("Best validation loss:", study.best_value)
        return study.best_params

    def run(self, epochs=30, early_stop_patience=10):
        print(f"Training RNN model for {self.stock_name}...")
        # best_params = self.tune_hyperparameters(n_trials=15)
        best_params = {'num_layers': 1, 'hidden_dim': 300, 'dropout': 0.1, 'learning_rate': 0.00031703223453147364, 'batch_size': 16}

        input_dim = self.X_train.shape[2]
        self.build_model(
            input_dim=input_dim,
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"]
            )

        self.train(batch_size=best_params["batch_size"], epochs=epochs)
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
