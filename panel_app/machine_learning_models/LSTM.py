import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import pickle
import pandas as pd
import numpy as np
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    create_sequences,
    train_test_split_time_series,
    fill_na_values,
    extract_date_features
)
from machine_learning_models.evaluation import (
    predict_and_inverse_transform,
)
from skimage.restoration import denoise_wavelet
import optuna
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf

class LSTMStockModel:
    def __init__(self, file_path, stock_name):
        """
        Initializes the LSTMStockModel with the necessary preprocessing and hyperparameters.

        Parameters:
            file_path (str): Path to the input CSV file.
            stock_name (str): Name of the stock being analyzed.
            hyperparameters (dict, optional): Model hyperparameters.
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
            self.X_train, self.y_train
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

    def build_model(self, input_dim, hidden_dims, num_layers, dropout, learning_rate, bidirectional=False):
        """
        Builds and initializes the LSTM model.

        Parameters:
            input_dim (int): Number of input features.
            hidden_dims (list of int): Number of units in each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate.
        """
        self.model = Sequential()
        self.model.add(Input(shape=(self.sequence_length, input_dim)))

        for i in range(num_layers):
            lstm_layer = LSTM(hidden_dims[i], return_sequences=(i < num_layers - 1))
            if bidirectional:
                lstm_layer = Bidirectional(lstm_layer)
            self.model.add(lstm_layer)
            self.model.add(Dropout(dropout))

        # Output layer
        self.model.add(Dense(1))

        # Compile the model with gradient clipping
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss="mse",
            metrics=["mae"],
        )
        return self.model

    def train(self, batch_size, learning_rate, epochs, early_stop_patience):
        """
        Trains the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() before training.")

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=early_stop_patience, restore_best_weights=True
        )

        # Reduce learning rate on plateau callback
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        # Train the model
        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1,
        )
        return history

    def custom_train_loop(self, epochs):
        """
        Custom training loop with explicit control over training and gradient clipping.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        loss_fn = tf.keras.losses.MeanSquaredError()

        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(32)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training loop
            for X_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(X_batch, training=True)
                    loss = loss_fn(y_batch, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Validation loop
            val_loss = 0
            for X_val_batch, y_val_batch in val_dataset:
                val_predictions = self.model(X_val_batch, training=False)
                val_loss += loss_fn(y_val_batch, val_predictions).numpy()

            print(f"Validation Loss: {val_loss / len(val_dataset):.4f}")

    def predict(self):
        """
        Generates predictions for the test data, reverses the differencing to get the original scale, 
        and returns the final predictions.

        Returns:
            np.array: Predictions in the original scale.
        """
        predictions = self.model.predict(self.X_test)
        predictions = self.target_scaler.inverse_transform(predictions)
        return predictions.flatten()

    def save_model(self):
        """
        Saves the trained LSTM model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_lstm_model.pkl")
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
        prediction_path = os.path.join(prediction_dir, f"LSTM_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions,
        })
        prediction_df.to_csv(prediction_path, index=False)

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning.
        """
        # Define the hyperparameter search space
        num_layers = trial.suggest_int("num_layers", 1, 5)
        hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 50, 300) for i in range(num_layers)]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Build and train the model
        self.build_model(input_dim=self.X_train.shape[2], hidden_dims=hidden_dims, num_layers=num_layers, dropout=dropout)
        self.train(batch_size=batch_size, learning_rate=learning_rate, epochs=50, early_stop_patience=10)

        # Evaluate on the validation set
        val_predictions = self.model.predict(self.X_val)
        val_predictions_original_scale = self.target_scaler.inverse_transform(val_predictions)

        y_val_actual = self.target_scaler.inverse_transform(self.y_val.reshape(-1, 1))
        rmse = np.sqrt(np.mean((val_predictions_original_scale.flatten() - y_val_actual.flatten()) ** 2))

        return rmse

    def tune_hyperparameters(self, n_trials=50):
        """
        Perform hyperparameter tuning using Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)

        print("Best trial:")
        print(study.best_trial)
        print("Best hyperparameters:")
        print(study.best_params)

        return study.best_params

    def run(self, epochs=150, early_stop_patience=10):
        """
        Runs the full pipeline: trains the model, generates predictions, and saves the model and predictions.
        """
        # best_params = self.tune_hyperparameters(n_trials=20)
        best_params = {'num_layers': 1, 'hidden_dim_0': 269, 'dropout': 0.1757380485131196, 'learning_rate': 9.668279090525918e-05, 'batch_size': 16}

        print("Building the LSTM model...")
        # self.build_model(input_dim=self.X_train.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.build_model(
            input_dim=self.X_train.shape[2],
            hidden_dims=[best_params[f"hidden_dim_{i}"] for i in range(best_params["num_layers"])],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"],
        )

        print("Training the LSTM model...")
        # self.train(batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, early_stop_patience=early_stop_patience)

        self.train(
            batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
            epochs=epochs,
            early_stop_patience=early_stop_patience,
        )

        print("Generating predictions...")
        predictions = self.predict()

        print("Saving predictions...")
        self.save_predictions(predictions)
        return predictions