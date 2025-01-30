import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
import os
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
import optuna
from machine_learning_models.evaluation import predict_and_inverse_transform

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout):
        """Single Transformer encoder layer."""
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads, key_dim=model_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(model_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        """Forward pass for the encoder layer."""
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.norm1(inputs + attention_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

class TimeSeriesTransformer(Model):
    def __init__(self, num_features, sequence_length, model_dim=64, num_heads=4, ff_dim=128, num_layers=2, dropout=0.1):
        """
        Transformer model for multivariate time series forecasting.
        Args:
            num_features: Number of input features
            sequence_length: Length of input sequences
            model_dim: Dimensionality of the model embeddings
            num_heads: Number of attention heads
            ff_dim: Dimensionality of feed-forward network
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
        """
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = Dense(model_dim)
        self.positional_encoding = self._create_positional_encoding(sequence_length, model_dim) * 0.7
        # self.positional_encoding = tf.Variable(
        #     initial_value=tf.random.uniform([sequence_length, model_dim]),
        #     trainable=True,
        #     dtype=tf.float32
        # )

        self.encoder_layers = [
            TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout)
        self.output_layer = Dense(1)  # Forecast single target value (e.g., closing price)

    def _create_positional_encoding(self, sequence_length, model_dim):
        """Generates positional encodings."""
        positions = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
        pos_enc = np.zeros((sequence_length, model_dim))
        pos_enc[:, 0::2] = np.sin(positions * div_term)
        pos_enc[:, 1::2] = np.cos(positions * div_term)
        return tf.constant(pos_enc, dtype=tf.float32)

    def call(self, inputs, training=False):
        """Forward pass for the model."""
        x = self.embedding(inputs) + self.positional_encoding
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        x = self.dropout(x, training=training)
        return self.output_layer(x[:, -1])  # Output based on the final time step

class TransformerStockModel:
    def __init__(self, file_path, stock_name, hyperparameters=None):
        self.file_path = file_path
        self.stock_name = stock_name
        self.hyperparameters = hyperparameters or {
            "dec_seq_len": 30,
            "max_seq_len": 500,
            "d_model": 64,
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "n_heads": 8,
            "dim_feedforward": 256,
            "dropout": 0.2,
            "learning_rate": 0.0001,
            "epochs": 150, 
            "batch_size": 16, 
        }
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

        # Add the last 29 rows (sequence length) from the train data to create sequences
        self.X_test = np.vstack([self.X_train[-self.sequence_length:], self.X_test])
        self.y_test = np.concatenate([self.y_train[-self.sequence_length:], self.y_test])

        # Split train test split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train, test_size=0.2
        )

        # Scale data
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, self.feature_scaler, self.target_scaler = preprocess_data(self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, add_feature_dim=False)

        # Concatenate features and targets for sequence creation (train)
        data_train = np.hstack([self.X_train, self.y_train.reshape(-1, 1)])
        self.X_train, self.y_train = create_sequences(
            data_train, sequence_length=self.sequence_length, target_col="Close", is_df=False, is_transformers=True
        )

        # Concatenate features and targets for sequence creation (test)
        data_test = np.hstack([self.X_test, self.y_test.reshape(-1, 1)])
        self.X_test, self.y_test = create_sequences(
            data_test, sequence_length=self.sequence_length, target_col="Close", is_df=False, is_transformers=True
        )

        data_val = np.hstack([self.X_val, self.y_val.reshape(-1, 1)])
        self.X_val, self.y_val = create_sequences(
            data_val, sequence_length=self.sequence_length, target_col="Close", is_df=False, is_transformers=True
        )

    def train(self):
        """
        Trains the Transformer model with early stopping and learning rate reduction on plateau.
        """
        self.model = TimeSeriesTransformer(
            num_features=self.X_train.shape[2],
            sequence_length=self.X_train.shape[1],
            model_dim=self.hyperparameters["d_model"],
            num_heads=self.hyperparameters["n_heads"],
            ff_dim=self.hyperparameters["dim_feedforward"],
            num_layers=self.hyperparameters["n_encoder_layers"],
            dropout=self.hyperparameters["dropout"],
        )
        self.model.compile(optimizer=Adam(learning_rate=self.hyperparameters["learning_rate"]),
                        loss="mse", metrics=["mae"])

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # ReduceLROnPlateau callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,        # Reduce learning rate by a factor of 0.5
            patience=5,        # Wait for 5 epochs without improvement before reducing
            min_lr=1e-6,       # Minimum learning rate
            verbose=1
        )

        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.hyperparameters["batch_size"],
            epochs=self.hyperparameters["epochs"],
            verbose=2,
            callbacks=[early_stopping, reduce_lr],  # Add callbacks here
        )

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter tuning with pruning.
        """
        # Sample hyperparameters
        d_model = trial.suggest_int("d_model", 32, 128, step=32)
        num_heads = trial.suggest_int("n_heads", 2, 8, step=2)
        dim_feedforward = trial.suggest_int("dim_feedforward", 64, 512, step=64)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = 50  # Fixed number of epochs for trials

        # Update hyperparameters for this trial
        self.hyperparameters.update({
            "d_model": d_model,
            "n_heads": num_heads,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        })

        # Initialize and compile the model
        self.model = TimeSeriesTransformer(
            num_features=self.X_train.shape[2],
            sequence_length=self.X_train.shape[1],
            model_dim=d_model,
            num_heads=num_heads,
            ff_dim=dim_feedforward,
            num_layers=self.hyperparameters["n_encoder_layers"],
            dropout=dropout,
        )
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )

        # Train the model
        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[early_stopping, reduce_lr],
        )

        # Get the best validation loss during training
        best_val_loss = min(history.history["val_loss"])

        # Report the best validation loss to Optuna
        trial.report(best_val_loss, step=0)

        # Prune trial if not improving
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return best_val_loss

    def tune_hyperparameters(self, n_trials=50):
        """
        Runs Optuna hyperparameter tuning.
        """
        # Create a study with a pruner for early stopping
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)

        print("Best hyperparameters:", study.best_params)
        return study.best_params

    def predict(self):
        """
        Generates predictions for the test data and scales them back to original values.
        """
        predictions = self.model.predict(self.X_test, verbose=1)
        # Reshape to match scaler expectations if necessary
        predictions = predictions.reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions).flatten()

    def save_model(self):
        """
        Saves the trained Transformer model to disk.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_transformer_model.keras")
        self.model.save(model_path)

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.

        Parameters:
            predictions (np.array): Predicted values for the test data.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"Transformers_{self.stock_name}_predictions.csv")

        # Match timestamps with predictions
        prediction_dates = self.data.index[-len(predictions):]

        prediction_df = pd.DataFrame({
            "Date": prediction_dates,
            "Predicted Close": predictions,
        })

        prediction_df.to_csv(prediction_path, index=False)

    def save_hyperparameters(self):
        """
        Saves the hyperparameters as a CSV file.
        """
        hyperparam_dir = os.path.join("Output_Data", "Hyperparameters", "Transformers")
        os.makedirs(hyperparam_dir, exist_ok=True)
        hyperparam_path = os.path.join(hyperparam_dir, f"{self.stock_name}_hyperparameter.csv")

        hyperparam_df = pd.DataFrame.from_dict(self.hyperparameters, orient="index", columns=["Value"])
        hyperparam_df.to_csv(hyperparam_path)
        print(f"Hyperparameters saved to {hyperparam_path}")

    def run(self):
        """
        Runs the full pipeline: builds, trains, generates predictions, and saves the model and predictions.
        """
        self.best_params = self.tune_hyperparameters(n_trials=50)
        self.hyperparameters = self.best_params

        print(f"Training Transformer model for {self.stock_name}...")
        self.train()
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        self.save_hyperparameters()
        return predictions
