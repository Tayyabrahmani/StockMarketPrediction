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
from machine_learning_models.evaluation import predict_and_inverse_transform

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self._get_positional_encoding(d_model, max_seq_len)

    def _get_positional_encoding(self, d_model, max_seq_len):
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.positional_encoding[: tf.shape(x)[1], :]

def transformer_encoder(inputs, d_model, n_heads, dim_feedforward, dropout):
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(dim_feedforward, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

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

    def build_model(self):
        input_seq = Input(shape=(self.sequence_length, self.X_train.shape[-1]))
        
        # Encoder
        enc_inputs = Dense(self.hyperparameters["d_model"])(input_seq)
        enc_inputs = PositionalEncoding(
            self.hyperparameters["d_model"], self.hyperparameters["max_seq_len"]
        )(enc_inputs)

        for _ in range(self.hyperparameters["n_encoder_layers"]):
            enc_inputs = transformer_encoder(
                enc_inputs,
                self.hyperparameters["d_model"],
                self.hyperparameters["n_heads"],
                self.hyperparameters["dim_feedforward"],
                self.hyperparameters["dropout"],
            )
        
        # Decoder (same input as encoder)
        dec_inputs = Dense(self.hyperparameters["d_model"])(input_seq)  # Use the same sequence
        dec_inputs = PositionalEncoding(
            self.hyperparameters["d_model"], self.hyperparameters["max_seq_len"]
        )(dec_inputs)

        for _ in range(self.hyperparameters["n_encoder_layers"]):
            dec_inputs = transformer_encoder(
                dec_inputs,
                self.hyperparameters["d_model"],
                self.hyperparameters["n_heads"],
                self.hyperparameters["dim_feedforward"],
                self.hyperparameters["dropout"],
            )

        # Output Layer
        outputs = TimeDistributed(Dense(1))(dec_inputs)
        self.model = Model(input_seq, outputs)  # Single input now
        self.model.compile(optimizer=Adam(learning_rate=self.hyperparameters["learning_rate"]),
                        loss="mse")

    def train(self, batch_size):
        """
        Trains the Transformer model.
        """
        self.model.fit(
            self.X_train,  # Single input sequence
            self.y_train,
            batch_size=batch_size,
            epochs=self.hyperparameters["epochs"],
        )

    def predict(self):
        """
        Generates predictions for the test data.
        """
        predictions = self.model.predict(self.X_test, verbose=1)  # Single input sequence
        predictions = self.target_scaler.inverse_transform(predictions.squeeze())
        # predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions

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
        prediction_path = os.path.join(prediction_dir, f"Transformer_{self.stock_name}_predictions.csv")

        final_predictions = predictions[:, -1]  # Extract last timestep predictions
        num_predictions = len(final_predictions)
        dates = pd.to_datetime(self.data.index[-num_predictions:])  # Ensure consistent lengths

        print(f"Dates length: {len(dates)}")
        print(f"Predictions length: {len(final_predictions)}")
        prediction_df = pd.DataFrame({
            "Date": dates,
            "Predicted Close": final_predictions,
        })
        prediction_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: builds, trains, generates predictions, and saves the model and predictions.
        """
        print(f"Training Transformer model for {self.stock_name}...")
        self.build_model()
        self.train(batch_size=16)
        predictions = self.predict()
        self.save_model()
        self.save_predictions(predictions)
        return predictions
