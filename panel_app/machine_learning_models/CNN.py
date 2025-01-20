import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import os
import pickle
import numpy as np
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
import optuna

class CNN(tf.keras.Model):
    def __init__(self, input_dim, sequence_length, num_filters=128, dropout_rates=(0.3, 0.3),
                 embed_dim=128, kernel_sizes=(3, 5, 7), activation="relu",
                 pooling_type="global_avg", num_conv_layers=3, use_residual=False):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.use_residual = use_residual

        self.conv_layers = []
        self.bn_layers = []

        # Dynamically add convolutional layers
        for i in range(num_conv_layers):
            filters = num_filters * (2 ** i)  # Double filters for each layer
            kernel_size = kernel_sizes[min(i, len(kernel_sizes) - 1)]  # Cycle through kernel sizes
            conv_layer = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                       padding="same", activation=activation,
                                       input_shape=(sequence_length, input_dim) if i == 0 else None)
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(layers.BatchNormalization())

        # Predefine Conv1D layer for residual alignment
        self.residual_projection = layers.Conv1D(filters=num_filters, kernel_size=1, padding="same")

        # Add global pooling
        if pooling_type == "global_avg":
            self.pooling = layers.GlobalAveragePooling1D()
        elif pooling_type == "global_max":
            self.pooling = layers.GlobalMaxPooling1D()

        # Dense projection to embedding dimension
        self.project_to_embed = layers.Dense(embed_dim, activation=activation)

        # Dropout layers
        self.dropout1 = layers.Dropout(dropout_rates[0])
        self.dropout2 = layers.Dropout(dropout_rates[1])

        # Fully connected layers
        self.fc1 = layers.Dense(64, activation=activation)
        self.fc2 = layers.Dense(32, activation=activation)
        self.fc3 = layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = inputs
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            x = bn(x)
            if self.use_residual:
                # Use predefined residual projection layer for alignment
                if x.shape[-1] != inputs.shape[-1]:  # Adjust only if dimensions mismatch
                    residual_projection = layers.Conv1D(filters=x.shape[-1], kernel_size=1, padding="same")
                    inputs = residual_projection(inputs)
                x += inputs  # Residual connection

        # Apply pooling
        x = self.pooling(x)
        print(f"Shape after pooling: {x.shape}")  # Debugging step

        # Ensure the output shape is fully defined before passing to Dense
        x = tf.reshape(x, [-1, x.shape[-1]])  # Flatten to (batch_size, features)
        x = self.project_to_embed(x)

        # Dropout and fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CNNStockModel:
    def __init__(self, file_path, stock_name):
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

    def objective(self, trial):
        # Hyperparameter suggestions
        num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
        dropout_rate1 = trial.suggest_float("dropout_rate1", 0.1, 0.5)
        dropout_rate2 = trial.suggest_float("dropout_rate2", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        optimizer_type = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "selu"])
        pooling_type = trial.suggest_categorical("pooling_type", ["global_avg", "global_max"])
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)
        use_residual = trial.suggest_categorical("use_residual", [True, False])

        # Initialize the model
        sequence_length = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]
        model = CNN(
            input_dim=input_dim,
            sequence_length=sequence_length,
            num_filters=num_filters,
            dropout_rates=(dropout_rate1, dropout_rate2),
            embed_dim=128,  # Fixed, or you can tune this as well
            activation=activation,
            pooling_type=pooling_type,
            num_conv_layers=num_conv_layers,
            use_residual=use_residual,
        )

        # Compile the model with the selected optimizer
        if optimizer_type == "adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == "sgd":
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Training
        early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        lr_scheduler = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0)
        model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=trial.suggest_int("epochs", 10, 30),
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping, lr_scheduler],
        )

        # Evaluate on validation data
        val_loss = model.evaluate(self.X_val, self.y_val, verbose=0)[0]
        return val_loss

    def run_tuning(self, n_trials=50, update_optuna_study=False):
        if not update_optuna_study:
            return

        # Define Optuna study
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials)

        print("Best hyperparameters:", study.best_params)
        print("Best validation loss:", study.best_value)

        return study.best_params

    def train(self, learning_rate=1e-3, epochs=10, batch_size=32):
        """
        Trains the CNN model.
        """
        # Define optimizer, loss, and learning rate scheduler
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        self.model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Train the model
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[lr_scheduler, early_stopping],
        )

    def predict(self):
        """
        Generates predictions for the test data and inverse transforms them.

        Returns:
            np.array: Predictions in the original scale.
        """
        predictions = self.model.predict(self.X_test)
        predictions_original_scale = self.target_scaler.inverse_transform(predictions)
        return predictions_original_scale.flatten()

    def save_model(self):
        """
        Saves the trained CNN model as a pickle file.
        """
        model_dir = "Output_Data/saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.stock_name}_cnn_model.keras")
        self.model.save(model_path)

    def save_predictions(self, predictions):
        """
        Saves the predictions as a CSV file.

        Parameters:
            predictions (np.array): Predicted values for the test data.
        """
        prediction_dir = "Output_Data/saved_predictions"
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_path = os.path.join(prediction_dir, f"CNN_{self.stock_name}_predictions.csv")

        # Save actual vs predicted values
        prediction_df = pd.DataFrame({
            "Date": pd.to_datetime(self.data.index[-len(predictions):]),
            "Predicted Close": predictions
        })
        prediction_df.to_csv(prediction_path, index=False)

    def run(self):
        """
        Runs the full pipeline: trains the model, generates predictions, and saves the model and predictions.
        """
        # best_params = self.run_tuning(n_trials=20, update_optuna_study=True)
        best_params = {'num_filters': 128, 'dropout_rates': (0.25551670158269935, 0.26769817375735305),
                       'embed_dim': 128, 'kernel_sizes': (3, 5, 7),
                       'num_conv_layers': 3, 'batch_size': 32,
                       'learning_rate': 0.0001449580804866673,
                       'epochs': 50}
        print(f"Best parameters found: {best_params}")

        sequence_length = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]

        self.model = CNN(input_dim=input_dim,
                         sequence_length=sequence_length,
                         num_filters=best_params["num_filters"],
                         dropout_rates=best_params["dropout_rates"],
                         embed_dim=best_params["embed_dim"],
                         kernel_sizes=best_params["kernel_sizes"],
                         num_conv_layers=best_params["num_conv_layers"],
        )

        print("Training model...")
        self.train(learning_rate=best_params["learning_rate"], epochs=best_params["epochs"], batch_size=best_params["batch_size"])

        print("Generating predictions...")
        predictions = self.predict()

        print("Saving predictions...")
        self.save_predictions(predictions)

        print("Saving the model...")
        self.save_model()
        
        return predictions