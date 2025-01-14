import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
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
import optuna

class CNN(tf.keras.Model):
    def __init__(self, input_dim, sequence_length, num_filters=128, dropout_rate=0.3, embed_dim=128):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length

        self.conv1 = layers.Conv1D(filters=num_filters, kernel_size=3, padding="same", activation="relu",
                                   input_shape=(sequence_length, input_dim))
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(filters=num_filters * 2, kernel_size=5, padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv1D(filters=num_filters * 4, kernel_size=7, padding="same", activation="relu")
        self.bn3 = layers.BatchNormalization()

        self.global_pool = layers.GlobalAveragePooling1D()

        self.project_to_embed = layers.Dense(embed_dim, activation="relu")

        self.attention = layers.Attention()

        self.dropout = layers.Dropout(dropout_rate)

        self.fc1 = layers.Dense(64, activation="relu")
        self.fc2 = layers.Dense(32, activation="relu")
        self.fc3 = layers.Dense(1, activation="linear")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # Reshape before attention: (batch_size, sequence_length, feature_dim)
        x = self.project_to_embed(x)

        # Prepare for Attention layer: (batch_size, sequence_length, feature_dim)
        attention_input = tf.expand_dims(x, axis=1) if len(x.shape) == 2 else x

        # Apply Attention
        x = self.attention([attention_input, attention_input])
        x = tf.reduce_mean(x, axis=1)  # Reduce sequence dimension

        x = self.dropout(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CNNStockModel:
    def __init__(self, file_path, stock_name):
        self.file_path = file_path
        self.stock_name = stock_name
        self.model = None

        # Load and preprocess data
        self.data = load_data(self.file_path)

        # Create lagged features
        self.data = create_lagged_features(self.data, target_col="Close")
        self.data = fill_na_values(self.data)
        self.data = extract_date_features(self.data)

        X, y = create_sequences(self.data, sequence_length=30, target_col="Close")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_time_series(
            X, y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split_time_series(
            self.X_train, self.y_train
        )

        (self.X_train, self.X_test, self.X_val,
         self.y_train, self.y_test, self.y_val,
         self.feature_scaler, self.target_scaler) = preprocess_data(self.X_train, self.X_test, self.X_val,
                                                                    self.y_train, self.y_test, self.y_val,
                                                                    add_feature_dim=True)

    def objective(self, trial):
        # Suggest hyperparameters for Optuna to tune
        num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 10, 30)

        # Initialize and train the model
        sequence_length = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]
        model = CNN(
            input_dim=input_dim,
            sequence_length=sequence_length,
            num_filters=num_filters, 
            dropout_rate=dropout_rate
            )

        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        # Training
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        )
        model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=32,
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
        study = optuna.create_study(direction="minimize")
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
        best_params = {'num_filters': 32, 'dropout_rate': 0.21596336761657278, 'learning_rate': 0.0006700845786374102, 'epochs': 20}
        print(f"Best parameters found: {best_params}")

        sequence_length = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]

        self.model = CNN(input_dim=input_dim,
                         sequence_length=sequence_length,
                         num_filters=best_params["num_filters"],
                         dropout_rate=best_params["dropout_rate"],
        )

        print("Training model...")
        self.train(learning_rate=best_params["learning_rate"], epochs=best_params["epochs"])

        print("Generating predictions...")
        predictions = self.predict()

        print("Saving predictions...")
        self.save_predictions(predictions)

        print("Saving the model...")
        self.save_model()
        
        return predictions