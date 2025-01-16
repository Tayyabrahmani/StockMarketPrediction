from machine_learning_models.ARIMA import ARIMAStockModel
from machine_learning_models.CNN import CNNStockModel
from machine_learning_models.LSTM import LSTMStockModel
from machine_learning_models.RNN import RNNStockModel
from machine_learning_models.SVR import SVRStockModel
from machine_learning_models.Transformers import TransformerStockModel
from machine_learning_models.XGBoost import XGBoostStockModel
from machine_learning_models.ARIMAXGB import DWT_ARIMA_GSXGB
from machine_learning_models.Crossformer import CrossformerStockModel
import traceback

class MachineLearningFramework:
    """
    A framework to execute selected machine learning models for stock prediction.
    """

    def __init__(self, file_path, stock_name):
        """
        Initializes the Machine Learning Framework.

        Parameters:
            file_path (str): Path to the input data file.
            stock_name (str): Name of the stock being analyzed.
            sequence_length (int): Number of past steps used for predictions.
            test_size (float): Fraction of data used for testing.
        """
        self.file_path = file_path
        self.stock_name = stock_name
        self.models = {
            "ARIMA": ARIMAStockModel,
            "XGBoost": XGBoostStockModel,
            "RNN": RNNStockModel,
            "LSTM": LSTMStockModel,
            "CNN": CNNStockModel,
            "SVR": SVRStockModel,
            "Transformers": TransformerStockModel,
            "ARIMA-XGB": DWT_ARIMA_GSXGB,
            "Crossformers": CrossformerStockModel,
        }
        self.selected_models = []

    def select_models(self, selected_model_names):
        """
        Updates the selected models based on user input.

        Parameters:
            selected_model_names (list): List of model names selected by the user.
        """
        self.selected_models = [
            self.models[model_name] for model_name in selected_model_names if model_name in self.models
        ]
        if not self.selected_models:
            raise ValueError("No valid models selected. Please choose from the available models.")

    def run_models(self):
        """
        Runs all selected models and saves their outputs.
        """
        for model_class in self.selected_models:
            print(f"\nRunning {model_class.__name__} for stock: {self.stock_name}...")
            try:
                # Instantiate and run the model
                if hasattr(model_class, "sequence_length"):
                    model = model_class(
                        file_path=self.file_path,
                        stock_name=self.stock_name,
                    )
                else:
                    model = model_class(
                        file_path=self.file_path,
                        stock_name=self.stock_name,
                    )
                model.run()

                # Save the model and predictions
                print(f"{model_class.__name__} successfully completed for {self.stock_name}.")
            except Exception as e:
                print(f"Error running {model_class.__name__}: {e}")
                print(traceback.format_exc())
                continue
