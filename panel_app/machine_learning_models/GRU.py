import torch
import torch.nn as nn
import os
import pickle
import pandas as pd
from machine_learning_models.preprocessing import (
    load_data,
    create_lagged_features,
    preprocess_data,
    train_test_split_time_series,
)
from machine_learning_models.evaluation import predict_and_inverse_transform
from machine_learning_models.LSTM import LSTMStockModel

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class GRUStockModel(LSTMStockModel):
    def build_model(self, input_dim):
        """
        Builds and initializes the GRU model.
        """
        self.model = GRU(input_dim, hidden_dim=self.hyperparameters["hidden_dim"])
        return self.model
