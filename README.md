# Stock Market Forecasting Panel App

This repository contains a **Panel-based web application** for interactive **stock market forecasting**, developed as part of a master's thesis. The application enables users to compare multiple forecasting models—including **ARIMA, XGBoost, LSTM, RNN, and Transformer-based architectures (Crossformers, PatchTST)**—across various asset classes and stock markets.

## 🔍 Research Motivation
Stock market forecasting is inherently complex due to its **nonlinear and volatile** nature. Traditional statistical models like **ARIMA** and machine learning techniques such as **XGBoost** have shown effectiveness, but **Transformer-based models** have recently gained traction due to their superior ability to capture **long-term dependencies and complex patterns**.

This study provides a **comparative analysis** of these models on stock, commodity, and forex data from **India, Germany, and the USA**, evaluating their predictive performance based on:
- **SMAPE (Symmetric Mean Absolute Percentage Error)**
- **RMSE (Root Mean Square Error)**
- **Directional Accuracy (Trend Prediction Effectiveness)**

## 🌟 Features of the Panel App
✔ **Interactive Model Selection** – Compare ARIMA, XGBoost, LSTM, RNN, and Transformer-based models.  
✔ **Multi-Market Forecasting** – Evaluate stock predictions across different financial markets.  
✔ **User-Friendly Visualization** – View time-series forecasts, error metrics, and trend accuracy.  
✔ **Historical Stock Data Processing** – Analyze market trends using real-world financial datasets.  
✔ **Flexible Forecasting Horizons** – Short-term vs. long-term prediction comparisons.  

## 📂 Project Structure
```
panel_app/
│── app.py                    # Main entry point for the Panel app
│── logging_config.py          # Configures logging for debugging
│── train_model.py             # Script for training stock forecasting models
│── components/                # Modular app components
│   ├── ForecastTable.py       # Displays forecast results
│   ├── MachineLearningFramework.py  # Implements different forecasting models
│   ├── metrics.py             # Defines evaluation metrics (SMAPE, RMSE, etc.)
│   ├── prediction_duration.py # Handles time horizon settings
│   ├── Selectors.py           # UI selectors for model configuration
│   ├── Sidebar.py             # Sidebar UI component
```

## 🛠 Installation
Ensure you have **Python 3.8+** installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Application
To start the **Panel-based forecasting tool**, run:

```bash
python panel_app/app.py
```

## 📊 Model Training
To train the models before running predictions:

```bash
python panel_app/train_model.py
```

## 📜 Research Contributions
- **Comprehensive Benchmarking**: Provides a structured comparison of forecasting models across diverse market conditions.  
- **Transformer Model Evaluation**: Highlights the strengths and weaknesses of **Crossformers** and **PatchTST** in financial time-series forecasting.  
- **User-Centric Financial Tool**: Enables interactive model selection and visualization, making stock prediction more accessible.  

## 🤝 Contributing
Contributions, suggestions, and feedback are welcome! Feel free to raise issues or submit pull requests.

## 📄 License
This project is licensed under the **MIT License**.
