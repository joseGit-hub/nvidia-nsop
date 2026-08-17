# NVIDIA - Long-Term Stock Outlook Prediction (NSOP)

<img width="1497" height="841" alt="image" src="https://github.com/user-attachments/assets/e604d0b7-8ef7-4ff6-bd34-edc7cc819938" />

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

### Description

NVIDIA Long-Term Stock Outlook (NSOP) serves as an analytical framework for investors and researchers to evaluate historical market behavior and generate future-looking price projections. The application integrates sequential return-based modeling using PyTorch LSTMs to capture complex temporal dependencies in stock pricing. By processing historical market rates alongside deep learning sequence transformations, the tool assists users in visualizing speculative long-term trends and comparative multi-currency outlooks.

### Key Features

#### A. Data Pipeline & Preprocessing
* **Robust Cleaning:** Automatically parses historical CSV data (1999–2026) (Kaggle Data), standardizes column formats, and handles missing or erroneous data points.
* **Return Normalization:** Calculates percentage returns and scales data using MinMaxScaler within the $[-1, 1]$ range for optimal neural network convergence.

#### B. Deep Learning Architecture
* **LSTM Time-Series Model:** Utilizes a multi-layer Long Short-Term Memory (LSTM) network with dropout regularization to predict normalized sequence returns over a 60-day rolling window.
* **Iterative Forecasting:** Dynamically rolls forward day-by-day business-day predictions up to December 31, 2030, compounding projected returns to reconstruct future stock prices.

#### C. Dual-Currency Visualization
* **Custom Dark Theme:** Renders high-resolution, publication-ready visualizations featuring historical price action alongside projected outlooks.
* **Multi-Axis Scaling:** Features synchronous secondary Y-axis tracking for direct USD to PHP conversions based on set exchange rate baselines.
