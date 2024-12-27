
# Deep Learning Time-Series Forecasting Models

This repository showcases a collection of time-series forecasting models built using deep learning techniques. The models implemented include Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), Bidirectional RNN (Bi-RNN), Bidirectional LSTM (Bi-LSTM), Bidirectional GRU (Bi-GRU), as well as hybrid models such as CNN+RNN, CNN+LSTM, and CNN+GRU. Additionally, the Temporal Fusion Transformer (TFT) model has been implemented for advanced time-series forecasting tasks.

## Table of Contents

- [Overview](#overview)
- [Implemented Models](#implemented-models)
  - [RNN](#rnn)
  - [LSTM](#lstm)
  - [GRU](#gru)
  - [Bi-RNN](#bi-rnn)
  - [Bi-LSTM](#bi-lstm)
  - [Bi-GRU](#bi-gru)
  - [CNN + RNN](#cnn--rnn)
  - [CNN + LSTM](#cnn--lstm)
  - [CNN + GRU](#cnn--gru)
  - [Temporal Fusion Transformer (TFT)](#temporal-fusion-transformer-tft)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Normalization & Standardization](#normalization--standardization)
  - [Time-Series Specific Preprocessing](#time-series-specific-preprocessing)
- [Sequential Data Modeling with BPTT](#sequential-data-modeling-with-bptt)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository is designed for experimenting with various deep learning architectures applied to time-series forecasting tasks. The models can be used for predicting continuous values (e.g., stock prices, sensor data, etc.) over time. Each model architecture leverages different techniques to capture patterns and trends in time-series data, with a particular focus on recurrent and convolutional layers, as well as cutting-edge architectures like the Temporal Fusion Transformer (TFT).

### Key Features:

- **Multiple Deep Learning Models**: A wide range of architectures, from simple RNNs to sophisticated TFT.
- **Hybrid Models**: Combination of CNNs with RNNs, LSTMs, and GRUs to enhance feature extraction and time-series forecasting.
- **Flexible**: Models are implemented with customization options (e.g., hyperparameters) for ease of experimentation.
- **Optimized for Time-Series Data**: These models are optimized for handling sequential data and capturing temporal dependencies.

## Implemented Models

### RNN

- **Recurrent Neural Network (RNN)** is one of the most fundamental architectures for time-series analysis. It processes data sequentially and uses its internal state to capture temporal dependencies.

### LSTM

- **Long Short-Term Memory (LSTM)** addresses the vanishing gradient problem in traditional RNNs and can capture long-term dependencies in sequential data.

### GRU

- **Gated Recurrent Units (GRU)** are similar to LSTMs but use a simplified architecture, offering a balance between performance and computational efficiency.

### Bi-RNN

- **Bidirectional RNN (Bi-RNN)** processes the data in both forward and backward directions, capturing both past and future context.

### Bi-LSTM

- **Bidirectional LSTM (Bi-LSTM)** combines the advantages of LSTM and Bi-RNN, processing data in both directions and capturing long-term dependencies.

### Bi-GRU

- **Bidirectional GRU (Bi-GRU)** is similar to Bi-LSTM but with the computational efficiency of GRUs.

### CNN + RNN

- **CNN + RNN** architecture combines Convolutional Neural Networks (CNN) for feature extraction with RNNs for sequence modeling, allowing the model to focus on important local patterns and long-term dependencies.

### CNN + LSTM

- **CNN + LSTM** utilizes CNNs for automatic feature extraction followed by LSTM layers to capture temporal dependencies, making it effective for both spatial and temporal patterns.

### CNN + GRU

- **CNN + GRU** is similar to CNN + LSTM but uses GRU layers, providing a more efficient model for time-series forecasting.

### Temporal Fusion Transformer (TFT)

- **Temporal Fusion Transformer (TFT)** is a state-of-the-art architecture for time-series forecasting. It uses attention mechanisms to focus on important time steps and capture complex relationships between different variables.

## Data Cleaning & Preprocessing

Before applying deep learning models to time-series forecasting, proper data cleaning and preprocessing are crucial for the models to perform effectively.

### Handling Missing Values

- **Imputation**: Missing values can be imputed using methods such as forward-fill, backward-fill, or by filling with statistical values like the mean, median, or mode.
- **Drop Missing Data**: In some cases, missing values can be dropped, especially if the dataset is large, and the missing data represents only a small portion of it.

### Normalization & Standardization

- **Normalization**: For features with varying scales, normalization (scaling to the range [0, 1]) is often used to ensure uniformity across the data.
- **Standardization**: Standardizing the features (subtracting the mean and dividing by the standard deviation) is also important to ensure each feature contributes equally to the model’s training.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
```

### Time-Series Specific Preprocessing

- **Reshaping the Data**: Time-series data needs to be reshaped into a format suitable for sequential models. The data should be structured as `(samples, time steps, features)`.
- **Lag Features**: Including previous time steps as features can help the model capture temporal dependencies more effectively.
- **Sequence Splitting**: Split the data into training, validation, and test sets based on time, ensuring the test set always represents future data points.

```python
def create_sequences(data, time_steps=sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        labels.append(data[i + time_steps])
    return np.array(sequences), np.array(labels)
```

## Sequential Data Modeling with BPTT

**Backpropagation Through Time (BPTT)** is a specific variant of backpropagation used to train Recurrent Neural Networks (RNNs) and their variants (LSTM, GRU). It helps to optimize the model by computing gradients through the unfolding of the network across time steps. The key aspects of BPTT are:

- **Unfolding**: The RNN is unrolled over time so that each time step becomes a layer in the network.
- **Gradient Computation**: Gradients are calculated at each time step, and errors are propagated backward through the unrolled network.

### Key Steps in BPTT:
1. **Forward Pass**: The input sequence is passed through the RNN layer to compute the hidden states.
2. **Loss Calculation**: The predicted output is compared with the true output to compute the loss.
3. **Backward Pass**: The loss is backpropagated through each time step to compute gradients for the weights.
4. **Weight Update**: The weights are updated using an optimizer like Adam, SGD, etc., to minimize the loss.

BPTT allows RNNs to learn from sequential data by propagating gradients through multiple time steps. However, long sequences can cause the gradient to vanish or explode, which is why architectures like LSTM and GRU are preferred, as they address these issues.

## Dependencies

- Python 3.x
- TensorFlow >= 2.0
- TensorFlow Probability
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

## Usage

1. **Prepare Your Data**: Ensure your time-series data is in the correct format. The data should be divided into training, validation, and test sets.
   
2. **Run the Models**:
   - You can train any model by calling the `fit` function with your training data.
   - Example:
     ```python
     model = LSTM(input_shape=(24, 6))  # Example model
     model.fit(X_train, y_train, epochs=20, batch_size=32)
     ```

3. **Evaluate the Model**:
   - After training, you can evaluate the model on the test set:
     ```python
     loss, mae = model.evaluate(X_test, y_test)
     print(f"Test Loss: {loss}")
     print(f"Test MAE: {mae}")
     ```

4. **Make Predictions**:
   - Use the trained model to make predictions:
     ```python
     predictions = model.predict(X_test)
     ```

5. **Visualize the Results**:
   - You can visualize the predictions and the actual values to compare:
     ```python
     import matplotlib.pyplot as plt
     plt.plot(y_test, label='Actual')
     plt.plot(predictions, label='Predicted')
     plt.legend()
     plt.show()
     ```

## Results

### Model Performance

For each model, you can track the following metrics during evaluation:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Explained Variance Score (EVS)**
- **Root Mean Squared Error (RMSE)**


## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.
