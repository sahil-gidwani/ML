# LSTM_Gold_Prediction.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Generate synthetic gold price data
np.random.seed(0)
days = np.arange(1, 201)
gold_prices = 1200 + 10 * days + np.random.randn(200) * 5

# Data preprocessing
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(gold_prices.reshape(-1, 1))

# Create sequences for training
X, y = prices_scaled[:-1], prices_scaled[1:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model training
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Model evaluation
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(y_test), label='True Prices')
plt.plot(scaler.inverse_transform(y_pred), label='Predicted Prices')
plt.title(f'Gold Price Prediction (RMSE: {rmse:.2f})')
plt.legend()
plt.show()

"""
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is designed to handle sequences and time-dependent data. It is particularly useful for tasks that involve remembering and utilizing information over long time lags. LSTM networks are an enhancement of traditional RNNs and address some of the challenges associated with them, such as vanishing gradients.

Key features of LSTM networks:

1. **Memory Cells**: LSTM networks have memory cells that store information over a series of time steps. These memory cells can maintain information for long durations, making them suitable for modeling sequences.

2. **Gates**: LSTMs incorporate three types of gates – input gate, forget gate, and output gate. These gates control the flow of information into and out of the memory cell, allowing LSTMs to regulate what to remember and what to forget.

    - **Input Gate**: It decides which information to update in the memory cell. It combines the current input and the previous memory cell content.
    
    - **Forget Gate**: It decides which information to discard from the memory cell. It allows the network to forget unnecessary or irrelevant information.
    
    - **Output Gate**: It decides what to output from the memory cell. The output can be based on the current input and the memory cell content.

3. **Backpropagation Through Time (BPTT)**: Like traditional RNNs, LSTMs are trained using BPTT, a variant of backpropagation, to adjust weights and learn from sequences of data.

4. **Vanishing Gradient Problem**: LSTMs address the vanishing gradient problem associated with traditional RNNs. The gating mechanism allows LSTMs to capture long-term dependencies by regulating the flow of gradients during training.

Applications of LSTM networks:

- **Time Series Prediction**: LSTMs are used for time series forecasting, including stock price prediction, weather forecasting, and energy consumption prediction.

- **Natural Language Processing (NLP)**: LSTMs are applied to tasks such as text generation, language translation, and sentiment analysis.

- **Speech Recognition**: LSTMs are used for speech recognition tasks, including speech-to-text conversion.

- **Anomaly Detection**: LSTMs can detect anomalies in sequences of data, such as identifying fraudulent transactions or network intrusions.

- **Image Captioning**: LSTMs are combined with Convolutional Neural Networks (CNNs) to generate textual descriptions of images.

Overall, LSTM networks are powerful tools for modeling sequential data and have wide-ranging applications in fields like finance, healthcare, natural language processing, and more. Their ability to capture long-term dependencies and remember past information makes them suitable for a variety of time series and sequential tasks.
"""
