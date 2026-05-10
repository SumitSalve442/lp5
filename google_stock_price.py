# google_stock_rnn.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


# 1. Load Dataset

df = pd.read_csv("Google_Stock_Price.csv", thousands=',')

# Convert Open column to numeric
data = pd.to_numeric(df['Open'], errors='coerce')

# Remove NaN values
data = data.dropna().values.reshape(-1, 1)


# 2. Scale Data

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# 3. Train-Test Split

train_size = int(len(data_scaled) * 0.8)

train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]


# 4. Create Dataset Function

def create_dataset(dataset):
    X = []
    y = []

    for i in range(60, len(dataset)):
        X.append(dataset[i-60:i, 0])
        y.append(dataset[i, 0])

    return np.array(X), np.array(y)


X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)



# 5. Reshape Data for RNN

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# 6. Build RNN Model

model = Sequential()

model.add(SimpleRNN(50, return_sequences=True, input_shape=(60, 1)))
model.add(SimpleRNN(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')



# 7. Model Summary
model.summary()


# 8. Train Model
model.fit(X_train, y_train, epochs=20, batch_size=32)


# 9. Prediction
predicted = model.predict(X_test)

predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y_test.reshape(-1, 1))


# 10. Plot Results
plt.figure(figsize=(10, 5))

plt.plot(real, color='red', label='Real Price')
plt.plot(predicted, color='blue', label='Predicted Price')

plt.title("Google Stock Price Prediction (RNN)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.show()