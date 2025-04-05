# Google Stock Price Prediction
This project predicts Google stock prices using a Long Short-Term Memory (LSTM) neural network. Built with Python and Keras, it processes historical stock data, trains an LSTM model, and visualizes predictions. Through this, I explored time series forecasting, data preprocessing, and deep learning techniques.

![image](https://github.com/user-attachments/assets/7cbfa8fb-f095-40d7-92d4-ab886e1b49a0)

## Dataset
**Source**: Historical (2012-2016) Google stock data (Google_train_data.csv for training, Google_test_data.csv for testing).  <br/>
**Features**: Date, Open, High, Low, Close, Volume.  <br/>
**Target**: Close price for prediction.
<br>
<br>

## Learning Process & Knowledge Gained

1. **Data Preprocessing**:
   - Converted Close from string to numeric, handling commas with pd.to_numeric(errors='coerce').
   - Dropped NaN values, reducing the dataset from 1258 to 1149 rows.
   - Scaled data to [0, 1] using MinMaxScaler for LSTM compatibility.

2. **Time Series Preparation**:
   - Created 60-day sequences (X_train) and next-day targets (y_train), resulting in 1089 samples.
   - Reshaped data to (samples, timesteps, features)—(1089, 60, 1)—for LSTM input.

3. **LSTM Model Design**:
   - Built a 4-layer LSTM with 100 units each, using return_sequences=True for stacking and False for the final output.
   - Added 20% dropout after each LSTM to prevent overfitting.
   - Used a single Dense layer for regression output, optimized with Adam and Mean Squared Error (MSE) loss.

4. **Training & Evaluation**:
   - Trained for 20 epochs with a batch size of 32, reducing MSE from 0.0380 to 0.0045.
   - Learned to handle missing test data steps, adding preprocessing and prediction logic.

5. **Visualization**:
   - Plotted actual vs. predicted Close prices, mastering inverse scaling with MinMaxScaler.


## Prerequisites
  - Python 3.6
  - Python, NumPy, Pandas, Matplotlib, scikit-learn, keras
