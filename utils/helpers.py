import numpy as np

def create_sequences(data, window=7):
    """
    Converts time-series into sequences for LSTM

    Input:
        data → numpy array (num_days, 4)

    Output:
        X → (samples, window, 4)
        y → (samples, 4)
    """

    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])

    return np.array(X), np.array(y)