from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, Flatten, Reshape

def build_model():
    """
    CNN + LSTM model for spatio-temporal seismic prediction
    """

    input_layer = Input(shape=(7, 9))

    x = Reshape((7, 3, 3, 1))(input_layer)
    # CNN applied per timestep
    x = TimeDistributed(Conv2D(16, (2,2), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)

    # LSTM for temporal learning
    x = LSTM(64, return_sequences=False)(x)

    # Dense layers
    x = Dense(32, activation='relu')(x)

    # Output: 4 grids prediction
    output = Dense(9)(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer='adam',
        loss='huber',
        metrics=['mae']
    )

    return model