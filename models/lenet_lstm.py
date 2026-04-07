from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, Add, Flatten, Reshape

def build_model():
    input_layer = Input(shape=(7, 9))

    x = Reshape((7, 3, 3, 1))(input_layer)

    # Residual block
    conv1 = TimeDistributed(Conv2D(16, (2,2), activation='relu', padding='same'))(x)
    conv2 = TimeDistributed(Conv2D(16, (2,2), activation='relu', padding='same'))(conv1)

    res = Add()([conv1, conv2])

    x = TimeDistributed(Flatten())(res)

    x = LSTM(64)(x)

    x = Dense(32, activation='relu')(x)
    output = Dense(9)(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer='adam', loss='huber', metrics=['mae'])

    return model