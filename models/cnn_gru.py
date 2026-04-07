from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, TimeDistributed, Conv2D, Flatten, Reshape

def build_model():
    input_layer = Input(shape=(7, 9))

    x = Reshape((7, 3, 3, 1))(input_layer)

    x = TimeDistributed(Conv2D(16, (2,2), activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)

    x = GRU(64)(x)

    x = Dense(32, activation='relu')(x)
    output = Dense(9)(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer='adam', loss='huber', metrics=['mae'])

    return model