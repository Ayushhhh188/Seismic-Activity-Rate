from models.cnn_lstm import build_model as cnn_lstm
from models.cnn_gru import build_model as cnn_gru
from models.lenet_lstm import build_model as lenet_lstm
from models.resnet_lstm import build_model as resnet_lstm

from sklearn.model_selection import train_test_split

def compare_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "CNN_LSTM": cnn_lstm,
        "CNN_GRU": cnn_gru,
        "LeNet_LSTM": lenet_lstm,
        "ResNet_LSTM": resnet_lstm
    }

    results = {}

    for name, build_fn in models.items():
        print(f"\nTraining {name}...")

        model = build_fn()

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )

        loss, mae = model.evaluate(X_test, y_test, verbose=0)

        results[name] = mae
        print(f"{name} MAE: {mae}")

    return results