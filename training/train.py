from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import os


def train_and_select_best(X, y, model_builders):
    """
    Trains multiple models and selects the best based on MAE.

    Args:
        X: input sequences
        y: targets
        model_builders: dict {name: build_function}

    Returns:
        best_model, results
    """

    # Time-series split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    results = {}
    best_mae = float('inf')
    best_model = None
    best_name = None

    # Ensure output folder exists
    os.makedirs("outputs", exist_ok=True)

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    for name, build_fn in model_builders.items():
        print(f"\nTraining {name}...")

        model = build_fn()

        # Train model
        model.fit(
            X_train, y_train,
            epochs=25,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )

        # Evaluate
        loss, mae = model.evaluate(X_test, y_test, verbose=0)

        print(f"{name} -> MAE: {mae:.4f}")

        results[name] = mae

        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    # Save best model in modern format
    best_model.save("outputs/best_model.keras")

    print("\nBest Model:", best_name)
    print("Best MAE:", best_mae)

    return best_model, results