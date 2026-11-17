
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_DIR / "models"

METADATA_PATH = PROCESSED_DATA_DIR / "ravdess_metadata.csv"

MFCC_FEATURES_PATH = PROCESSED_DATA_DIR / "mfcc_features.npy"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.npy"
PLOTS_DIR = DATA_DIR / "plots"

LSTM_X_TRAIN_PATH = PROCESSED_DATA_DIR / "lstm_x_train.npy"
LSTM_X_TEST_PATH = PROCESSED_DATA_DIR / "lstm_x_test.npy"

Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"

LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
NORMALIZATION_PATH = MODEL_DIR / "normalization_stats.npz"

LSTM_MODEL_PATH = MODEL_DIR / "lstm_emotion_model.keras"
LSTM_ACCURACY_PLOT_PATH = PLOTS_DIR / "lstm_acc_plot.png"
LSTM_LOSS_PLOT_PATH = PLOTS_DIR / "lstm_loss_plot.png"

BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
LEARNING_RATE = 0.0005
VALIDATION_SIZE = 0.2
RANDOM_SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    required_files = [
        LSTM_X_TRAIN_PATH,
        LSTM_X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    x_train = np.load(LSTM_X_TRAIN_PATH)
    x_test = np.load(LSTM_X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)

    return x_train, x_test, y_train, y_test


def split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_train_main, x_val, y_train_main, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train
    )

    return x_train_main, x_val, y_train_main, y_val


def build_lstm(
    input_shape: tuple[int, int],
    num_class: int
) -> Sequential:

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(Dropout(0.3))

    model.add(
        LSTM(
            units=32,
            return_sequences=False
        )
    )

    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(num_class, activation="softmax"))

    return model


def compile_model(model: Sequential) -> Sequential:

    optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    model: Sequential,
    x_train_main: np.ndarray,
    y_train_main: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray
):

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        x_train_main,
        y_train_main,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stopping]
    )

    return history


def save_model(model: Sequential) -> None:

    model.save(LSTM_MODEL_PATH)
    print("Model is saved")


def plot_training_history(history) -> None:

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("LSTM Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LSTM_ACCURACY_PLOT_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LSTM_LOSS_PLOT_PATH)
    plt.close()


def main():

    x_train, x_test, y_train, y_test = load_training_data()

    x_train_main, x_val, y_train_main, y_val = split_train_validation(
        x_train,
        y_train
    )

    input_shape = (x_train_main.shape[1], x_train_main.shape[2])
    num_class = len(np.unique(y_train))

    model = build_lstm(
        input_shape=input_shape,
        num_class=num_class
    )

    model = compile_model(model)

    model.summary()

    history = train_model(
        model,
        x_train_main,
        y_train_main,
        x_val,
        y_val
    )

    save_model(model)

    plot_training_history(history)

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
