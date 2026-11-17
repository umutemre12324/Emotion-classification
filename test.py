"""Evaluate the saved LSTM emotion classifier on the held-out test set."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
PLOTS_DIR = PROJECT_DIR / "data" / "plots"

MODEL_PATH = MODEL_DIR / "lstm_emotion_model.keras"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
X_TEST_PATH = PROCESSED_DATA_DIR / "lstm_x_test.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "lstm_confusion_matrix.png"


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    for path in (MODEL_PATH, LABEL_ENCODER_PATH, X_TEST_PATH, Y_TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {path}\n"
                "Önce README'deki veri hazırlama ve eğitim adımlarını çalıştırın."
            )

    x_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    if len(x_test) != len(y_test):
        raise ValueError("Test özellikleri ile etiket sayısı eşleşmiyor.")
    return x_test, y_test


def main() -> None:
    x_test, y_test = load_test_data()
    encoder = joblib.load(LABEL_ENCODER_PATH)
    model = load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    class_ids = np.arange(len(encoder.classes_))
    target_names = [str(label) for label in encoder.classes_]
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}\n")
    print("Sınıflandırma raporu:")
    print(classification_report(
        y_test,
        predictions,
        labels=class_ids,
        target_names=target_names,
        zero_division=0
    ))

    matrix = confusion_matrix(y_test, predictions, labels=class_ids)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=target_names
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    axis.set_title("LSTM Duygu Sınıflandırma - Test Confusion Matrix")
    figure.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(figure)
    print(f"Confusion matrix kaydedildi: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
