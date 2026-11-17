from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_DIR / "models"

METADATA_PATH = PROCESSED_DATA_DIR / "ravdess_metadata.csv"

MFCC_FEATURES_PATH = PROCESSED_DATA_DIR / "mfcc_features.npy"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.npy"

LSTM_X_TRAIN_PATH = PROCESSED_DATA_DIR / "lstm_x_train.npy"
LSTM_X_TEST_PATH = PROCESSED_DATA_DIR / "lstm_x_test.npy"

Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy"

LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
NORMALIZATION_PATH = MODEL_DIR / "normalization_stats.npz"

TEST_SIZE = 0.2
RANDOM_STATE = 42

AUGMENTATION_FACTOR = 3

LSTM_NOISE_STD = 0.02

MAX_LSTM_TIME_SHIFT = 10
MAX_LSTM_FEATURE_SHIFT = 3

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)


def load_feature_files() -> tuple[np.ndarray, np.ndarray]:

    required_files = [
        MFCC_FEATURES_PATH,
        LABELS_PATH
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File is not found: {file_path}"
            )

    mfcc_features = np.load(MFCC_FEATURES_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    if len(mfcc_features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    return mfcc_features, labels


def encoding_emotions(
    labels: np.ndarray
) -> tuple[np.ndarray, LabelEncoder]:

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    return encoded_labels, label_encoder


def normalize_features(
    features: np.ndarray,
    mean_value: np.ndarray,
    std_value: np.ndarray
) -> np.ndarray:

    normalized_value = (
        features - mean_value
    ) / (std_value + 1e-8)

    return normalized_value.astype(np.float32)


def split_train_data(
    lstm_features: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_train, x_test, y_train, y_test = train_test_split(
        lstm_features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    return x_train, x_test, y_train, y_test


def add_lstm_noise(mfcc_seq: np.ndarray) -> np.ndarray:

    noise = np.random.normal(
        loc=0.0,
        scale=LSTM_NOISE_STD,
        size=mfcc_seq.shape
    )

    augmented_mfcc = mfcc_seq + noise

    return augmented_mfcc.astype(np.float32)


def apply_lstm_time_shift(mfcc_seq: np.ndarray) -> np.ndarray:

    shift_value = np.random.randint(
        -MAX_LSTM_TIME_SHIFT,
        MAX_LSTM_TIME_SHIFT + 1
    )

    augmented_mfcc = np.roll(
        mfcc_seq,
        shift=shift_value,
        axis=0
    )

    return augmented_mfcc.astype(np.float32)


def apply_lstm_feature_shift(mfcc_seq: np.ndarray) -> np.ndarray:

    shift_value = np.random.randint(
        -MAX_LSTM_FEATURE_SHIFT,
        MAX_LSTM_FEATURE_SHIFT + 1
    )

    augmented_mfcc = np.roll(
        mfcc_seq,
        shift=shift_value,
        axis=1
    )

    return augmented_mfcc.astype(np.float32)


def augmented_signal_lstm_sample(
    mfcc_seq: np.ndarray
) -> np.ndarray:

    augmented_mfcc = np.copy(mfcc_seq)

    if np.random.rand() < 0.7:
        augmented_mfcc = add_lstm_noise(augmented_mfcc)

    if np.random.rand() < 0.7:
        augmented_mfcc = apply_lstm_time_shift(augmented_mfcc)

    if np.random.rand() < 0.5:
        augmented_mfcc = apply_lstm_feature_shift(augmented_mfcc)

    return augmented_mfcc.astype(np.float32)


def augment_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    augmented_features = [x_train]
    augmented_labels = [y_train]

    for _ in range(AUGMENTATION_FACTOR):

        augmented_batch = []

        for mfcc_seq in x_train:
            augmented_sample = augmented_signal_lstm_sample(
                mfcc_seq=mfcc_seq
            )

            augmented_batch.append(augmented_sample)

        augmented_features.append(
            np.array(augmented_batch, dtype=np.float32)
        )

        augmented_labels.append(y_train.copy())

    x_train_augmented = np.concatenate(
        augmented_features,
        axis=0
    )

    y_train_augmented = np.concatenate(
        augmented_labels,
        axis=0
    )

    shuffle_indexes = np.random.permutation(
        len(y_train_augmented)
    )

    x_train_augmented = x_train_augmented[shuffle_indexes]
    y_train_augmented = y_train_augmented[shuffle_indexes]

    return x_train_augmented, y_train_augmented


def save_train_data(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> None:

    np.save(LSTM_X_TRAIN_PATH, x_train)
    np.save(LSTM_X_TEST_PATH, x_test)

    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)


def save_encoder_label(label_encoder: LabelEncoder) -> None:

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)


def save_normalization_stats(
    mean_value: np.ndarray,
    std_value: np.ndarray
) -> None:

    np.savez(
        NORMALIZATION_PATH,
        mean=mean_value,
        std=std_value
    )


def main() -> None:

    print("Loading MFCC features...")

    mfcc_features, labels = load_feature_files()

    print("Encoding emotions...")

    encoded_labels, label_encoder = encoding_emotions(labels)

    print("Splitting dataset...")

    x_train, x_test, y_train, y_test = split_train_data(
        mfcc_features,
        encoded_labels
    )

    print("Normalizing features...")

    mean_value = np.mean(x_train, axis=(0, 1), keepdims=True)
    std_value = np.std(x_train, axis=(0, 1), keepdims=True)

    x_train = normalize_features(
        x_train,
        mean_value,
        std_value
    )

    x_test = normalize_features(
        x_test,
        mean_value,
        std_value
    )

    print("Augmenting training data...")

    x_train, y_train = augment_training_data(
        x_train,
        y_train
    )

    print("Saving datasets...")

    save_train_data(
        x_train,
        x_test,
        y_train,
        y_test
    )

    print("Saving label encoder...")

    save_encoder_label(label_encoder)

    save_normalization_stats(
        mean_value,
        std_value
    )

    print("\nProcessing completed successfully!")

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    print(f"Label encoder: {LABEL_ENCODER_PATH}")
    print(f"Normalization: {NORMALIZATION_PATH}")


if __name__ == "__main__":
    main()
