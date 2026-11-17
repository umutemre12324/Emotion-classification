
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOTS_DIR = DATA_DIR / "plots"

METADATA_PATH = PROCESSED_DATA_DIR / "ravdess_metadata.csv"
MFCC_FEATURES_PATH = PROCESSED_DATA_DIR / "mfcc_features.npy"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.npy"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


SAMPLE_RATE = 22050
DURATION = 3
TARGET_LEN = int(DURATION * SAMPLE_RATE)

N_MFCC = 40
N_FFT = 2048
HOP_LEN = 512
MAX_TIME_STEPS = 130


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path)


def load_audio(file_path: str) -> np.ndarray:

    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(signal) < TARGET_LEN:
        padding_len = TARGET_LEN - len(signal)
        signal = np.pad(
            signal,
            (0, padding_len),
            mode="constant"
        )
    else:
        signal = signal[:TARGET_LEN]

    return signal


def fix_feature_len(
    feature: np.ndarray,
    max_time_step: int
) -> np.ndarray:

    current_time_step = feature.shape[1]

    if current_time_step < max_time_step:

        padding_len = max_time_step - current_time_step

        feature = np.pad(
            feature,
            ((0, 0), (0, padding_len)),
            mode="constant"
        )

    else:
        feature = feature[:, :max_time_step]

    return feature


def extract_mfcc(signal: np.ndarray) -> np.ndarray:

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LEN
    )

    mfcc = fix_feature_len(
        feature=mfcc,
        max_time_step=MAX_TIME_STEPS
    )

    return mfcc.T


def extract_features_from_dataset(
    metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:

    mfcc_features = []
    labels = []

    total_files = len(metadata)
    failed_files = 0

    for index, row in metadata.iterrows():

        file_path = row["file_path"]
        label = row["emotion"]

        try:
            signal = load_audio(file_path)

            mfcc = extract_mfcc(signal)

            mfcc_features.append(mfcc)
            labels.append(label)

        except Exception as e:
            failed_files += 1
            print(f"Error processing {file_path}: {e}")

        if (index + 1) % 100 == 0:
            print(
                f"Processed files: {index + 1}/{total_files}"
            )

    mfcc_features_array = np.array(
        mfcc_features,
        dtype=np.float32
    )

    labels_array = np.array(labels)

    print(f"\nSuccessful files: {len(labels)}")
    print(f"Failed files: {failed_files}")

    return mfcc_features_array, labels_array


def save_features(
    mfcc_features: np.ndarray,
    labels: np.ndarray
) -> None:

    np.save(MFCC_FEATURES_PATH, mfcc_features)
    np.save(LABELS_PATH, labels)

    print(f"MFCC saved: {MFCC_FEATURES_PATH}")
    print(f"Labels saved: {LABELS_PATH}")


def plot_samples_mfcc(
    mfcc_features: np.ndarray,
    labels: np.ndarray
) -> None:

    if len(mfcc_features) == 0:
        print("No MFCC features available for plotting.")
        return

    sample_index = 0
    sample_label = labels[sample_index]

    plt.figure(figsize=(10, 5))

    librosa.display.specshow(
        mfcc_features[sample_index].T,
        sr=SAMPLE_RATE,
        hop_length=HOP_LEN,
        x_axis="time"
    )

    plt.colorbar()
    plt.title(f"Sample MFCC - {sample_label}")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "plt_mfcc_sample.png",
        dpi=300
    )

    plt.close()


def main() -> None:

    print("Loading metadata...")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = load_metadata(METADATA_PATH)

    print(f"Total audio files: {len(metadata)}")

    print("\nExtracting MFCC features...")

    mfcc_features, labels = extract_features_from_dataset(metadata)

    if len(mfcc_features) == 0:
        print("No features extracted. Check audio paths and errors.")
        return

    print("\nSaving extracted features...")

    save_features(mfcc_features, labels)

    print("\nPlotting sample MFCC...")

    plot_samples_mfcc(mfcc_features, labels)

    print("\nFeature extraction completed successfully!")

    print(f"MFCC shape: {mfcc_features.shape}")
    print(f"Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
