
from pathlib import Path

import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOTS_DIR = DATA_DIR / "plots"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def find_audio_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.rglob("*.wav"))


def get_emotion_label(audio_file: Path) -> str:
    filename_parts = audio_file.stem.split("-")

    if len(filename_parts) != 7:
        raise ValueError(
            f"Unexpected RAVDESS filename format: {audio_file.name}"
        )

    emotion_code = filename_parts[2]

    try:
        return EMOTION_MAP[emotion_code]
    except KeyError as error:
        raise ValueError(
            f"Unknown emotion code '{emotion_code}' in {audio_file.name}"
        ) from error


def parse_ravdess_filename(file_path: Path) -> dict:
    filename_parts = file_path.stem.split("-")

    if len(filename_parts) != 7:
        raise ValueError(
            f"Unexpected RAVDESS filename format: {file_path.name}"
        )

    modality = filename_parts[0]
    vocal_channel = filename_parts[1]
    emotion_code = filename_parts[2]
    intensity = filename_parts[3]
    statement = filename_parts[4]
    repetition = filename_parts[5]
    actor_id = filename_parts[6]

    emotion = get_emotion_label(file_path)

    return {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion_code,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor_id": actor_id
    }


def load_dataset_metadata(
    dataset_dir: Path = RAW_DATA_DIR
) -> pd.DataFrame:

    audio_files = find_audio_files(dataset_dir)
    records = []

    for audio_file in audio_files:
        try:
            record = parse_ravdess_filename(audio_file)
            records.append(record)

        except ValueError as error:
            print(f"Skipping file: {error}")

    metadata_df = pd.DataFrame(records)

    return metadata_df


def save_metadata_to_csv(
    metadata_df: pd.DataFrame,
    output_file: Path
) -> None:

    metadata_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"Metadata saved to: {output_file}")


def dataset_summary(metadata_df: pd.DataFrame) -> None:

    total_samples = len(metadata_df)
    total_classes = metadata_df["emotion"].nunique()
    total_actors = metadata_df["actor_id"].nunique()

    print("\n========== DATASET SUMMARY ==========")

    print(f"Total samples: {total_samples}")
    print(f"Total classes: {total_classes}")
    print(f"Total actors: {total_actors}")

    print("\nEmotion distribution:")

    print(
        metadata_df["emotion"]
        .value_counts()
        .to_string()
    )

    print("=====================================\n")


def plot_emotion_distribution(
    metadata_df: pd.DataFrame,
    output_file: Path
) -> None:

    emotion_counts = metadata_df["emotion"].value_counts()

    plt.figure(figsize=(10, 6))

    emotion_counts.plot(
        kind="bar",
        color="skyblue"
    )

    plt.title("Emotion Distribution in RAVDESS Dataset")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Emotion distribution plot saved to: {output_file}")


def plot_waveform(
    audio_file: Path,
    output_file: Path
) -> None:

    y, sr = librosa.load(audio_file, sr=None)

    plt.figure(figsize=(10, 4))

    librosa.display.waveshow(y, sr=sr)

    plt.title(f"Waveform of {audio_file.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Waveform plot saved to: {output_file}")


def main() -> None:

    print("Loading RAVDESS dataset...")

    dataset = load_dataset_metadata()

    if dataset.empty:
        print("No audio files found in the dataset directory.")
        return

    print(f"\nFound {len(dataset)} audio files.")

    metadata_csv_file = (
        PROCESSED_DATA_DIR / "ravdess_metadata.csv"
    )

    save_metadata_to_csv(
        dataset,
        metadata_csv_file
    )

    dataset_summary(dataset)

    emotion_distribution_plot_file = (
        PLOTS_DIR / "emotion_distribution.png"
    )

    plot_emotion_distribution(
        dataset,
        emotion_distribution_plot_file
    )

    audio_files = find_audio_files(RAW_DATA_DIR)

    sample_audio_file = audio_files[0]

    waveform_plot_file = (
        PLOTS_DIR / f"{sample_audio_file.stem}_waveform.png"
    )

    plot_waveform(
        sample_audio_file,
        waveform_plot_file
    )

    print("\nDataset processing completed successfully!")


if __name__ == "__main__":
    main()

