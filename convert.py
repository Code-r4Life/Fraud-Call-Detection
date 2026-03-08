import os
import subprocess
import filetype

INPUT_DATASET = "Wake_Words"
OUTPUT_DATASET = "Wake_Words_processed"

TARGET_SAMPLE_RATE = "16000"


def convert_to_wav(input_path, output_path):
    """
    Converts audio to 16kHz mono WAV (ML-ready)
    """

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",              # mono
        "-ar", TARGET_SAMPLE_RATE,  # sample rate
        "-vn",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_file(input_file, output_file):
    kind = filetype.guess(input_file)

    if not kind:
        print(f"Skipping unknown file: {input_file}")
        return

    ext = kind.extension.lower()
    print(f"Processing {ext}: {input_file}")

    convert_to_wav(input_file, output_file)


for root, dirs, files in os.walk(INPUT_DATASET):

    for file in files:
        if file.endswith(".unknown") or file.lower().endswith((
            ".wav", ".mp3", ".aac", ".mp4", ".m4a", ".ogg"
        )):
            input_path = os.path.join(root, file)

            # Preserve SCAM_CALLS / NORMAL_CALLS structure
            relative_path = os.path.relpath(root, INPUT_DATASET)
            output_dir = os.path.join(OUTPUT_DATASET, relative_path)

            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.splitext(file)[0] + ".wav"
            output_path = os.path.join(output_dir, output_file)

            process_file(input_path, output_path)

print("\n✅ All files converted to ML-ready WAV format!")