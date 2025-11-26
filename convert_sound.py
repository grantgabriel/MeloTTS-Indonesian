import os
import subprocess
from pathlib import Path

src_dir = Path("../datasets-indo/BAK-wavs")
dst_dir = Path("../datasets-indo/wavs")

dst_dir.mkdir(parents=True, exist_ok=True)

# Loop semua file wav
for wav_file in src_dir.glob("*.wav"):
    counter = counter + 1
    out_file = dst_dir / wav_file.name
    # Jalankan ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_file),
        "-ar", "16000",  # set sample rate
        str(out_file)
    ])
    print(f"Converted: {wav_file.name} -> {out_file}, counter = {counter}")
