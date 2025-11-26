import os
import soundfile as sf
from collections import Counter
from tqdm import tqdm

folder = "datasets/Bahasa-Kita/wavs/"

rates = []

# Ambil daftar semua file wav
files = [f for f in os.listdir(folder) if f.endswith(".wav")]

print(f"Total WAV files ditemukan: {len(files)}\n")

for file in tqdm(files, desc="Mengecek sample rate"):
    path = os.path.join(folder, file)
    try:
        info = sf.info(path)   # jauh lebih cepat
        rates.append(info.samplerate)
    except Exception as e:
        print(f"Error membaca {file}: {e}")

print("\n=== Sample Rate Count ===")
counter = Counter(rates)
for rate, count in counter.items():
    print(f"{rate} Hz : {count} files")

print("\nTotal files terbaca:", len(rates))
