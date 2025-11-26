import subprocess
import os
import sys

def run_inference(text, model_path="melo/logs/LJSpeech-1.1/testing/G_644000.pth", output_dir="output_gradio"):
    # pastikan folder output ada
    os.makedirs(output_dir, exist_ok=True)

    # panggil infer.py via subprocess
    result = subprocess.run(
        [
            "python3", "melo/infer.py",
            "--text", text,
            "-m", model_path,
            "-o", output_dir
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Error saat inference:")
        print(result.stderr)
        return None

    print("✅ Inference sukses!")
    print(result.stdout)

    return output_dir


if __name__ == "__main__":
    # kalau mau jalanin langsung dari terminal
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "hello, this is a test inference from python script. I love patricia indry ely"

    run_inference(input_text)
