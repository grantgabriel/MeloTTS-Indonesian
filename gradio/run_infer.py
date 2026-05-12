import subprocess
import os
import sys

# FIX: Fix audio path.
def run_inference(text, model_path="models/LJSpeech Models/G_1637000.pth", output_dir="output_gradio"):
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        [
            "python", "melo/infer.py",
            "--text", text,
            "-m", model_path,
            "-o", output_dir,
            "--language", "EN"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error at inference:")
        print(result.stderr)
        return None

    print("Inference success!")
    print(result.stdout)

    return output_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "Hello, this is a test inference from python script."

    run_inference(input_text)
