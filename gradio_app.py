import gradio as gr
import subprocess
import os

AUDIO_PATH = "output_gradio/LJSpeech/output.wav"

def infer_and_play(text):
    print(f"[UI Input] {text}")

    os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)

    result = subprocess.run(
        ["python3", "run_infer.py", text],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Error saat inference:")
        print(result.stderr)
        return None, "❌ Inference gagal!"

    print("✅ Inference sukses!")
    print(result.stdout)

    if os.path.exists(AUDIO_PATH):
        return AUDIO_PATH, f"✅ Teks diproses: {text}"
    else:
        return None, "❌ File audio tidak ditemukan!"

with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ MeloTTS Demo - PT Bahasa Kinerja Utama")

    text_input = gr.Textbox(label="Input Text", placeholder="Tulis teks di sini...", lines=2)
    btn = gr.Button("Submit")

    audio_output = gr.Audio(label="Generated Audio", type="filepath")
    status_output = gr.Textbox(label="Status", interactive=False)

    btn.click(fn=infer_and_play, inputs=text_input, outputs=[audio_output, status_output])

if __name__ == "__main__":
    demo.launch(share=True)
