import gradio as gr

def echo(text):
    print(f"[DEBUG] input: {text}")  # tampil di terminal
    return f"Kamu mengetik: {text}"

with gr.Blocks() as demo:
    gr.Markdown("# 🔎 Test Gradio UI Sederhana")
    inp = gr.Textbox(label="Masukkan teks")
    out = gr.Textbox(label="Output")
    btn = gr.Button("Submit")

    btn.click(fn=echo, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)   # pakai share biar tidak kena localhost error
