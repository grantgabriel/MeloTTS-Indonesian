import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys
import os

# IndoBERT base; bisa override dengan env ID_MODEL_ID kalau mau ganti
MODEL_ID = os.environ.get("ID_MODEL_ID", "indobenchmark/indobert-base-p2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = None  # lazy load supaya nggak langsung makan RAM


def get_bert_feature(text, word2ph, device=None):
    """
    Menghasilkan embedding BERT level-phone, sama persis formatnya dengan malay_bert:

    - Input:
        text: string yang sudah dinormalisasi
        word2ph: list int (setelah DIMODIFIKASI oleh clean_text_bert, jadi panjang = jumlah token BERT)
    - Output:
        Tensor shape [hidden_dim, N_phone], siap dipakai MeloTTS.
    """
    global model

    # Device handling sama seperti malay_bert
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"

    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).to(device)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        res = model(**inputs, output_hidden_states=True)

        # Ambil 1 layer terakhir (atau bisa diganti kombinasi beberapa layer)
        # res["hidden_states"]: list [layer0, layer1, ..., layerL] dengan shape [1, T, H]
        hidden = torch.cat(res["hidden_states"][-3:-2], dim=-1)[0].cpu()
        # hidden shape: [T, hidden_dim]

    # Wajib: jumlah token BERT == len(word2ph) setelah hack di clean_text_bert
    assert inputs["input_ids"].shape[-1] == len(word2ph), (
        f"len(word2ph)={len(word2ph)} "
        f"!= num_tokens={inputs['input_ids'].shape[-1]}"
    )

    word2phone = word2ph
    phone_level_feature = []

    # Expand embedding token BERT → level-phone
    for i in range(len(word2phone)):
        # hidden[i]: [hidden_dim]
        # repeat sesuai jumlah phone untuk token ini: [word2phone[i], hidden_dim]
        repeat_feature = hidden[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # Concatenate semua → [N_phone, hidden_dim]
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # Transpose ke [hidden_dim, N_phone], seperti malay_bert
    return phone_level_feature.T
