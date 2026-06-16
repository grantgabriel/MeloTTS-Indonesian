import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys
import os

MODEL_ID = os.environ.get("ID_MODEL_ID", "indobenchmark/indobert-base-p2")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = None  # Lazy loading to save memory

def get_bert_feature(text, word2ph, device=None):
    """
    Extracts phone-level contextual embeddings from IndoBERT.
    Output shape aligns with MeloTTS acoustic models: [hidden_dim, N_phone]
    """
    global model

    if sys.platform == "darwin" and torch.backends.mps.is_available() and device == "cpu":
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

        # Extract the 3rd to last layer directly to avoid empty concatenation
        hidden = res["hidden_states"][-3][0].cpu()

    # Validate alignment between BERT tokens and phoneme distribution
    assert inputs["input_ids"].shape[-1] == len(word2ph), (
        f"len(word2ph)={len(word2ph)} != num_tokens={inputs['input_ids'].shape[-1]}"
    )

    phone_level_feature = []

    # Expand subword embeddings to match the number of phonemes
    for i in range(len(word2ph)):
        repeat_feature = hidden[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # Transpose for final output
    return phone_level_feature.T