import os
import re
from functools import cache
from transformers import AutoTokenizer
from .japanese import distribute_phone

# Default model can be overridden via environment variable
MODEL_ID = os.environ.get("ID_MODEL_ID", "indobenchmark/indobert-base-p2")

@cache
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

def text_normalize(text: str) -> str:
    """
    Normalizes Indonesian text by removing bad characters, 
    expanding common abbreviations, and converting numbers to words.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove non-phonetic characters
    bad_chars = [";", "»", "”", "“", "‘", "’", "(", ")", "[", "]"]
    for bc in bad_chars:
        text = text.replace(bc, "")

    text = text.replace("\u00a0", " ")
    text = text.replace("\u0085", " ")

    # Expand common Indonesian abbreviations
    abbreviations = {
        r"\byg\b": "yang",
        r"\bdgn\b": "dengan",
        r"\bdlm\b": "dalam",
        r"\btgl\b": "tanggal",
        r"\bkrn\b": "karena",
        r"\bdr\b": "dari"
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Convert numeric digits to Indonesian words
    try:
        from num2words import num2words
        text = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='id'), text)
    except ImportError:
        pass

    text = text.strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Ensure sentence ends with proper punctuation
    if len(text) > 0 and text[-1] not in [".", "!", "?"]:
        text = text + "."

    return text

@cache
def get_phonemizer():
    """
    Initializes espeak-ng backend for Indonesian phonemization.
    """
    import phonemizer
    from phonemizer.separator import Separator

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="id",
        preserve_punctuation=True,
        with_stress=False,
    )
    separator = Separator(phone="-", word="|")

    return global_phonemizer, separator

def g2p(text, pad_start_end: bool = True, tokenized=None):
    """
    Converts text to phonemes and aligns them with BERT tokens via word2ph.
    """
    global_phonemizer, separator = get_phonemizer()

    if tokenized is None:
        tokenizer = get_tokenizer()
        tokenized = tokenizer.tokenize(text)

    # Group subwords to align with whole words
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            # Fallback if the very first token somehow starts with '#'
            if not ph_groups:
                ph_groups.append([t.replace("#", "")])
            else:
                ph_groups[-1].append(t.replace("#", ""))

    phones = []
    tones = []
    word2ph = []

    for group in ph_groups:
        w = "".join(group)
        word_len = len(group)
        phone_len = 0

        r = global_phonemizer.phonemize(
            [w], separator=separator
        )[0].replace("|", "")
        splitted = r.split("-")

        for s in splitted:
            if len(s) == 0:
                continue
            phones.append(s)
            tones.append(0)  # Constant 0 for non-tonal language
            phone_len += 1

        # Distribute phoneme counts across the BERT subword tokens
        if word_len > 0:
            word2ph += distribute_phone(phone_len, word_len)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    try:
        from melo.text import indonesian_bert
    except ImportError:
        from melo.text import indonesian_bert

    return indonesian_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    example = "saya beli baju Rp15000 yg berwarna merah"
    norm = text_normalize(example)
    print("Normalized:", norm)
    phones, tones, word2ph = g2p(norm)
    print("Phones:", phones)
    print("Tones:", tones)
    print("word2ph:", word2ph)