import os
from functools import cache
from transformers import AutoTokenizer
from .japanese import distribute_phone  # sama seperti malay.py

# Bisa dioverride dari environment kalau mau ganti model
MODEL_ID = os.environ.get("ID_MODEL_ID", "indobenchmark/indobert-base-p2")


@cache
def get_tokenizer():
    """
    Tokenizer IndoBERT, digunakan baik untuk g2p (untuk ph_groups/word2ph)
    maupun oleh indonesia_bert.py (harus konsisten).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return tokenizer

def text_normalize(text: str) -> str:
    """
    Normalisasi sederhana untuk Bahasa Indonesia.
    Membersihkan karakter aneh seperti ; » ” yang tidak punya nilai fonetis.
    """

    import re

    if not isinstance(text, str):
        text = str(text)

    # Bersihkan karakter berbahaya / tidak diinginkan
    # -----------------------------------------------------------
    # Karakter aneh yang ingin kita hilangkan:
    # ; » ” “ ‘ ’
    bad_chars = [";", "»", "”", "“", "‘", "’", "(", ")", "[", "]",]
    for bc in bad_chars:
        text = text.replace(bc, "")

    # Tambahan: beberapa dataset punya kombinasi UTF-8 buruk seperti \x85, \xa0
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\u0085", " ")  # NEXT LINE char, sering muncul di OCR

    # Hilangkan leading/trailing whitespace
    text = text.strip()

    # Ganti newline/tab dengan spasi
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Ratakan spasi ganda
    text = re.sub(r"\s+", " ", text)

    # Tambah titik di akhir kalau belum ada tanda akhir kalimat
    if len(text) > 0 and text[-1] not in [".", "!", "?"]:
        text = text + "."

    return text

@cache
def get_phonemizer():
    """
    Phonemizer untuk Bahasa Indonesia.
    Menggunakan espeak-ng backend dengan language='id'.

    - preserve_punctuation=True → tanda baca tetap keluar sebagai fonem
    - with_stress=False → kita TIDAK ambil informasi stress sama sekali
    """
    import phonemizer
    from phonemizer.separator import Separator

    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="id",
        preserve_punctuation=True,
        with_stress=False,
    )
    # phone='-' supaya bisa split per fonem, word='|' kalau suatu saat mau multi-kata
    separator = Separator(phone="-", word="|")

    return global_phonemizer, separator


def g2p(text, pad_start_end: bool = True, tokenized=None):
    """
    Grapheme-to-phoneme Bahasa Indonesia.
    Menghasilkan (phones, tones, word2ph) dengan format yang kompatibel MeloTTS.

    - phones: list string fonem, misal ["_", "s", "a", "j", "a", ".", "_"]
    - tones: list int, sama panjang dengan phones, SEMUA 0 (tanpa stress)
    - word2ph: list int, jumlah phone per token BERT (setelah grouping subword)
    """
    global_phonemizer, separator = get_phonemizer()

    # Tokenisasi pakai IndoBERT, sama seperti malay.py
    if tokenized is None:
        tokenizer = get_tokenizer()
        tokenized = tokenizer.tokenize(text)

    # Kelompokkan subword: token yang tidak diawali '##' mulai group baru,
    # sisanya menempel ke group terakhir (dengan '##' dihapus)
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            # buang '#' agar jadi bentuk kata utuh
            ph_groups[-1].append(t.replace("#", ""))

    phones = []
    tones = []
    word2ph = []

    for group in ph_groups:
        # gabungkan subword jadi string kata
        w = "".join(group)
        word_len = len(group)
        phone_len = 0

        # phonemize satu kata
        # hasil contoh: "s-a-j-a." atau "s-a-j-a-."
        r = global_phonemizer.phonemize(
            [w], separator=separator
        )[0].replace("|", "")
        splitted = r.split("-")

        for s in splitted:
            if len(s) == 0:
                continue
            # Tambah fonem
            phones.append(s)
            # TONES: untuk Bahasa Indonesia kita pakai 0 semua (no stress)
            tones.append(0)
            phone_len += 1

        # Distribusikan jumlah fonem ke subword-token BERT dari kata ini
        # contoh: phone_len=4, word_len=2 → mungkin [2,2]
        if word_len > 0:
            aaa = distribute_phone(phone_len, word_len)
            word2ph += aaa

    if pad_start_end:
        # Sentinel '_' di awal & akhir, seperti English/Malay
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        # word2ph juga diberi 1 di awal & akhir
        word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph


def get_bert_feature(text, word2ph, device=None):
    """
    Wrapper ke indonesia_bert.get_bert_feature agar dipanggil clean_text_bert.
    """
    try:
        from text import indonesia_bert
    except ImportError:
        from melo.text import indonesia_bert

    return indonesia_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    # Tes kecil manual
    example = "saya suka baju berwarna merah tua"
    norm = text_normalize(example)
    print("Normalized:", norm)
    phones, tones, word2ph = g2p(norm)
    print("Phones:", phones)
    print("Tones:", tones)
    print("word2ph:", word2ph)
