import unicodedata

from jiwer import cer, wer


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def accuracy(reference: str | None, hypothesis: str):
    ref = normalize(reference or "")
    if not ref:
        return {"wer": None, "cer": None}
    hyp = normalize(hypothesis)
    return {"wer": wer(ref, hyp), "cer": cer(ref, hyp)}
