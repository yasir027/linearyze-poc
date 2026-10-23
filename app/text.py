"""Text helpers shared by ingestion and search: Arabic-aware normalisation and fixed user-facing messages."""
import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟ]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TOKEN = re.compile(r"[a-z0-9]+|[ء-ي]+")


def normalize(text: str) -> str:
    """Remove Arabic diacritics and tatweel, unify common letter variants, map Arabic-Indic digits, lowercase."""
    text = _DIACRITICS.sub("", text).replace("ـ", "")
    return text.translate(_ALEF).translate(_ARABIC_INDIC).lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum("؀" <= c <= "ۿ" for c in letters)
    return arabic / len(letters) > 0.5


# Wording taken from the SRS (sections 0.10.2, 2.9 and 0.10.7).
MESSAGES = {
    "M1": {
        "en": "I could not find a document matching your request within your access.",
        "ar": "لم أتمكن من العثور على مستند مطابق لطلبك ضمن صلاحيات الوصول المتاحة لك.",
    },
    "M10007": {
        "en": "I cannot process payments or financial transactions. This system covers operations and revenue analysis only.",
        "ar": "لا يمكنني تنفيذ المدفوعات أو المعاملات المالية. يقتصر هذا النظام على العمليات وتحليل الإيرادات.",
    },
}
