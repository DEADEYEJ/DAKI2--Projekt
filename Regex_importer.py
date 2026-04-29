import re

# ---------------- Patterns ----------------
PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE_NUMBER": r"(?:\+\d[\d\s\-().]{7,}\d|\b\d{8,15}\b)",
    "STREET_ADDRESS": r"\b\d{1,5}\s+[A-Za-zÆØÅæøå0-9 .'-]{3,}\b",
    "COORDINATES": r"\b-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)\s*,\s*-?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d?\d(?:\.\d+)?)\b",

    "ACCOUNT_NUMBER": r"\b\d{8,20}\b",
    "BANK_ACCOUNT_NUMBER": r"\b\d{8,20}\b",
    "CREDIT_CARD_NUMBER": r"\b(?:\d[ -]*?){13,16}\b",
    "CREDIT_CARD_CVV": r"\b\d{3,4}\b",
    "PIN_NUMBER": r"\b\d{4,6}\b",
    "PASSPORT_NUMBER": r"\b[A-Z0-9]{5,9}\b",
    "DRIVER_LICENSE_NUMBER": r"\b[A-Z0-9]{5,15}\b",
    "CUSTOMER_ID": r"\b[A-Z0-9]{4,15}\b",
    "EMPLOYEE_ID": r"\b[A-Z0-9]{4,15}\b",
    "ID_CARD_NUMBER": r"\b[A-Z0-9]{5,15}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
    "SWIFT_CODE": r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
    "ROUTING_NUMBER": r"\b\d{9}\b",
    "API_KEY": r"[A-Za-z0-9]{32,40}",
    "TAX_NUMBER": r"\b\d{8,15}\b",
    #"PASSWORD": r".{8,}",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
    "IPV4": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "IPV6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
}

COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}


# ---------------- Inference function ----------------
def regex_predict_text(text):
    results = []

    for label, pattern in COMPILED.items():
        for match in pattern.finditer(text):
            results.append({
                "label": label,
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

    return results


# ---------------- Optional scoring ----------------
def regex_score(text):
    matches = regex_predict_text(text)

    if not matches:
        return 0.0

    # score based on coverage
    total_len = sum(m["end"] - m["start"] for m in matches)
    return min(total_len / len(text), 1.0)