#Svm Testfile

import pickle
import numpy as np

from Regex_importer import regex_score, regex_predict_text

import warnings
warnings.filterwarnings("ignore")

# Load model
with open("pkl-Filer/crf_pii_model_v5.pkl", "rb") as f:
    crf = pickle.load(f)

with open("pkl-Filer/svm_pii_model.pkl","rb") as f:
    svm = pickle.load(f)


def text_to_features(text):
    tokens = text.split()
    return [{"word": t} for t in tokens], tokens


def crf_score(model, text):
    features, tokens = text_to_features(text)

    if hasattr(model, "predict_marginals_single"):
        probs = model.predict_marginals_single(features)
        return np.mean([max(p.values()) for p in probs])

    return 0.5

def svm_score(model, text, target_label=None):

    try:

        preds = model.predict([text])

        if not preds:
            return 0.5

        pred = preds[0]

        # Expected PIIClassifier format
        if isinstance(pred, dict):

            label = pred.get("label")
            confidence = pred.get("confidence", 0.5)

            # If asking for specific label
            if target_label is not None:
                if label == target_label:
                    return float(confidence)
                return 0.0

            return float(confidence)

        # Numeric fallback
        if isinstance(pred, (int, float, np.number)):
            return float(pred)

        return 0.5

    except Exception:
        import traceback
        traceback.print_exc()
        return 0.5

def combined_score(r, c, s):
    return 0.4 * r + 0.3 * c + 0.3 * s


if __name__ == "__main__":

    text = ""

    while text != "exit":
        text = input("Enter text to analyze (or 'exit' to quit): ")
        
        r = regex_score(text)
        c = crf_score(crf, text)
        s = svm_score(svm, text)
        final = combined_score(r, c, s)

        print("\nInput:", text)
        print("Regex matches:", regex_predict_text(text))
        print(f"Regex score: {r:.3f}")
        print(f"CRF score: {c:.3f}")
        print(f"SVM score: {s:.3f}")
        print(f"Final score: {final:.3f}")

        print("PII detected" if final > 0.5 else "No PII detected")