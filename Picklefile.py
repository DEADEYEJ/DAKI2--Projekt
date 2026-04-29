import sys
import pickle
import numpy as np

from Regex_importer import regex_score, regex_predict_text

# Load model
with open("crf_pii_model.pkl", "rb") as f:
    crf = pickle.load(f)

with open("svm_pii_classifier.pkl","rb") as f:
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

def svm_score(model, text):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([text])[0]
        return max(prob)

    if hasattr(model, "decision_function"):
        score = model.decision_function([text])[0]
        return 1 / (1 + np.exp(-score))

    return 0.5

def combined_score(r, c, s):
    return 0.4 * r + 0.4 * c + 0.2 * s


if __name__ == "__main__":
    if len(sys.argv) < 2:
        text = input("Enter text to analyze: ")
    else:
        text = sys.argv[1]

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