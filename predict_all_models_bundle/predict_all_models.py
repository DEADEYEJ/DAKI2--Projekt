from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from Regex_importer import regex_predict_text
from crf5_pipeline.crf5_dataset import SequenceSample, normalize_text as normalize_crf_text
from crf5_pipeline.crf5_dataset import sample_to_features, tokenize_with_offsets
from crf5_pipeline.crf5_metrics import bio_labels_to_entities


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CRF_MODEL_PATH = PROJECT_ROOT / "crf_pii_model_v5.pkl"
DEFAULT_SVM_SUPPORT_DIR = PROJECT_ROOT / "svm" / "SVM_supercomputer"
DEFAULT_SVM_MODEL_PATH = DEFAULT_SVM_SUPPORT_DIR / "svm_hybrid_hpc_20260511_132650" / "svm_pii_classifier.pkl"

LABEL_ALIASES = {
    "SSN": "SOCIAL_SECURITY_NUMBER",
}

REGEX_CONFIDENCE = {
    "EMAIL": 0.99,
    "PHONE_NUMBER": 0.92,
    "STREET_ADDRESS": 0.72,
    "COORDINATES": 0.98,
    "ACCOUNT_NUMBER": 0.65,
    "BANK_ACCOUNT_NUMBER": 0.78,
    "CREDIT_CARD_NUMBER": 0.97,
    "CREDIT_CARD_CVV": 0.55,
    "PIN_NUMBER": 0.50,
    "PASSPORT_NUMBER": 0.60,
    "DRIVER_LICENSE_NUMBER": 0.55,
    "CUSTOMER_ID": 0.45,
    "EMPLOYEE_ID": 0.45,
    "ID_CARD_NUMBER": 0.55,
    "IBAN": 0.99,
    "SWIFT_CODE": 0.99,
    "ROUTING_NUMBER": 0.90,
    "API_KEY": 0.75,
    "TAX_NUMBER": 0.75,
    "SOCIAL_SECURITY_NUMBER": 0.97,
    "IPV4": 0.97,
    "IPV6": 0.98,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regex, CRF and SVM in a shared pipeline over one input text."
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="Input text. If omitted, the script prompts for one text.",
    )
    parser.add_argument(
        "--crf-model-path",
        type=Path,
        default=DEFAULT_CRF_MODEL_PATH,
        help=f"Path to CRF .pkl file. Default: {DEFAULT_CRF_MODEL_PATH}",
    )
    parser.add_argument(
        "--svm-model-path",
        type=Path,
        default=DEFAULT_SVM_MODEL_PATH,
        help=f"Path to SVM .pkl file. Default: {DEFAULT_SVM_MODEL_PATH}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only.",
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def unique_warning_messages(captured_warnings: list[warnings.WarningMessage]) -> list[str]:
    seen: set[str] = set()
    unique_messages: list[str] = []
    for warning in captured_warnings:
        message = str(warning.message)
        if message not in seen:
            seen.add(message)
            unique_messages.append(message)
    return unique_messages


def load_pickle_model(model_path: Path):
    captured_warnings: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        captured_warnings.extend(caught)
    return model, unique_warning_messages(captured_warnings)


def load_svm_model(model_path: Path):
    support_dir = str(model_path.resolve().parent.parent)
    if support_dir not in sys.path:
        sys.path.insert(0, support_dir)
    return load_pickle_model(model_path)


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sorted_label_probs(label_probs: dict[str, float]) -> dict[str, float]:
    return {
        label: round(score, 6)
        for label, score in sorted(label_probs.items(), key=lambda item: (-item[1], item[0]))
        if score > 0.0
    }


def noisy_or(probabilities: list[float]) -> float:
    product = 1.0
    for probability in probabilities:
        product *= 1.0 - clamp_probability(probability)
    return clamp_probability(1.0 - product)


def merge_label_probabilities(items: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in items:
        for label, score in item.items():
            merged[label] = max(merged.get(label, 0.0), float(score))
    return sorted_label_probs(merged)


def top_label_info(label_probs: dict[str, float]) -> tuple[str, float]:
    if not label_probs:
        return "NONE", 0.0
    label, score = max(label_probs.items(), key=lambda item: item[1])
    return label, float(score)


def regex_inference(normalized_text: str) -> dict[str, Any]:
    matches = regex_predict_text(normalized_text)
    merged_spans: dict[tuple[int, int, str], dict[str, Any]] = {}
    text_label_probs: dict[str, float] = {}

    for match in matches:
        label = normalize_label(str(match["label"]))
        confidence = REGEX_CONFIDENCE.get(label, REGEX_CONFIDENCE.get(str(match["label"]), 0.5))
        key = (int(match["start"]), int(match["end"]), str(match["value"]))
        if key not in merged_spans:
            merged_spans[key] = {
                "start": key[0],
                "end": key[1],
                "value": key[2],
                "label_probs": {},
            }
        merged_spans[key]["label_probs"][label] = max(
            merged_spans[key]["label_probs"].get(label, 0.0),
            clamp_probability(confidence),
        )
        text_label_probs[label] = max(text_label_probs.get(label, 0.0), confidence)

    spans: list[dict[str, Any]] = []
    for merged in sorted(merged_spans.values(), key=lambda item: (item["start"], item["end"], item["value"])):
        label_probs = sorted_label_probs(merged["label_probs"])
        top_label = next(iter(label_probs), "UNKNOWN")
        top_prob = label_probs.get(top_label, 0.0)
        spans.append({
            "start": merged["start"],
            "end": merged["end"],
            "value": merged["value"],
            "label": top_label,
            "pii_prob": round(top_prob, 6),
            "label_probs": label_probs,
        })

    return {
        "text_pii_prob": round(max(text_label_probs.values(), default=0.0), 6),
        "label_probs": sorted_label_probs(text_label_probs),
        "spans": spans,
    }


def token_label_probabilities(marginal: dict[str, float]) -> tuple[float, dict[str, float]]:
    pii_prob = 1.0 - float(marginal.get("O", 0.0))
    label_probs: dict[str, float] = defaultdict(float)

    for raw_label, score in marginal.items():
        if raw_label == "O":
            continue
        label = raw_label.split("-", 1)[1] if "-" in raw_label else raw_label
        label_probs[normalize_label(label)] += float(score)

    return clamp_probability(pii_prob), dict(label_probs)


def crf_inference(model, normalized_text: str) -> dict[str, Any]:
    tokens = tokenize_with_offsets(normalized_text)
    if not tokens:
        return {"text_pii_prob": 0.0, "label_probs": {}, "spans": [], "tokens": []}

    sample = SequenceSample(
        sample_id="inference",
        raw_text=normalized_text,
        sequence_text=normalized_text,
        tokens=tokens,
    )
    features = sample_to_features(sample)
    predicted_labels = model.predict_single(features)
    marginals = model.predict_marginals_single(features)
    entities = bio_labels_to_entities(tokens, predicted_labels, normalized_text)

    token_infos: list[dict[str, Any]] = []
    text_label_probs: dict[str, float] = {}

    for token, label, marginal in zip(tokens, predicted_labels, marginals):
        pii_prob, label_probs = token_label_probabilities(marginal)
        for base_label, score in label_probs.items():
            text_label_probs[base_label] = max(text_label_probs.get(base_label, 0.0), score)

        token_infos.append({
            "token": token.text,
            "start": token.start,
            "end": token.end,
            "predicted_label": normalize_label(label.split("-", 1)[1]) if label != "O" and "-" in label else label,
            "pii_prob": round(pii_prob, 6),
            "label_probs": sorted_label_probs(label_probs),
        })

    spans: list[dict[str, Any]] = []
    for entity in entities:
        covered_tokens = [
            token_info
            for token_info in token_infos
            if token_info["start"] >= entity.start and token_info["end"] <= entity.end
        ]
        span_label_probs = merge_label_probabilities([token_info["label_probs"] for token_info in covered_tokens])
        span_pii_prob = (
            sum(token_info["pii_prob"] for token_info in covered_tokens) / len(covered_tokens)
            if covered_tokens
            else 0.0
        )
        spans.append({
            "start": entity.start,
            "end": entity.end,
            "value": entity.value if entity.value is not None else normalized_text[entity.start:entity.end],
            "label": normalize_label(entity.label),
            "pii_prob": round(span_pii_prob, 6),
            "label_probs": span_label_probs,
        })

    return {
        "text_pii_prob": round(max((token["pii_prob"] for token in token_infos), default=0.0), 6),
        "label_probs": sorted_label_probs(text_label_probs),
        "spans": spans,
        "tokens": token_infos,
    }


def candidate_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["start"]), int(span["end"]), str(span["value"])


def build_candidates(regex_result: dict[str, Any], crf_result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[tuple[int, int, str], dict[str, Any]] = {}

    for source_name, source_result in (("regex", regex_result), ("crf", crf_result)):
        for span in source_result["spans"]:
            key = candidate_key(span)
            if key not in candidates:
                candidates[key] = {
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "value": str(span["value"]),
                    "proposed_by": [],
                    "regex": None,
                    "crf": None,
                    "svm": None,
                }
            if source_name not in candidates[key]["proposed_by"]:
                candidates[key]["proposed_by"].append(source_name)
            candidates[key][source_name] = span

    return sorted(candidates.values(), key=lambda item: (item["start"], item["end"], item["value"]))


def svm_inference(model, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"text_pii_prob": 0.0, "label_probs": {}, "spans": []}

    values = [candidate["value"] for candidate in candidates]
    predictions = model.predict(values)

    spans: list[dict[str, Any]] = []
    text_label_probs: dict[str, float] = {}
    text_pii_prob = 0.0

    for candidate, prediction in zip(candidates, predictions):
        raw_probs = dict(prediction.get("probabilities", {}))
        non_pii_prob = float(raw_probs.get("NON_PII", 0.0))
        pii_prob = clamp_probability(1.0 - non_pii_prob)

        label_probs = {
            normalize_label(label): float(score)
            for label, score in raw_probs.items()
            if label != "NON_PII"
        }
        label_probs = sorted_label_probs(label_probs)

        for label, score in label_probs.items():
            text_label_probs[label] = max(text_label_probs.get(label, 0.0), score)
        text_pii_prob = max(text_pii_prob, pii_prob)

        svm_span = {
            "start": candidate["start"],
            "end": candidate["end"],
            "value": candidate["value"],
            "label": normalize_label(str(prediction.get("label", "UNKNOWN"))),
            "pii_prob": round(pii_prob, 6),
            "label_probs": label_probs,
            "non_pii_prob": round(non_pii_prob, 6),
        }
        candidate["svm"] = svm_span
        spans.append(svm_span)

    return {
        "text_pii_prob": round(text_pii_prob, 6),
        "label_probs": sorted_label_probs(text_label_probs),
        "spans": spans,
    }


def normalize_inputs(texts: list[str]) -> str:
    if texts:
        return " ".join(texts).strip()
    return input("Enter text to analyze: ").strip()


def build_output(
    raw_text: str,
    normalized_text: str,
    regex_result: dict[str, Any],
    crf_result: dict[str, Any],
    svm_result: dict[str, Any],
    candidates: list[dict[str, Any]],
    crf_warnings: list[str],
    svm_warnings: list[str],
    crf_model_path: Path,
    svm_model_path: Path,
) -> dict[str, Any]:
    model_summaries = {}
    combined_label_inputs: dict[str, list[float]] = defaultdict(list)

    for model_name, model_result in (
        ("regex", regex_result),
        ("crf", crf_result),
        ("svm", svm_result),
    ):
        top_label, top_score = top_label_info(model_result["label_probs"])
        model_summaries[model_name] = {
            "pii_prob": round(float(model_result["text_pii_prob"]), 6),
            "top_label": top_label,
            "top_label_prob": round(top_score, 6),
        }
        for label, score in model_result["label_probs"].items():
            combined_label_inputs[label].append(float(score))

    combined_label_probs = sorted_label_probs({
        label: noisy_or(scores)
        for label, scores in combined_label_inputs.items()
    })
    combined_top_label, combined_top_score = top_label_info(combined_label_probs)
    combined_pii_prob = noisy_or([
        float(regex_result["text_pii_prob"]),
        float(crf_result["text_pii_prob"]),
        float(svm_result["text_pii_prob"]),
    ])

    return {
        "input_text": raw_text,
        "normalized_text": normalized_text,
        "model_paths": {
            "crf": str(crf_model_path),
            "svm": str(svm_model_path),
        },
        "warnings": {
            "crf": crf_warnings,
            "svm": svm_warnings,
        },
        "summary": {
            "models": model_summaries,
            "combined": {
                "pii_prob": round(combined_pii_prob, 6),
                "top_label": combined_top_label,
                "top_label_prob": round(combined_top_score, 6),
                "label_probs": combined_label_probs,
            },
        },
        "models": {
            "regex": regex_result,
            "crf": {
                "text_pii_prob": crf_result["text_pii_prob"],
                "label_probs": crf_result["label_probs"],
                "spans": crf_result["spans"],
            },
            "svm": svm_result,
        },
        "candidates": candidates,
    }


def print_summary(output: dict[str, Any]) -> None:
    print(f"Input: {output['input_text']}")

    for model_name in ("regex", "crf", "svm"):
        model_summary = output["summary"]["models"][model_name]
        print(
            f"{model_name.upper()}: "
            f"PII={model_summary['pii_prob']:.4f}, "
            f"top_label={model_summary['top_label']}, "
            f"label_prob={model_summary['top_label_prob']:.4f}"
        )

    combined = output["summary"]["combined"]
    print(
        f"COMBINED: "
        f"PII={combined['pii_prob']:.4f}, "
        f"top_label={combined['top_label']}, "
        f"label_prob={combined['top_label_prob']:.4f}"
    )

    if output["warnings"]["svm"]:
        print("Note: SVM model was loaded with sklearn version warnings.")


def main() -> int:
    args = parse_args()
    raw_text = normalize_inputs(args.texts)
    if not raw_text:
        print("No input text provided.", file=sys.stderr)
        return 1

    crf_model_path = args.crf_model_path.resolve()
    svm_model_path = args.svm_model_path.resolve()

    if not crf_model_path.exists():
        print(f"CRF model file not found: {crf_model_path}", file=sys.stderr)
        return 1
    if not svm_model_path.exists():
        print(f"SVM model file not found: {svm_model_path}", file=sys.stderr)
        return 1

    normalized_text = normalize_crf_text(raw_text)
    crf_model, crf_warnings = load_pickle_model(crf_model_path)
    svm_model, svm_warnings = load_svm_model(svm_model_path)

    regex_result = regex_inference(normalized_text)
    crf_result = crf_inference(crf_model, normalized_text)
    candidates = build_candidates(regex_result, crf_result)
    svm_result = svm_inference(svm_model, candidates)

    output = build_output(
        raw_text=raw_text,
        normalized_text=normalized_text,
        regex_result=regex_result,
        crf_result=crf_result,
        svm_result=svm_result,
        candidates=candidates,
        crf_warnings=crf_warnings,
        svm_warnings=svm_warnings,
        crf_model_path=crf_model_path,
        svm_model_path=svm_model_path,
    )

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    print_summary(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
