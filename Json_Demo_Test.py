# Json Demo Test

from __future__ import annotations
 
import argparse
import json
import sys
from pathlib import Path
 
import openpyxl
 
# ── Import the pipeline functions from predict_all_models ─────────────────────
# predict_all_models.py must be on the Python path (same directory is fine).
from predict_all_models_bundle.predict_all_models import (
    DEFAULT_CRF_MODEL_PATH,
    DEFAULT_SVM_MODEL_PATH,
    build_candidates,
    build_output,
    crf_inference,
    load_pickle_model,
    load_svm_model,
    regex_inference,
    svm_inference,
)
from crf5_pipeline.crf5_dataset import normalize_text as normalize_crf_text  # noqa: E402
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
PII_CONFIDENCE_THRESHOLD = 0.5
 
 
def read_mails(excel_path: Path) -> list[str]:
    """Return the non-empty strings from the MAIL column (column A)."""
    wb = openpyxl.load_workbook(str(excel_path), read_only=True, data_only=True)
    ws = wb.active
    mails: list[str] = []
    first_row = True
    for row in ws.iter_rows(values_only=True):
        if first_row:          # skip header
            first_row = False
            continue
        cell = row[0]
        if cell is not None:
            text = str(cell).strip()
            if text:
                mails.append(text)
    wb.close()
    return mails
 
 
def high_confidence_labels(label_probs: dict[str, float], threshold: float) -> dict[str, float]:
    """Return only the labels whose probability exceeds the threshold."""
    return {
        label: round(prob, 6)
        for label, prob in label_probs.items()
        if prob > threshold
    }
 
 
def summarize_mail(
    mail_index: int,
    raw_text: str,
    output: dict,
) -> dict:
    """
    Build the per-mail result dict with the three requested sections.
    """
    models = output["summary"]["models"]
    combined = output["summary"]["combined"]
 
    # ── 1. All PII labels with confidence > 0.5 ──────────────────────────────
    pii_labels = high_confidence_labels(
        combined["label_probs"], PII_CONFIDENCE_THRESHOLD
    )
 
    # ── 2. Did each model predict PII? ────────────────────────────────────────
    #    A model "predicted PII" when its text_pii_prob > 0.5
    regex_pii_prob = models["regex"]["pii_prob"]
    crf_pii_prob   = models["crf"]["pii_prob"]
    svm_pii_prob   = models["svm"]["pii_prob"]
 
    model_predicted_pii = {
        "regex": regex_pii_prob > PII_CONFIDENCE_THRESHOLD,
        "crf":   crf_pii_prob   > PII_CONFIDENCE_THRESHOLD,
        "svm":   svm_pii_prob   > PII_CONFIDENCE_THRESHOLD,
    }
 
    # ── 3. Each model's confidence score ─────────────────────────────────────
    model_confidence_scores = {
        "regex": round(regex_pii_prob, 6),
        "crf":   round(crf_pii_prob,   6),
        "svm":   round(svm_pii_prob,   6),
    }
 
    return {
        "mail_index":             mail_index,
        "mail_preview":           raw_text[:120].replace("\n", " "),
        "pii_labels_above_0_5":   pii_labels,
        "model_predicted_pii":    model_predicted_pii,
        "model_confidence_scores": model_confidence_scores,
    }
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch PII prediction over an Excel file using predict_all_models."
    )
    parser.add_argument("--excel",          type=Path, default=Path("PII_Humble_Bundle.xlsx"))
    parser.add_argument("--output",         type=Path, default=Path("pii_results.json"))
    parser.add_argument("--crf-model-path", type=Path, default=DEFAULT_CRF_MODEL_PATH)
    parser.add_argument("--svm-model-path", type=Path, default=DEFAULT_SVM_MODEL_PATH)
    return parser.parse_args()
 
 
def main() -> int:
    args = parse_args()
 
    # Validate paths
    for label, path in (
        ("Excel file",  args.excel),
        ("CRF model",   args.crf_model_path),
        ("SVM model",   args.svm_model_path),
    ):
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1
 
    print(f"Loading CRF model from {args.crf_model_path} …")
    crf_model, crf_warnings = load_pickle_model(args.crf_model_path.resolve())
 
    print(f"Loading SVM model from {args.svm_model_path} …")
    svm_model, svm_warnings = load_svm_model(args.svm_model_path.resolve())
 
    print(f"Reading mails from {args.excel} …")
    mails = read_mails(args.excel)
    print(f"Found {len(mails)} mails.")
 
    results: list[dict] = []
 
    for i, raw_text in enumerate(mails, start=1):
        print(f"  [{i:>3}/{len(mails)}] {raw_text[:60].replace(chr(10), ' ')} …")
 
        normalized_text = normalize_crf_text(raw_text)
 
        regex_result = regex_inference(normalized_text)
        crf_result   = crf_inference(crf_model, normalized_text)
        candidates   = build_candidates(regex_result, crf_result)
        svm_result   = svm_inference(svm_model, candidates)
 
        output = build_output(
            raw_text=raw_text,
            normalized_text=normalized_text,
            regex_result=regex_result,
            crf_result=crf_result,
            svm_result=svm_result,
            candidates=candidates,
            crf_warnings=crf_warnings,
            svm_warnings=svm_warnings,
            crf_model_path=args.crf_model_path,
            svm_model_path=args.svm_model_path,
        )
 
        results.append(summarize_mail(i, raw_text, output))
 
    # Write output JSON
    output_data = {
        "meta": {
            "total_mails": len(mails),
            "pii_confidence_threshold": PII_CONFIDENCE_THRESHOLD,
            "crf_model_path": str(args.crf_model_path),
            "svm_model_path": str(args.svm_model_path),
        },
        "results": results,
    }
 
    args.output.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Results written to: {args.output}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())