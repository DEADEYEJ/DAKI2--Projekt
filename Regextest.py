import re

import pandas as pd
from preprocessing2 import (
    BasePreprocessor,
    SequenceProcessor,
    RegexAdapter,
    EntitySpan,
    create_dev_test_split,
    make_cv_splits,
    flatten_label_sequences,
    evaluate_labels,
    summarize_cv_results,
    evaluate_on_test_if_enabled,
)


N_SPLITS = 9
ACTIVATE_TEST = False
USE_CLEAN_TEXT = False



PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE_NUMBER": r"(?:\+\d[\d\s\-().]{7,}\d|\b\d{8,15}\b)",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
    "CREDIT_CARD_NUMBER": r"\b(?:\d[ -]*?){13,16}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b",
    "IPV4": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "IPV6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    "PASSPORT_NUMBER": r"\b[A-Z0-9]{5,9}\b",
    "DRIVER_LICENSE_NUMBER": r"\b[A-Z0-9]{5,15}\b",
    "CUSTOMER_ID": r"\b[A-Z0-9]{4,15}\b",
    "EMPLOYEE_ID": r"\b[A-Z0-9]{4,15}\b",
    "ACCOUNT_NUMBER": r"\b\d{8,20}\b",
    "TAX_NUMBER": r"\b\d{8,15}\b",
    "PIN_NUMBER": r"\b\d{4,6}\b",
}

PRIORITY = [
    "EMAIL", "IBAN", "CREDIT_CARD_NUMBER", "SSN", "IPV4", "IPV6", "PHONE_NUMBER",
    "PASSPORT_NUMBER", "DRIVER_LICENSE_NUMBER", "CUSTOMER_ID", "EMPLOYEE_ID",
    "ACCOUNT_NUMBER", "TAX_NUMBER", "PIN_NUMBER",
]


if __name__ == "__main__":
    df = pd.read_parquet("train-00000-of-00001.parquet")

    base = BasePreprocessor(
        text_col="source_text",
        annotation_col="privacy",
        lowercase=False,
        validate_on_raw_text=True,
    )
    seq = SequenceProcessor()
    regex_adapter = RegexAdapter()

    compiled = {k: re.compile(v) for k, v in PATTERNS.items()}

    dev_df, test_df = create_dev_test_split(df, test_size=0.10, random_state=42)
    cv_folds = make_cv_splits(dev_df, n_splits=N_SPLITS, random_state=42)

    cv_results = []
    for fold_no, (train_df, val_df) in enumerate(cv_folds, start=1):
        train_samples = base.process_dataframe(train_df)
        _ = regex_adapter.to_texts(train_samples)

        val_samples = base.process_dataframe(val_df)
        true_seq = seq.add_sequence_info(val_samples, tagging_scheme="BIO", use_clean_text=USE_CLEAN_TEXT)

        pred_seq = []
        for sample, seq_sample in zip(val_samples, true_seq):
            text = sample.clean_text if USE_CLEAN_TEXT else sample.raw_text
            occupied = [False] * len(text)
            pred_entities = []

            for label in PRIORITY:
                pattern = compiled[label]
                for match in pattern.finditer(text):
                    s, e = match.start(), match.end()
                    if any(occupied[s:e]):
                        continue
                    pred_entities.append(EntitySpan(start=s, end=e, label=label, value=match.group()))
                    for i in range(s, e):
                        occupied[i] = True

            pred_labels, _ = seq.align_spans_to_tokens(
                tokens=seq_sample["tokens"],
                entities=pred_entities,
                tagging_scheme="BIO"
            )

            pred_seq.append({"token_labels": pred_labels})

        y_true = flatten_label_sequences([s["token_labels"] for s in true_seq])
        y_pred = flatten_label_sequences([s["token_labels"] for s in pred_seq])

        fold_metrics = evaluate_labels(y_true, y_pred)
        cv_results.append(fold_metrics)
        print(f"Fold {fold_no} macro-F1: {fold_metrics['macro']['f1']:.4f}")

    summary = summarize_cv_results(cv_results)
    print("\n=== CV summary ===")
    print(summary)

    if ACTIVATE_TEST:
        test_samples = base.process_dataframe(test_df)
        true_test_seq = seq.add_sequence_info(test_samples, tagging_scheme="BIO", use_clean_text=USE_CLEAN_TEXT)

        pred_test_seq = []
        for sample, seq_sample in zip(test_samples, true_test_seq):
            text = sample.clean_text if USE_CLEAN_TEXT else sample.raw_text
            occupied = [False] * len(text)
            pred_entities = []

            for label in PRIORITY:
                pattern = compiled[label]
                for match in pattern.finditer(text):
                    s, e = match.start(), match.end()
                    if any(occupied[s:e]):
                        continue
                    pred_entities.append(EntitySpan(start=s, end=e, label=label, value=match.group()))
                    for i in range(s, e):
                        occupied[i] = True

            pred_labels, _ = seq.align_spans_to_tokens(
                tokens=seq_sample["tokens"],
                entities=pred_entities,
                tagging_scheme="BIO"
            )
            pred_test_seq.append({"token_labels": pred_labels})

        y_true_test = flatten_label_sequences([s["token_labels"] for s in true_test_seq])
        y_pred_test = flatten_label_sequences([s["token_labels"] for s in pred_test_seq])

        test_metrics = evaluate_on_test_if_enabled(
            enabled=True,
            y_true_test=y_true_test,
            y_pred_test=y_pred_test,
        )

        if test_metrics is not None:
            print("\n=== Hold-out test ===")
            print(test_metrics["classification_report"])


def regex_predict_text(text):
    compiled = {k: re.compile(v) for k, v in PATTERNS.items()}

    occupied = [False] * len(text)
    pred_entities = []

    for label in PRIORITY:
        pattern = compiled[label]
        for match in pattern.finditer(text):
            s, e = match.start(), match.end()

            if any(occupied[s:e]):
                continue

            pred_entities.append({
                "label": label,
                "start": s,
                "end": e,
                "value": match.group()
            })

            for i in range(s, e):
                occupied[i] = True

    return pred_entities