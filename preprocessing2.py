from __future__ import annotations

import re
import html
import json
import ast
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report


# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# Dataclasses
# =========================================================
@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    value: Optional[str] = None


@dataclass
class ProcessedSample:
    sample_id: Any
    raw_text: str
    clean_text: str
    entities: List[EntitySpan] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class Token:
    text: str
    start: int
    end: int


# =========================================================
# 1. Base preprocessing
# =========================================================
class BasePreprocessor:
    def __init__(
        self,
        text_col: str = "source_text",
        annotation_col: str = "privacy",
        lowercase: bool = False,
        strip_html: bool = True,
        normalize_special_chars: bool = True,
        normalize_whitespace: bool = True,
        validate_on_raw_text: bool = True,
    ):
        self.text_col = text_col
        self.annotation_col = annotation_col
        self.lowercase = lowercase
        self.strip_html = strip_html
        self.normalize_special_chars = normalize_special_chars
        self.normalize_whitespace = normalize_whitespace
        self.validate_on_raw_text = validate_on_raw_text

    def normalize_text(self, text: Any) -> str:
        if pd.isna(text):
            return ""

        text = str(text)
        text = html.unescape(text)

        if self.strip_html:
            text = re.sub(r"<[^>]+>", " ", text)

        if self.normalize_special_chars:
            replacements = {
                "\u00A0": " ",
                "\u200B": "",
                "\u2013": "-",
                "\u2014": "-",
                "\u2018": "'",
                "\u2019": "'",
                "\u201C": '"',
                "\u201D": '"',
                "\t": " ",
                "\r": " ",
                "\n": " ",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)

        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self.lowercase:
            text = text.lower()

        return text

    def find_value_span_in_text(self, text: str, value: str) -> Optional[Tuple[int, int]]:
        """
        Finder første eksakte forekomst af value i raw text.
        Returnerer (start, end) eller None.
        """
        if not text or not value:
            return None

        start = text.find(value)
        if start == -1:
            return None

        end = start + len(value)
        return start, end

    def parse_annotation_object(
        self,
        item: dict,
        raw_text: str,
        idx: int
    ) -> Tuple[Optional[EntitySpan], List[str]]:
        issues = []

        label = item.get("label")
        start = item.get("start")
        end = item.get("end")
        value = item.get("value")

        if start is None:
            start = item.get("begin_offset") or item.get("offset_start")
        if end is None:
            end = item.get("stop") or item.get("offset_end")

        if label is None:
            issues.append(f"annotation_item_{idx}_missing_label")
            return None, issues

        # Hvis spans mangler, prøv at finde dem ud fra value i raw_text
        if (start is None or end is None) and value is not None:
            inferred = self.find_value_span_in_text(raw_text, str(value))
            if inferred is not None:
                start, end = inferred
                issues.append(f"annotation_item_{idx}_span_inferred_from_value")

        if start is None or end is None:
            issues.append(f"annotation_item_{idx}_missing_span")
            return None, issues

        try:
            start = int(start)
            end = int(end)
        except Exception:
            issues.append(f"annotation_item_{idx}_invalid_span_type")
            return None, issues

        if start < 0 or end <= start:
            issues.append(f"annotation_item_{idx}_invalid_span_range")
            return None, issues

        return EntitySpan(
            start=start,
            end=end,
            label=str(label),
            value=value
        ), issues

    def parse_annotations(
        self,
        annotation_obj: Any,
        raw_text: str
    ) -> Tuple[List[EntitySpan], List[str]]:
        issues = []

        if annotation_obj is None or (isinstance(annotation_obj, float) and pd.isna(annotation_obj)):
            return [], ["missing_annotation"]

        parsed = None

        if isinstance(annotation_obj, (list, tuple)):
            parsed = list(annotation_obj)

        elif hasattr(annotation_obj, "tolist") and not isinstance(annotation_obj, str):
            try:
                parsed = annotation_obj.tolist()
            except Exception:
                return [], ["unsupported_annotation_type"]

        elif isinstance(annotation_obj, str):
            annotation_obj = annotation_obj.strip()

            if not annotation_obj:
                return [], ["empty_annotation"]

            try:
                parsed = json.loads(annotation_obj)
            except Exception:
                try:
                    parsed = ast.literal_eval(annotation_obj)
                except Exception:
                    return [], ["malformed_annotation"]
        else:
            return [], ["unsupported_annotation_type"]

        if not isinstance(parsed, list):
            return [], ["annotation_not_list"]

        entities = []

        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                issues.append(f"annotation_item_{idx}_not_dict")
                continue

            entity, item_issues = self.parse_annotation_object(
                item=item,
                raw_text=raw_text,
                idx=idx
            )
            issues.extend(item_issues)

            if entity is not None:
                entities.append(entity)

        return entities, issues

    def validate_spans(
        self,
        text_for_validation: str,
        entities: List[EntitySpan]
    ) -> Tuple[List[EntitySpan], List[str]]:
        issues = []
        valid_entities = []

        for i, ent in enumerate(entities):
            if ent.end > len(text_for_validation):
                issues.append(f"entity_{i}_span_out_of_bounds")
                continue

            extracted = text_for_validation[ent.start:ent.end]

            if ent.value is not None:
                clean_value = self.normalize_text(ent.value)
                clean_extracted = self.normalize_text(extracted)

                if clean_value and clean_value != clean_extracted:
                    issues.append(f"entity_{i}_value_text_mismatch")

            valid_entities.append(ent)

        return valid_entities, issues

    def process_sample(self, sample_id: Any, raw_text: Any, raw_annotations: Any) -> ProcessedSample:
        issues = []

        raw_text_str = "" if pd.isna(raw_text) else str(raw_text)
        clean_text = self.normalize_text(raw_text_str)

        entities, parse_issues = self.parse_annotations(raw_annotations, raw_text_str)
        issues.extend(parse_issues)

        validation_text = raw_text_str if self.validate_on_raw_text else clean_text
        entities, span_issues = self.validate_spans(validation_text, entities)
        issues.extend(span_issues)

        return ProcessedSample(
            sample_id=sample_id,
            raw_text=raw_text_str,
            clean_text=clean_text,
            entities=entities,
            issues=issues
        )

    def process_dataframe(self, df: pd.DataFrame, id_col: Optional[str] = None) -> List[ProcessedSample]:
        samples = []

        for idx, row in df.iterrows():
            sample_id = row[id_col] if id_col and id_col in df.columns else idx
            sample = self.process_sample(
                sample_id=sample_id,
                raw_text=row.get(self.text_col, ""),
                raw_annotations=row.get(self.annotation_col, None)
            )
            samples.append(sample)

        return samples

    @staticmethod
    def summarize_issues(samples: List[ProcessedSample]) -> Dict[str, int]:
        counter = Counter()
        for sample in samples:
            counter.update(sample.issues)
        return dict(counter)


# =========================================================
# 2. Sequence processing
# =========================================================
class SequenceProcessor:
    def tokenize_with_offsets(self, text: str) -> List[Token]:
        pattern = r"\w+|[^\w\s]"
        tokens = []

        for match in re.finditer(pattern, text, flags=re.UNICODE):
            tokens.append(Token(
                text=match.group(),
                start=match.start(),
                end=match.end()
            ))

        return tokens

    def align_spans_to_tokens(
        self,
        tokens: List[Token],
        entities: List[EntitySpan],
        tagging_scheme: str = "BIO"
    ) -> Tuple[List[str], List[str]]:
        issues = []
        labels = ["O"] * len(tokens)

        for ent_idx, ent in enumerate(entities):
            overlapping_token_idxs = []

            for i, tok in enumerate(tokens):
                overlap = tok.start < ent.end and tok.end > ent.start
                if overlap:
                    overlapping_token_idxs.append(i)

            if not overlapping_token_idxs:
                issues.append(f"entity_{ent_idx}_no_token_overlap")
                continue

            for token_i in overlapping_token_idxs:
                tok = tokens[token_i]
                if not (tok.start >= ent.start and tok.end <= ent.end):
                    issues.append(f"entity_{ent_idx}_partial_token_overlap")

            if tagging_scheme.upper() == "BIO":
                for j, token_i in enumerate(overlapping_token_idxs):
                    prefix = "B-" if j == 0 else "I-"
                    new_label = f"{prefix}{ent.label}"

                    if labels[token_i] != "O":
                        issues.append(f"token_{token_i}_overlapping_entities")

                    labels[token_i] = new_label
            else:
                for token_i in overlapping_token_idxs:
                    if labels[token_i] != "O":
                        issues.append(f"token_{token_i}_overlapping_entities")
                    labels[token_i] = ent.label

        return labels, issues

    def add_sequence_info(
        self,
        samples: List[ProcessedSample],
        tagging_scheme: str = "BIO",
        use_clean_text: bool = False
    ) -> List[Dict[str, Any]]:
        enriched_samples = []

        for sample in samples:
            text_for_sequence = sample.clean_text if use_clean_text else sample.raw_text

            tokens = self.tokenize_with_offsets(text_for_sequence)
            token_labels, alignment_issues = self.align_spans_to_tokens(
                tokens=tokens,
                entities=sample.entities,
                tagging_scheme=tagging_scheme
            )

            enriched_samples.append({
                "sample_id": sample.sample_id,
                "raw_text": sample.raw_text,
                "clean_text": sample.clean_text,
                "sequence_text": text_for_sequence,
                "entities": sample.entities,
                "issues": sample.issues + alignment_issues,
                "tokens": tokens,
                "token_labels": token_labels,
            })

        return enriched_samples


# =========================================================
# 3. Adapters
# =========================================================
class CRFAdapter:
    def to_features(self, sequence_samples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        X = []

        for sample in sequence_samples:
            tokens = sample["tokens"]
            sentence_features = []

            for i, tok in enumerate(tokens):
                feats = {
                    "token": tok.text,
                    "lower": tok.text.lower(),
                    "is_upper": tok.text.isupper(),
                    "is_title": tok.text.istitle(),
                    "is_digit": tok.text.isdigit(),
                    "prefix_2": tok.text[:2],
                    "suffix_2": tok.text[-2:],
                    "prev_token": tokens[i - 1].text if i > 0 else "<START>",
                    "next_token": tokens[i + 1].text if i < len(tokens) - 1 else "<END>",
                }
                sentence_features.append(feats)

            X.append(sentence_features)

        return X

    def to_labels(self, sequence_samples: List[Dict[str, Any]]) -> List[List[str]]:
        return [sample["token_labels"] for sample in sequence_samples]


class SVMAdapter:
    def to_texts(self, samples: List[ProcessedSample]) -> List[str]:
        return [sample.clean_text for sample in samples]


class RegexAdapter:
    def to_texts(self, samples: List[ProcessedSample]) -> List[str]:
        return [sample.clean_text for sample in samples]


# =========================================================
# 4. Split: hold-out test + cross-validation
# =========================================================
def create_dev_test_split(
    df: pd.DataFrame,
    test_size: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dev_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    return dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_cv_splits(
    dev_df: pd.DataFrame,
    n_splits: int = 9,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = []
    for train_idx, val_idx in kf.split(dev_df):
        train_fold = dev_df.iloc[train_idx].reset_index(drop=True)
        val_fold = dev_df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_fold, val_fold))

    return folds


# =========================================================
# 5. Evaluation
# =========================================================
def flatten_label_sequences(label_sequences: List[List[str]]) -> List[str]:
    return [label for seq in label_sequences for label in seq]


def evaluate_labels(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    eval_labels = [lbl for lbl in labels if lbl != "O"]

    if len(eval_labels) == 0:
        return {
            "per_label": {},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "classification_report": "No entity labels found in y_true/y_pred for this evaluation run."
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=eval_labels,
        average=None,
        zero_division=0
    )

    per_label = {}
    for lbl, p, r, f, s in zip(eval_labels, precision, recall, f1, support):
        per_label[lbl] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s)
        }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="micro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="weighted", zero_division=0
    )

    return {
        "per_label": per_label,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1)
        },
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1)
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1)
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=eval_labels,
            zero_division=0
        )
    }


def summarize_cv_results(cv_results: List[Dict[str, Any]]) -> Dict[str, float]:
    macro_f1s = [fold["macro"]["f1"] for fold in cv_results]
    macro_ps = [fold["macro"]["precision"] for fold in cv_results]
    macro_rs = [fold["macro"]["recall"] for fold in cv_results]

    return {
        "macro_precision_mean": sum(macro_ps) / len(macro_ps),
        "macro_recall_mean": sum(macro_rs) / len(macro_rs),
        "macro_f1_mean": sum(macro_f1s) / len(macro_f1s),
        "n_folds": len(cv_results)
    }


# =========================================================
# 6. Test activation
# =========================================================
def evaluate_on_test_if_enabled(
    enabled: bool,
    y_true_test: List[str],
    y_pred_test: List[str]
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    return evaluate_labels(y_true_test, y_pred_test)