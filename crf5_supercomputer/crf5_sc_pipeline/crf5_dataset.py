from __future__ import annotations

import html
import os
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .crf5_config import ANNOTATION_COL, TEXT_COL

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is only for nicer progress output.
    tqdm = None


@dataclass(slots=True)
class EntitySpan:
    start: int
    end: int
    label: str
    value: Optional[str] = None


@dataclass(slots=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(slots=True)
class TokenProfile:
    text: str
    lower: str
    prefix_1: str
    prefix_2: str
    prefix_3: str
    prefix_4: str
    suffix_1: str
    suffix_2: str
    suffix_3: str
    suffix_4: str
    shape: str
    compressed_shape: str
    is_upper: bool
    is_title: bool
    is_digit: bool
    is_alpha: bool
    is_alnum: bool
    is_punct: bool
    has_alpha: bool
    has_digit: bool
    has_hyphen: bool
    has_slash: bool
    has_dot: bool
    has_at: bool
    has_colon: bool
    has_plus: bool
    has_underscore: bool
    alpha_digit_mix: bool
    digit_count: int
    length: int
    length_bucket: int


@dataclass(slots=True)
class SequenceSample:
    sample_id: Any
    raw_text: str
    sequence_text: str
    entities: List[EntitySpan] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return bool(pd.isna(value))
    return False


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)

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

    return re.sub(r"\s+", " ", text).strip()


def normalize_privacy_items(value: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if is_missing(value):
        return [], ["missing_annotation"]

    items = value.tolist() if hasattr(value, "tolist") else value

    if not items:
        return [], ["empty_annotation"]
    if isinstance(items, tuple):
        items = list(items)
    if not isinstance(items, list):
        return [], ["unsupported_annotation_type"]

    return items, []


def find_all_occurrences(text: str, substring: str, ignore_case: bool = False) -> List[Tuple[int, int]]:
    if not text or not substring:
        return []

    search_text = text.lower() if ignore_case else text
    search_substring = substring.lower() if ignore_case else substring

    spans: List[Tuple[int, int]] = []
    start = 0
    step = max(len(substring), 1)
    while True:
        idx = search_text.find(search_substring, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(substring)))
        start = idx + step
    return spans


def int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def get_annotation_span(item: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    start = item.get("start")
    end = item.get("end")

    if start is None:
        start = item.get("begin_offset") or item.get("offset_start")
    if end is None:
        end = item.get("stop") or item.get("offset_end")

    return int_or_none(start), int_or_none(end)


def spans_from_annotation_item(
    item: Dict[str, Any],
    raw_text: str,
    sequence_text: str,
    item_idx: int,
) -> Tuple[List[EntitySpan], List[str]]:
    issues: List[str] = []
    label = item.get("label")
    value = item.get("value")

    if label is None or value is None:
        return [], [f"annotation_item_{item_idx}_missing_label_or_value"]

    label_str = str(label).strip()
    value_text = normalize_text(value)
    if not label_str or not value_text:
        return [], [f"annotation_item_{item_idx}_empty_label_or_value"]

    spans: List[Tuple[int, int]] = []
    start, end = get_annotation_span(item)
    if start is not None and end is not None:
        if 0 <= start < end <= len(sequence_text):
            extracted = normalize_text(sequence_text[start:end])
            if extracted == value_text:
                spans = [(start, end)]
            else:
                issues.append(f"annotation_item_{item_idx}_span_value_mismatch")
        elif 0 <= start < end <= len(raw_text):
            raw_extracted = normalize_text(raw_text[start:end])
            raw_matches = find_all_occurrences(sequence_text, raw_extracted)
            if len(raw_matches) == 1:
                spans = raw_matches
                issues.append(f"annotation_item_{item_idx}_raw_span_remapped")
            elif len(raw_matches) > 1:
                issues.append(f"annotation_item_{item_idx}_ambiguous_raw_span_remap")
            else:
                issues.append(f"annotation_item_{item_idx}_span_not_mappable")
        else:
            issues.append(f"annotation_item_{item_idx}_invalid_span_range")

    if not spans:
        value_matches = find_all_occurrences(sequence_text, value_text)
        if len(value_matches) == 1:
            spans = value_matches
        elif len(value_matches) > 1:
            issues.append(f"annotation_item_{item_idx}_ambiguous_value_match")
        else:
            case_insensitive_matches = find_all_occurrences(sequence_text, value_text, ignore_case=True)
            if len(case_insensitive_matches) == 1:
                spans = case_insensitive_matches
                issues.append(f"annotation_item_{item_idx}_case_insensitive_value_match")
            elif len(case_insensitive_matches) > 1:
                issues.append(f"annotation_item_{item_idx}_ambiguous_case_insensitive_value_match")

    if not spans:
        issues.append(f"annotation_item_{item_idx}_missing_span")
        return [], issues

    entities = [
        EntitySpan(start=current_start, end=current_end, label=label_str, value=value_text)
        for current_start, current_end in spans
        if 0 <= current_start < current_end <= len(sequence_text)
    ]
    return entities, issues


def resolve_overlapping_entities(entities: Sequence[EntitySpan]) -> Tuple[List[EntitySpan], List[str]]:
    selected: List[EntitySpan] = []
    issues: List[str] = []

    by_priority = sorted(
        entities,
        key=lambda entity: (-(entity.end - entity.start), entity.start, entity.end, entity.label),
    )

    for entity in by_priority:
        if any(entity.start < kept.end and entity.end > kept.start for kept in selected):
            issues.append("overlapping_entity_removed")
            continue
        selected.append(entity)

    selected.sort(key=lambda entity: (entity.start, entity.end, entity.label))
    return selected, issues


def parse_entities(raw_annotations: Any, raw_text: str, sequence_text: str) -> Tuple[List[EntitySpan], List[str]]:
    items, issues = normalize_privacy_items(raw_annotations)
    entities: List[EntitySpan] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(f"annotation_item_{idx}_not_dict")
            continue
        item_entities, item_issues = spans_from_annotation_item(
            item=item,
            raw_text=raw_text,
            sequence_text=sequence_text,
            item_idx=idx,
        )
        issues.extend(item_issues)
        entities.extend(item_entities)

    deduped: List[EntitySpan] = []
    seen = set()
    for entity in sorted(entities, key=lambda current: (current.start, current.end, current.label)):
        key = (entity.start, entity.end, entity.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)

    resolved, overlap_issues = resolve_overlapping_entities(deduped)
    issues.extend(overlap_issues)
    return resolved, issues


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_with_offsets(text: str) -> List[Token]:
    return [Token(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def align_entities_to_tokens(tokens: Sequence[Token], entities: Sequence[EntitySpan]) -> Tuple[List[str], List[str]]:
    labels = ["O"] * len(tokens)
    issues: List[str] = []

    if not tokens or not entities:
        return labels, issues

    token_starts = [token.start for token in tokens]
    token_ends = [token.end for token in tokens]

    for ent_idx, entity in enumerate(sorted(entities, key=lambda current: (current.start, current.end))):
        left = bisect_right(token_ends, entity.start)
        right = bisect_left(token_starts, entity.end, lo=left)

        if left == right:
            issues.append(f"entity_{ent_idx}_no_token_overlap")
            continue

        for offset, token_idx in enumerate(range(left, right)):
            token = tokens[token_idx]
            if token.start < entity.start or token.end > entity.end:
                issues.append(f"entity_{ent_idx}_partial_token_overlap")

            if labels[token_idx] != "O":
                issues.append(f"token_{token_idx}_overlapping_entities")
                continue

            prefix = "B-" if offset == 0 else "I-"
            labels[token_idx] = f"{prefix}{entity.label}"

    return labels, issues


def build_sequence_sample(sample_id: Any, raw_text_value: Any, raw_annotations: Any) -> SequenceSample:
    raw_text = "" if is_missing(raw_text_value) else str(raw_text_value)
    sequence_text = normalize_text(raw_text)
    entities, parse_issues = parse_entities(raw_annotations, raw_text, sequence_text)
    tokens = tokenize_with_offsets(sequence_text)
    labels, alignment_issues = align_entities_to_tokens(tokens, entities)

    return SequenceSample(
        sample_id=sample_id,
        raw_text=raw_text,
        sequence_text=sequence_text,
        entities=entities,
        tokens=tokens,
        labels=labels,
        issues=parse_issues + alignment_issues,
    )


def word_shape(word: str, compress: bool = False) -> str:
    chars: List[str] = []
    for char in word:
        if char.isupper():
            code = "X"
        elif char.islower():
            code = "x"
        elif char.isdigit():
            code = "d"
        else:
            code = char

        if not compress or not chars or chars[-1] != code:
            chars.append(code)
    return "".join(chars)


def build_token_profiles(tokens: Sequence[Token]) -> List[TokenProfile]:
    profiles: List[TokenProfile] = []

    for token in tokens:
        word = token.text
        lower = word.lower()
        has_alpha = any(char.isalpha() for char in word)
        digit_count = sum(char.isdigit() for char in word)
        has_digit = digit_count > 0

        profiles.append(TokenProfile(
            text=word,
            lower=lower,
            prefix_1=word[:1],
            prefix_2=word[:2],
            prefix_3=word[:3],
            prefix_4=word[:4],
            suffix_1=word[-1:],
            suffix_2=word[-2:],
            suffix_3=word[-3:],
            suffix_4=word[-4:],
            shape=word_shape(word),
            compressed_shape=word_shape(word, compress=True),
            is_upper=word.isupper(),
            is_title=word.istitle(),
            is_digit=word.isdigit(),
            is_alpha=word.isalpha(),
            is_alnum=word.isalnum(),
            is_punct=bool(word) and all(not char.isalnum() and not char.isspace() for char in word),
            has_alpha=has_alpha,
            has_digit=has_digit,
            has_hyphen="-" in word,
            has_slash="/" in word,
            has_dot="." in word,
            has_at="@" in word,
            has_colon=":" in word,
            has_plus="+" in word,
            has_underscore="_" in word,
            alpha_digit_mix=has_alpha and has_digit,
            digit_count=digit_count,
            length=len(word),
            length_bucket=min(len(word), 20),
        ))

    return profiles


def add_neighbor_features(features: Dict[str, Any], prefix: str, profile: TokenProfile) -> None:
    features.update({
        f"{prefix}:lower": profile.lower,
        f"{prefix}:shape": profile.shape,
        f"{prefix}:compressed_shape": profile.compressed_shape,
        f"{prefix}:is_title": profile.is_title,
        f"{prefix}:is_upper": profile.is_upper,
        f"{prefix}:is_digit": profile.is_digit,
        f"{prefix}:has_digit": profile.has_digit,
    })


def profiles_to_features(profiles: Sequence[TokenProfile]) -> List[Dict[str, Any]]:
    features_by_token: List[Dict[str, Any]] = []
    n_tokens = len(profiles)

    for index, profile in enumerate(profiles):
        window_start = max(0, index - 2)
        window_end = min(n_tokens, index + 3)
        window_profiles = profiles[window_start:window_end]

        features: Dict[str, Any] = {
            "bias": 1.0,
            "word": profile.text,
            "lower": profile.lower,
            "prefix_1": profile.prefix_1,
            "prefix_2": profile.prefix_2,
            "prefix_3": profile.prefix_3,
            "prefix_4": profile.prefix_4,
            "suffix_1": profile.suffix_1,
            "suffix_2": profile.suffix_2,
            "suffix_3": profile.suffix_3,
            "suffix_4": profile.suffix_4,
            "shape": profile.shape,
            "compressed_shape": profile.compressed_shape,
            "is_upper": profile.is_upper,
            "is_title": profile.is_title,
            "is_digit": profile.is_digit,
            "is_alpha": profile.is_alpha,
            "is_alnum": profile.is_alnum,
            "is_punct": profile.is_punct,
            "has_alpha": profile.has_alpha,
            "has_digit": profile.has_digit,
            "has_hyphen": profile.has_hyphen,
            "has_slash": profile.has_slash,
            "has_dot": profile.has_dot,
            "has_at": profile.has_at,
            "has_colon": profile.has_colon,
            "has_plus": profile.has_plus,
            "has_underscore": profile.has_underscore,
            "alpha_digit_mix": profile.alpha_digit_mix,
            "length": profile.length,
            "length_bucket": profile.length_bucket,
            "window_has_at": any(current.has_at for current in window_profiles),
            "window_has_dot": any(current.has_dot for current in window_profiles),
            "window_has_slash": any(current.has_slash for current in window_profiles),
            "window_has_hyphen": any(current.has_hyphen for current in window_profiles),
            "window_digit_count": sum(current.digit_count for current in window_profiles),
        }

        if index == 0:
            features["BOS"] = True
        else:
            prev_profile = profiles[index - 1]
            add_neighbor_features(features, "-1", prev_profile)
            features["-1:lower+lower"] = f"{prev_profile.lower}|{profile.lower}"

        if index > 1:
            add_neighbor_features(features, "-2", profiles[index - 2])

        if index == n_tokens - 1:
            features["EOS"] = True
        else:
            next_profile = profiles[index + 1]
            add_neighbor_features(features, "+1", next_profile)
            features["lower+1:lower"] = f"{profile.lower}|{next_profile.lower}"

        if index < n_tokens - 2:
            add_neighbor_features(features, "+2", profiles[index + 2])

        features_by_token.append(features)

    return features_by_token


def sample_to_features(sample: SequenceSample) -> List[Dict[str, Any]]:
    profiles = build_token_profiles(sample.tokens)
    return profiles_to_features(profiles)


SEVERE_ALIGNMENT_ISSUE_FRAGMENTS = (
    "missing_span",
    "partial_token_overlap",
    "overlapping_entity_removed",
    "overlapping_entities",
    "no_token_overlap",
    "ambiguous_value_match",
    "ambiguous_case_insensitive_value_match",
    "ambiguous_raw_span_remap",
)


def has_severe_alignment_issues(issues: Sequence[str]) -> bool:
    return any(fragment in issue for issue in issues for fragment in SEVERE_ALIGNMENT_ISSUE_FRAGMENTS)


def build_crf_dataset(
    dataframe: pd.DataFrame,
    description: str,
    skip_noisy_samples: bool = False,
) -> Tuple[List[List[Dict[str, Any]]], List[List[str]], List[SequenceSample], Counter]:
    iterator: Iterable[Tuple[Any, str, Any]] = dataframe.itertuples(index=True, name=None)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(dataframe), desc=description, file=sys.stdout)

    X: List[List[Dict[str, Any]]] = []
    y: List[List[str]] = []
    samples: List[SequenceSample] = []
    issue_counter: Counter = Counter()

    for sample_id, raw_text, raw_annotations in iterator:
        sample = build_sequence_sample(sample_id, raw_text, raw_annotations)
        issue_counter.update(sample.issues)

        if not sample.tokens:
            issue_counter.update(["empty_token_sequence"])
            continue

        if skip_noisy_samples and has_severe_alignment_issues(sample.issues):
            issue_counter.update(["skipped_noisy_sample"])
            continue

        X.append(sample_to_features(sample))
        y.append(sample.labels)
        samples.append(sample)

    return X, y, samples, issue_counter


def normalize_annotation_items(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    items = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(items, tuple):
        items = list(items)
    if isinstance(items, list):
        return items
    return []


def extract_label_set(value: Any) -> set[str]:
    labels: set[str] = set()
    for item in normalize_annotation_items(value):
        if isinstance(item, dict):
            label = item.get("label")
            if label:
                labels.add(str(label))
    return labels


def raw_label_counter(values: Iterable[Any]) -> Counter:
    counter: Counter = Counter()
    for value in values:
        for label in extract_label_set(value):
            counter[label] += 1
    return counter


def load_filtered_dataframe(parquet_path: str) -> pd.DataFrame:
    if not os.path.exists(parquet_path):
        print(f"Error: file not found: {parquet_path}")
        sys.exit(1)

    print(f"Loading data from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path, columns=[TEXT_COL, ANNOTATION_COL])
    except Exception as exc:
        print(f"Error while reading parquet: {exc}")
        sys.exit(1)

    missing_columns = {TEXT_COL, ANNOTATION_COL} - set(df.columns)
    if missing_columns:
        print(f"Error: missing columns: {sorted(missing_columns)}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    df = df[[TEXT_COL, ANNOTATION_COL]].dropna(subset=[TEXT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip() != ""].copy()

    if len(df) < 10:
        print("Error: too few rows after filtering.")
        sys.exit(1)

    return df


def select_dataframe_rows(
    dataframe: pd.DataFrame,
    max_rows: Optional[int],
    sample_mode: str,
    sample_random_state: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(dataframe):
        if sample_mode == "random":
            return dataframe.sample(frac=1.0, random_state=sample_random_state)
        return dataframe.copy()

    if sample_mode == "random":
        return dataframe.sample(n=max_rows, random_state=sample_random_state)

    return dataframe.head(max_rows).copy()
