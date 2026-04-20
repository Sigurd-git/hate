from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .config import CleaningConfig


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV conservatively so mixed identity fields stay as strings."""
    try:
        return pd.read_csv(
            path,
            dtype=str,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except (UnicodeDecodeError, pd.errors.ParserError):
        csv.field_size_limit(sys.maxsize)
        decoded_text = path.read_bytes().decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(StringIO(decoded_text), restkey="_extra_fields", restval="")
        rows: list[dict[str, Any]] = []
        for row in reader:
            row.pop("_extra_fields", None)
            rows.append(row)
        return pd.DataFrame(rows)


def _normalize_string_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column_name in columns:
        if column_name in frame.columns:
            frame[column_name] = frame[column_name].fillna("").astype(str).str.strip()
    return frame


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_quantile(series: pd.Series, quantile: float, fallback: float = math.nan) -> float:
    valid_values = series.dropna()
    if valid_values.empty:
        return fallback
    return float(valid_values.quantile(quantile))


def _safe_correlation(left_values: Iterable[float], right_values: Iterable[float]) -> float:
    left_array = np.asarray(list(left_values), dtype=float)
    right_array = np.asarray(list(right_values), dtype=float)
    valid_mask = np.isfinite(left_array) & np.isfinite(right_array)
    if valid_mask.sum() < 3:
        return math.nan
    left_valid = left_array[valid_mask]
    right_valid = right_array[valid_mask]
    if np.unique(left_valid).size <= 1 or np.unique(right_valid).size <= 1:
        return math.nan
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _tokenize_response_sequence(values: Iterable[float]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        numeric_value = float(value)
        if math.isnan(numeric_value):
            continue
        if numeric_value.is_integer():
            tokens.append(str(int(numeric_value)))
        else:
            tokens.append(f"{numeric_value:.3f}")
    return tokens


def _compute_normalized_entropy(tokens: list[str]) -> float:
    if len(tokens) <= 1:
        return 0.0
    token_counts = Counter(tokens)
    probabilities = np.asarray(list(token_counts.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    if len(probabilities) <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(probabilities))


def _compute_lz_complexity(tokens: list[str]) -> int:
    if not tokens:
        return 0
    seen_substrings: set[tuple[str, ...]] = set()
    complexity = 0
    start_index = 0
    while start_index < len(tokens):
        end_index = start_index + 1
        while end_index <= len(tokens) and tuple(tokens[start_index:end_index]) in seen_substrings:
            end_index += 1
        seen_substrings.add(tuple(tokens[start_index : min(end_index, len(tokens))]))
        complexity += 1
        start_index = min(end_index, len(tokens))
    return complexity


def _compute_best_cycle_match(tokens: list[str], config: CleaningConfig) -> tuple[float, float]:
    if len(tokens) <= config.sequence_cycle_min_lag:
        return math.nan, 0.0

    best_lag = math.nan
    best_match_rate = 0.0
    max_lag = min(config.sequence_cycle_max_lag, len(tokens) - 1)
    for lag in range(config.sequence_cycle_min_lag, max_lag + 1):
        comparisons = len(tokens) - lag
        if comparisons <= 0:
            continue
        matches = sum(tokens[index] == tokens[index - lag] for index in range(lag, len(tokens)))
        match_rate = matches / comparisons
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_lag = float(lag)
    return best_lag, best_match_rate


def _compute_dominant_bigram_prop(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = [f"{tokens[index]}->{tokens[index + 1]}" for index in range(len(tokens) - 1)]
    bigram_counts = Counter(bigrams)
    return max(bigram_counts.values()) / len(bigrams)


def _compute_longest_monotonic_run(values: Iterable[float]) -> int:
    numeric_values = [float(value) for value in values if not math.isnan(float(value))]
    if not numeric_values:
        return 0
    if len(numeric_values) == 1:
        return 1

    longest_run = 2
    increasing_run = 1
    decreasing_run = 1
    previous_diff_sign = 0

    for previous_value, current_value in zip(numeric_values[:-1], numeric_values[1:]):
        diff = current_value - previous_value
        diff_sign = 0 if diff == 0 else (1 if diff > 0 else -1)

        if diff_sign >= 0 and previous_diff_sign >= 0:
            increasing_run += 1
        elif diff_sign >= 0:
            increasing_run = 2
        else:
            increasing_run = 1

        if diff_sign <= 0 and previous_diff_sign <= 0:
            decreasing_run += 1
        elif diff_sign <= 0:
            decreasing_run = 2
        else:
            decreasing_run = 1

        longest_run = max(longest_run, increasing_run, decreasing_run)
        previous_diff_sign = diff_sign

    return longest_run


def _build_behavioral_sequence_metrics(long_df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    non_repeat_df = long_df.loc[~long_df["is_repeat_trial_bool"]].copy()
    if non_repeat_df.empty:
        return pd.DataFrame(columns=["session_id", "condition"])

    non_repeat_df = non_repeat_df.sort_values(["session_id", "trial_index_num"], kind="mergesort")
    metric_rows: list[dict[str, Any]] = []

    for (session_id, condition), group in non_repeat_df.groupby(["session_id", "condition"], dropna=False, sort=False):
        response_values = group["response_value_num"].astype(float).tolist()
        tokens = _tokenize_response_sequence(response_values)
        best_cycle_lag, best_cycle_match_rate = _compute_best_cycle_match(tokens, config)
        lz_complexity = _compute_lz_complexity(tokens)
        longest_monotonic_run = _compute_longest_monotonic_run(response_values)
        trial_count = len(tokens)
        metric_rows.append(
            {
                "session_id": session_id,
                "condition": condition,
                "non_repeat_trial_count": trial_count,
                "sequence_best_cycle_lag": best_cycle_lag,
                "sequence_best_cycle_match_rate": best_cycle_match_rate,
                "sequence_dominant_bigram_prop": _compute_dominant_bigram_prop(tokens),
                "sequence_lz_complexity": lz_complexity,
                "sequence_lz_complexity_norm": lz_complexity / trial_count if trial_count else math.nan,
                "sequence_longest_monotonic_run": longest_monotonic_run,
                "sequence_longest_monotonic_run_prop": longest_monotonic_run / trial_count if trial_count else math.nan,
                "response_entropy_norm": _compute_normalized_entropy(tokens),
            }
        )

    return pd.DataFrame(metric_rows)


def _build_consensus_and_peer_metrics(long_df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    non_repeat_df = long_df.loc[~long_df["is_repeat_trial_bool"]].copy()
    if non_repeat_df.empty:
        return pd.DataFrame(columns=["session_id", "condition"])

    non_repeat_df = non_repeat_df.loc[
        non_repeat_df["template_id"].fillna("").astype(str).ne("") & non_repeat_df["response_value_num"].notna()
    ].copy()
    if non_repeat_df.empty:
        return pd.DataFrame(columns=["session_id", "condition"])

    template_totals = (
        non_repeat_df.groupby(["condition", "template_id"], dropna=False)["response_value_num"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "template_total_score", "count": "template_total_count"})
    )
    non_repeat_df = non_repeat_df.merge(template_totals, on=["condition", "template_id"], how="left")
    non_repeat_df["leave_one_out_template_mean"] = np.where(
        non_repeat_df["template_total_count"] > 1,
        (non_repeat_df["template_total_score"] - non_repeat_df["response_value_num"])
        / (non_repeat_df["template_total_count"] - 1),
        np.nan,
    )

    consensus_rows: list[dict[str, Any]] = []
    for (session_id, condition), group in non_repeat_df.groupby(["session_id", "condition"], dropna=False, sort=False):
        consensus_rows.append(
            {
                "session_id": session_id,
                "condition": condition,
                "leave_one_out_item_consensus_corr": _safe_correlation(
                    group["response_value_num"],
                    group["leave_one_out_template_mean"],
                )
            }
        )

    peer_rows: list[dict[str, Any]] = []
    for condition, group in non_repeat_df.groupby("condition", dropna=False, sort=False):
        pivot_df = group.pivot_table(index="session_id", columns="template_id", values="response_value_num", aggfunc="mean")
        session_ids = pivot_df.index.astype(str).tolist()
        if not session_ids:
            continue
        response_matrix = pivot_df.to_numpy(dtype=float)
        pair_records: list[tuple[int, int, float, float]] = []
        correlation_values: list[float] = []
        exact_match_values: list[float] = []

        for left_index in range(len(session_ids)):
            for right_index in range(left_index + 1, len(session_ids)):
                valid_mask = np.isfinite(response_matrix[left_index]) & np.isfinite(response_matrix[right_index])
                shared_item_count = int(valid_mask.sum())
                if shared_item_count < config.peer_similarity_min_shared_items:
                    continue
                left_values = response_matrix[left_index, valid_mask]
                right_values = response_matrix[right_index, valid_mask]
                correlation = _safe_correlation(left_values, right_values)
                exact_match_prop = float(np.mean(np.isclose(left_values, right_values)))
                pair_records.append((left_index, right_index, correlation, exact_match_prop))
                if math.isfinite(correlation):
                    correlation_values.append(correlation)
                exact_match_values.append(exact_match_prop)

        correlation_threshold = max(
            config.peer_similarity_abs_floor,
            _safe_quantile(pd.Series(correlation_values, dtype=float), config.peer_similarity_quantile, fallback=config.peer_similarity_abs_floor),
        )
        exact_match_threshold = max(
            config.peer_exact_match_abs_floor,
            _safe_quantile(pd.Series(exact_match_values, dtype=float), config.peer_exact_match_quantile, fallback=config.peer_exact_match_abs_floor),
        )

        max_correlation = {session_id: math.nan for session_id in session_ids}
        max_exact_match = {session_id: 0.0 for session_id in session_ids}
        high_similarity_neighbors = {session_id: set() for session_id in session_ids}

        for left_index, right_index, correlation, exact_match_prop in pair_records:
            left_session = session_ids[left_index]
            right_session = session_ids[right_index]

            if math.isfinite(correlation):
                if not math.isfinite(max_correlation[left_session]) or correlation > max_correlation[left_session]:
                    max_correlation[left_session] = correlation
                if not math.isfinite(max_correlation[right_session]) or correlation > max_correlation[right_session]:
                    max_correlation[right_session] = correlation

            if exact_match_prop > max_exact_match[left_session]:
                max_exact_match[left_session] = exact_match_prop
            if exact_match_prop > max_exact_match[right_session]:
                max_exact_match[right_session] = exact_match_prop

            if (math.isfinite(correlation) and correlation >= correlation_threshold) or exact_match_prop >= exact_match_threshold:
                high_similarity_neighbors[left_session].add(right_session)
                high_similarity_neighbors[right_session].add(left_session)

        for session_id in session_ids:
            peer_rows.append(
                {
                    "session_id": session_id,
                    "condition": condition,
                    "peer_max_correlation": max_correlation[session_id],
                    "peer_max_exact_match_prop": max_exact_match[session_id],
                    "peer_high_similarity_neighbor_count": len(high_similarity_neighbors[session_id]),
                    "peer_similarity_corr_threshold": correlation_threshold,
                    "peer_similarity_exact_match_threshold": exact_match_threshold,
                }
            )

    consensus_df = pd.DataFrame(consensus_rows)
    peer_df = pd.DataFrame(peer_rows)
    if consensus_df.empty:
        return peer_df
    if peer_df.empty:
        return consensus_df
    return consensus_df.merge(peer_df, on=["session_id", "condition"], how="outer")


def _flag_behavioral_bot_metrics(behavioral_metrics: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    if behavioral_metrics.empty:
        return behavioral_metrics

    quantiles = (
        behavioral_metrics.groupby("condition", dropna=False)
        .agg(
            sequence_low_complexity_q=(
                "sequence_lz_complexity_norm",
                lambda series: _safe_quantile(series, config.low_complexity_quantile),
            ),
            sequence_low_complexity_extreme_q=(
                "sequence_lz_complexity_norm",
                lambda series: _safe_quantile(series, config.low_complexity_extreme_quantile),
            ),
            sequence_bigram_high_q=(
                "sequence_dominant_bigram_prop",
                lambda series: _safe_quantile(series, 1 - config.low_complexity_extreme_quantile),
            ),
            sequence_monotonic_high_q=(
                "sequence_longest_monotonic_run_prop",
                lambda series: _safe_quantile(series, 1 - config.low_complexity_extreme_quantile),
            ),
            sequence_cycle_high_q=(
                "sequence_best_cycle_match_rate",
                lambda series: _safe_quantile(series, 1 - config.low_complexity_extreme_quantile),
            ),
            entropy_high_q=(
                "response_entropy_norm",
                lambda series: _safe_quantile(series, config.high_entropy_quantile),
            ),
            consensus_corr_low_q=(
                "leave_one_out_item_consensus_corr",
                lambda series: _safe_quantile(series, config.low_consensus_corr_quantile),
            ),
            consensus_corr_low_extreme_q=(
                "leave_one_out_item_consensus_corr",
                lambda series: _safe_quantile(series, config.low_consensus_corr_extreme_quantile),
            ),
        )
        .reset_index()
    )
    flagged = behavioral_metrics.merge(quantiles, on="condition", how="left")

    flagged["pattern_cycle_flag"] = flagged["sequence_best_cycle_match_rate"] >= np.maximum(
        config.pattern_cycle_match_soft_threshold,
        flagged["sequence_cycle_high_q"].fillna(config.pattern_cycle_match_soft_threshold),
    )
    flagged["pattern_cycle_hard_flag"] = flagged["sequence_best_cycle_match_rate"] >= np.maximum(
        config.pattern_cycle_match_hard_threshold,
        flagged["sequence_cycle_high_q"].fillna(config.pattern_cycle_match_hard_threshold),
    )
    flagged["pattern_bigram_flag"] = flagged["sequence_dominant_bigram_prop"] >= np.maximum(
        config.pattern_bigram_abs_floor,
        flagged["sequence_bigram_high_q"].fillna(config.pattern_bigram_abs_floor),
    )
    flagged["pattern_low_complexity_flag"] = (
        flagged["sequence_lz_complexity_norm"] <= flagged["sequence_low_complexity_q"].fillna(-math.inf)
    )
    flagged["pattern_template_walk_flag"] = flagged["sequence_longest_monotonic_run_prop"] >= np.maximum(
        config.pattern_monotonic_run_abs_floor,
        flagged["sequence_monotonic_high_q"].fillna(config.pattern_monotonic_run_abs_floor),
    )
    flagged["pseudo_random_flag"] = (
        flagged["leave_one_out_item_consensus_corr"] <= flagged["consensus_corr_low_q"].fillna(-math.inf)
    ) & (
        flagged["response_entropy_norm"] >= flagged["entropy_high_q"].fillna(math.inf)
    )
    flagged["pseudo_random_hard_flag"] = (
        flagged["leave_one_out_item_consensus_corr"] <= flagged["consensus_corr_low_extreme_q"].fillna(-math.inf)
    ) & (
        flagged["response_entropy_norm"] >= flagged["entropy_high_q"].fillna(math.inf)
    )
    flagged["batch_high_similarity_flag"] = (
        flagged["peer_max_correlation"] >= flagged["peer_similarity_corr_threshold"].fillna(math.inf)
    )
    flagged["batch_exact_match_flag"] = (
        flagged["peer_max_exact_match_prop"] >= flagged["peer_similarity_exact_match_threshold"].fillna(math.inf)
    )
    flagged["batch_cluster_flag"] = (
        flagged["peer_high_similarity_neighbor_count"].fillna(0) >= config.peer_similarity_cluster_min_neighbors
    )
    flagged["script_clone_hard_flag"] = (
        flagged["peer_max_exact_match_prop"] >= config.peer_clone_hard_exact_match_threshold
    )

    soft_flag_columns = [
        "pattern_cycle_flag",
        "pattern_bigram_flag",
        "pattern_low_complexity_flag",
        "pattern_template_walk_flag",
        "pseudo_random_flag",
        "batch_high_similarity_flag",
        "batch_exact_match_flag",
        "batch_cluster_flag",
    ]
    flagged["bot_behavior_soft_flag_count"] = flagged[soft_flag_columns].fillna(False).sum(axis=1)
    flagged["bot_behavior_hard_flag"] = (
        flagged["pattern_cycle_hard_flag"]
        | flagged["pseudo_random_hard_flag"]
        | flagged["script_clone_hard_flag"]
    )
    flagged["suspicious_behavior_review_flag"] = flagged["bot_behavior_hard_flag"] | flagged["bot_behavior_soft_flag_count"].gt(0)
    flagged["ultra_clean_additional_exclusion_flag"] = flagged["bot_behavior_hard_flag"] | (
        flagged["bot_behavior_soft_flag_count"] >= config.ultra_clean_soft_flag_threshold
    )
    return flagged


def _build_identity_clusters(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Build connected duplicate clusters using name OR phone matches.

    Participants are linked when they share the same non-empty name or the same
    non-empty phone number. Any connected component of size > 1 is treated as a
    duplicate identity cluster, and every record in that cluster is marked for
    deletion.
    """

    parent = list(range(len(wide_df)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    for column_name in ["name", "phone"]:
        populated = wide_df.index[wide_df[column_name].ne("")]
        grouped_indices = wide_df.loc[populated].groupby(column_name, sort=False).indices
        for member_indices in grouped_indices.values():
            if len(member_indices) <= 1:
                continue
            anchor_index = int(member_indices[0])
            for current_index in member_indices[1:]:
                union(anchor_index, int(current_index))

    cluster_root = pd.Series([find(index) for index in range(len(wide_df))], index=wide_df.index)
    cluster_size = cluster_root.map(cluster_root.value_counts())

    wide_df["identity_cluster_root"] = cluster_root
    wide_df["duplicate_identity_group_size"] = cluster_size.astype(int)
    wide_df["duplicate_identity_any_flag"] = wide_df["duplicate_identity_group_size"].gt(1)
    wide_df["duplicate_identity_keep"] = False
    wide_df["duplicate_identity_drop"] = wide_df["duplicate_identity_any_flag"]
    return wide_df


def _prepare_wide_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    wide_df = wide_df.copy()
    wide_df = _normalize_string_columns(
        wide_df,
        [
            "session_id",
            "final_participant_number",
            "condition",
            "name",
            "phone",
            "student_id",
            "alipay_account",
        ],
    )
    wide_df["final_participant_number_num"] = _coerce_numeric(wide_df["final_participant_number"])

    wide_df = wide_df.sort_values(
        by=["final_participant_number_num", "session_id"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    wide_df["duplicate_name_group_size"] = (
        wide_df.groupby("name")["session_id"].transform("size").where(wide_df["name"].ne(""), 1).fillna(1).astype(int)
    )
    wide_df["duplicate_phone_group_size"] = (
        wide_df.groupby("phone")["session_id"].transform("size").where(wide_df["phone"].ne(""), 1).fillna(1).astype(int)
    )
    wide_df["duplicate_name_flag"] = wide_df["name"].ne("") & wide_df["duplicate_name_group_size"].gt(1)
    wide_df["duplicate_phone_flag"] = wide_df["phone"].ne("") & wide_df["duplicate_phone_group_size"].gt(1)

    wide_df = _build_identity_clusters(wide_df)
    return wide_df


def _prepare_long_table(long_df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df = _normalize_string_columns(
        long_df,
        [
            "session_id",
            "final_participant_number",
            "condition",
            "name",
            "phone",
            "item_id",
            "template_id",
            "is_repeat_trial",
            "repeat_pair_id",
            "repeat_source_trial_index",
        ],
    )

    for column_name in config.numeric_trial_columns:
        long_df[f"{column_name}_num"] = _coerce_numeric(long_df[column_name])

    long_df["is_repeat_trial_bool"] = long_df["is_repeat_trial"].str.upper().eq("TRUE")
    long_df["rt_first_invalid"] = long_df["rt_first_ms_num"] < 0
    long_df["rt_submit_invalid"] = long_df["rt_submit_ms_num"] < 0
    long_df.loc[long_df["rt_first_invalid"], "rt_first_ms_num"] = np.nan
    long_df.loc[long_df["rt_submit_invalid"], "rt_submit_ms_num"] = np.nan
    long_df["rt_extreme_long_flag"] = long_df["rt_submit_ms_num"] > config.extreme_long_rt_ms
    long_df["rt_usable_flag"] = (
        long_df["rt_first_ms_num"].notna()
        & long_df["rt_submit_ms_num"].notna()
        & ~long_df["rt_extreme_long_flag"].fillna(False)
        & long_df["page_hidden_count_num"].fillna(0).eq(0)
        & long_df["page_hidden_duration_ms_num"].fillna(0).eq(0)
    )
    long_df["page_hidden_flag"] = long_df["page_hidden_count_num"].fillna(0).gt(0)
    return long_df


def _build_structural_metrics(long_df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    repeat_gap = long_df.loc[long_df["is_repeat_trial_bool"]].copy()
    repeat_gap["repeat_gap"] = repeat_gap["trial_index_num"] - repeat_gap["repeat_source_trial_index_num"]

    structural_metrics = (
        long_df.groupby("session_id", dropna=False)
        .agg(
            condition_nunique=("condition", "nunique"),
            n_trials=("trial_index", "size"),
            n_repeat=("is_repeat_trial_bool", "sum"),
        )
        .reset_index()
    )
    repeat_gap_summary = (
        repeat_gap.groupby("session_id", dropna=False)
        .agg(min_repeat_gap=("repeat_gap", "min"))
        .reset_index()
    )
    structural_metrics = structural_metrics.merge(repeat_gap_summary, on="session_id", how="left")
    structural_metrics["min_repeat_gap"] = structural_metrics["min_repeat_gap"].fillna(np.inf)

    structural_metrics["condition_not_unique_flag"] = structural_metrics["condition_nunique"] != 1
    structural_metrics["wrong_trial_count_flag"] = structural_metrics["n_trials"] != config.expected_trial_count
    structural_metrics["wrong_repeat_count_flag"] = structural_metrics["n_repeat"] != config.expected_repeat_count
    structural_metrics["repeat_gap_below_min_flag"] = structural_metrics["min_repeat_gap"] < config.min_repeat_gap
    structural_metrics["structural_anomaly_flag"] = (
        structural_metrics[
            [
                "condition_not_unique_flag",
                "wrong_trial_count_flag",
                "wrong_repeat_count_flag",
                "repeat_gap_below_min_flag",
            ]
        ]
        .any(axis=1)
    )
    return structural_metrics


def _build_repeat_metrics(long_df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_trials = long_df[["session_id", "trial_index_num", "response_value_num"]].rename(
        columns={
            "trial_index_num": "repeat_source_trial_index_num",
            "response_value_num": "repeat_source_response_value",
        }
    )

    repeat_trials = long_df.loc[long_df["is_repeat_trial_bool"]].copy()
    repeat_trials = repeat_trials.merge(
        source_trials,
        on=["session_id", "repeat_source_trial_index_num"],
        how="left",
    )
    repeat_trials["repeat_abs_diff"] = (
        repeat_trials["response_value_num"] - repeat_trials["repeat_source_response_value"]
    ).abs()
    repeat_trials["repeat_severe_threshold"] = repeat_trials["condition"].map(config.repeat_severe_thresholds)
    repeat_trials["repeat_severe_flag"] = repeat_trials["repeat_abs_diff"] >= repeat_trials["repeat_severe_threshold"]

    repeat_summary = (
        repeat_trials.groupby("session_id", dropna=False)
        .agg(
            repeat_mean_abs_diff=("repeat_abs_diff", "mean"),
            repeat_max_abs_diff=("repeat_abs_diff", "max"),
            repeat_severe_count=("repeat_severe_flag", "sum"),
            repeat_missing_source_count=("repeat_source_response_value", lambda values: int(pd.isna(values).sum())),
        )
        .reset_index()
    )

    long_with_repeats = long_df.merge(
        repeat_trials[
            [
                "session_id",
                "trial_index_num",
                "repeat_source_response_value",
                "repeat_abs_diff",
                "repeat_severe_threshold",
                "repeat_severe_flag",
            ]
        ],
        on=["session_id", "trial_index_num"],
        how="left",
    )
    return long_with_repeats, repeat_summary


def _build_response_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    response_summary = (
        long_df.groupby(["session_id", "condition"], dropna=False)["response_value_num"]
        .agg(["nunique", "std", "var", "count"])
        .reset_index()
        .rename(
            columns={
                "nunique": "n_unique_response",
                "std": "response_sd",
                "var": "response_var",
                "count": "response_non_missing_count",
            }
        )
    )

    mode_summary = (
        long_df.groupby(["session_id", "condition", "response_value_num"], dropna=False)
        .size()
        .reset_index(name="response_count")
        .sort_values(
            by=["session_id", "condition", "response_count", "response_value_num"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["session_id", "condition"], keep="first")
        .rename(columns={"response_count": "mode_count", "response_value_num": "mode_response_value"})
    )

    response_summary = response_summary.merge(
        mode_summary[["session_id", "condition", "mode_count", "mode_response_value"]],
        on=["session_id", "condition"],
        how="left",
    )
    response_summary["mode_prop"] = response_summary["mode_count"] / response_summary["response_non_missing_count"].replace(0, np.nan)
    response_summary["all_same_response_flag"] = response_summary["n_unique_response"] == 1
    return response_summary


def _build_rt_metrics(long_df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable_trials = long_df.loc[long_df["rt_usable_flag"]].copy()
    usable_trials["log_rt_submit_ms"] = np.log(usable_trials["rt_submit_ms_num"])

    if usable_trials.empty:
        fast_thresholds = pd.DataFrame(columns=["condition", "fast_trial_log_rt_threshold"])
        usable_trials["fast_trial_flag"] = False
    else:
        fast_thresholds = (
            usable_trials.groupby("condition", dropna=False)["log_rt_submit_ms"]
            .quantile(config.fast_trial_quantile)
            .reset_index(name="fast_trial_log_rt_threshold")
        )
        usable_trials = usable_trials.merge(fast_thresholds, on="condition", how="left")
        usable_trials["fast_trial_flag"] = usable_trials["log_rt_submit_ms"] <= usable_trials["fast_trial_log_rt_threshold"]

    trial_base = long_df[["session_id", "condition", "trial_index_num"]].copy()
    usable_flag_by_trial = usable_trials[["session_id", "trial_index_num", "fast_trial_flag"]].copy()
    trial_base = trial_base.merge(usable_flag_by_trial, on=["session_id", "trial_index_num"], how="left")
    trial_base["fast_trial_flag"] = trial_base["fast_trial_flag"].astype("boolean").fillna(False).astype(bool)

    participant_rt = (
        long_df.groupby(["session_id", "condition"], dropna=False)
        .agg(
            prop_rt_usable=("rt_usable_flag", "mean"),
            hidden_trial_ratio=("page_hidden_flag", "mean"),
            total_hidden_duration_ms=("page_hidden_duration_ms_num", lambda values: float(np.nansum(values))),
            min_rt_submit_ms=("rt_submit_ms_num", "min"),
        )
        .reset_index()
    )

    usable_participant_rt = (
        usable_trials.groupby(["session_id", "condition"], dropna=False)
        .agg(
            median_rt_submit_ms=("rt_submit_ms_num", "median"),
            median_rt_first_ms=("rt_first_ms_num", "median"),
        )
        .reset_index()
    )
    participant_rt = participant_rt.merge(usable_participant_rt, on=["session_id", "condition"], how="left")

    fast_trial_ratio = (
        usable_trials.groupby(["session_id", "condition"], dropna=False)["fast_trial_flag"]
        .mean()
        .reset_index(name="fast_trial_ratio")
    )
    participant_rt = participant_rt.merge(fast_trial_ratio, on=["session_id", "condition"], how="left")
    participant_rt["fast_trial_ratio"] = participant_rt["fast_trial_ratio"].fillna(0.0)

    median_rt_quantiles = (
        participant_rt.groupby("condition", dropna=False)
        .agg(
            fast_participant_threshold=(
                "median_rt_submit_ms",
                lambda series: _safe_quantile(series, config.fast_participant_quantile),
            ),
            fast_participant_extreme_threshold=(
                "median_rt_submit_ms",
                lambda series: _safe_quantile(series, config.fast_participant_extreme_quantile),
            ),
        )
        .reset_index()
    )
    participant_rt = participant_rt.merge(median_rt_quantiles, on="condition", how="left")
    participant_rt["fast_participant_flag"] = participant_rt["median_rt_submit_ms"] <= participant_rt["fast_participant_threshold"]
    participant_rt["fast_participant_extreme_flag"] = participant_rt["median_rt_submit_ms"] <= participant_rt["fast_participant_extreme_threshold"]

    participant_rt["prop_rt_usable"] = participant_rt["prop_rt_usable"].fillna(0.0)
    participant_rt["hidden_trial_ratio"] = participant_rt["hidden_trial_ratio"].fillna(0.0)
    participant_rt["total_hidden_duration_ms"] = participant_rt["total_hidden_duration_ms"].fillna(0.0)
    participant_rt["fast_participant_flag"] = participant_rt["fast_participant_flag"].fillna(False)
    participant_rt["fast_participant_extreme_flag"] = participant_rt["fast_participant_extreme_flag"].fillna(False)
    participant_rt["min_rt_below_threshold_flag"] = participant_rt["min_rt_submit_ms"] < config.min_allowed_rt_ms
    participant_rt["min_rt_below_threshold_flag"] = participant_rt["min_rt_below_threshold_flag"].fillna(False)

    return participant_rt, usable_trials, trial_base[["session_id", "trial_index_num", "fast_trial_flag"]]


def _build_condition_variance_flags(response_summary: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    flagged = response_summary.copy()

    quantiles = (
        flagged.groupby("condition", dropna=False)
        .agg(
            n_unique_low_q=(
                "n_unique_response",
                lambda series: _safe_quantile(series, config.low_variance_quantile),
            ),
            n_unique_low_extreme_q=(
                "n_unique_response",
                lambda series: _safe_quantile(series, config.low_variance_extreme_quantile),
            ),
            response_sd_low_q=(
                "response_sd",
                lambda series: _safe_quantile(series, config.low_variance_quantile),
            ),
            response_sd_low_extreme_q=(
                "response_sd",
                lambda series: _safe_quantile(series, config.low_variance_extreme_quantile),
            ),
            mode_prop_high_q=(
                "mode_prop",
                lambda series: _safe_quantile(series, 1 - config.low_variance_quantile),
            ),
            mode_prop_high_extreme_q=(
                "mode_prop",
                lambda series: _safe_quantile(series, 1 - config.low_variance_extreme_quantile),
            ),
        )
        .reset_index()
    )

    flagged = flagged.merge(quantiles, on="condition", how="left")
    flagged["variance_low_extreme_flag"] = (
        (flagged["n_unique_response"] <= flagged["n_unique_low_extreme_q"])
        | (flagged["response_sd"] <= flagged["response_sd_low_extreme_q"])
        | (flagged["mode_prop"] >= flagged["mode_prop_high_extreme_q"])
    )
    flagged["variance_low_flag"] = (
        (flagged["n_unique_response"] <= flagged["n_unique_low_q"])
        | (flagged["response_sd"] <= flagged["response_sd_low_q"])
        | (flagged["mode_prop"] >= flagged["mode_prop_high_q"])
    )
    flagged["variance_low_extreme_flag"] = flagged["variance_low_extreme_flag"].fillna(False)
    flagged["variance_low_flag"] = flagged["variance_low_flag"].fillna(False)
    return flagged


def _load_manual_session_flags(optional_path: Optional[Path], flag_column_name: str) -> pd.DataFrame:
    if optional_path is None or not optional_path.exists():
        return pd.DataFrame(columns=["session_id", flag_column_name])

    flags_df = pd.read_csv(optional_path, dtype=str, encoding="utf-8-sig")
    if flags_df.empty:
        return pd.DataFrame(columns=["session_id", flag_column_name])

    if "session_id" not in flags_df.columns:
        flags_df = flags_df.rename(columns={flags_df.columns[0]: "session_id"})

    flags_df["session_id"] = flags_df["session_id"].fillna("").astype(str).str.strip()
    flags_df = flags_df.loc[flags_df["session_id"].ne("")].copy()
    flags_df = flags_df.drop_duplicates(subset=["session_id"])
    flags_df[flag_column_name] = True
    return flags_df[["session_id", flag_column_name]]


def _load_manual_fraud_flags(optional_path: Optional[Path]) -> pd.DataFrame:
    return _load_manual_session_flags(optional_path, "manual_fraud_cluster_flag")


def _load_manual_final_exclusion_flags(optional_path: Optional[Path]) -> pd.DataFrame:
    return _load_manual_session_flags(optional_path, "manual_final_exclusion_flag")


def _merge_participant_master(
    wide_df: pd.DataFrame,
    structural_metrics: pd.DataFrame,
    response_metrics: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    rt_metrics: pd.DataFrame,
    behavioral_metrics: pd.DataFrame,
    manual_fraud_flags: pd.DataFrame,
    config: CleaningConfig,
) -> pd.DataFrame:
    participant_qc = wide_df.merge(structural_metrics, on="session_id", how="left")
    participant_qc = participant_qc.merge(
        response_metrics.drop(columns=["condition"], errors="ignore"),
        on="session_id",
        how="left",
    )
    participant_qc = participant_qc.merge(repeat_metrics, on="session_id", how="left")
    participant_qc = participant_qc.merge(
        rt_metrics.drop(columns=["condition"], errors="ignore"),
        on="session_id",
        how="left",
    )
    participant_qc = participant_qc.merge(
        behavioral_metrics.drop(columns=["condition"], errors="ignore"),
        on="session_id",
        how="left",
    )
    participant_qc = participant_qc.merge(manual_fraud_flags, on="session_id", how="left")

    for boolean_column in [
        "duplicate_identity_drop",
        "duplicate_identity_keep",
        "condition_not_unique_flag",
        "wrong_trial_count_flag",
        "wrong_repeat_count_flag",
        "repeat_gap_below_min_flag",
        "structural_anomaly_flag",
        "all_same_response_flag",
        "variance_low_extreme_flag",
        "variance_low_flag",
        "fast_participant_flag",
        "fast_participant_extreme_flag",
        "min_rt_below_threshold_flag",
        "manual_fraud_cluster_flag",
        "pattern_cycle_flag",
        "pattern_cycle_hard_flag",
        "pattern_bigram_flag",
        "pattern_low_complexity_flag",
        "pattern_template_walk_flag",
        "pseudo_random_flag",
        "pseudo_random_hard_flag",
        "batch_high_similarity_flag",
        "batch_exact_match_flag",
        "batch_cluster_flag",
        "script_clone_hard_flag",
        "bot_behavior_hard_flag",
        "suspicious_behavior_review_flag",
        "ultra_clean_additional_exclusion_flag",
    ]:
        if boolean_column in participant_qc.columns:
            participant_qc[boolean_column] = participant_qc[boolean_column].astype("boolean").fillna(False).astype(bool)

    for numeric_column in [
        "repeat_severe_count",
        "repeat_missing_source_count",
        "prop_rt_usable",
        "hidden_trial_ratio",
        "total_hidden_duration_ms",
        "fast_trial_ratio",
        "min_rt_submit_ms",
        "sequence_best_cycle_lag",
        "sequence_best_cycle_match_rate",
        "sequence_dominant_bigram_prop",
        "sequence_lz_complexity",
        "sequence_lz_complexity_norm",
        "sequence_longest_monotonic_run",
        "sequence_longest_monotonic_run_prop",
        "response_entropy_norm",
        "leave_one_out_item_consensus_corr",
        "peer_max_correlation",
        "peer_max_exact_match_prop",
        "peer_high_similarity_neighbor_count",
        "peer_similarity_corr_threshold",
        "peer_similarity_exact_match_threshold",
        "bot_behavior_soft_flag_count",
    ]:
        if numeric_column in participant_qc.columns:
            participant_qc[numeric_column] = pd.to_numeric(participant_qc[numeric_column], errors="coerce")

    participant_qc["repeat_severe_count"] = participant_qc["repeat_severe_count"].fillna(0)
    participant_qc["prop_rt_usable"] = participant_qc["prop_rt_usable"].fillna(0.0)
    participant_qc["hidden_trial_ratio"] = participant_qc["hidden_trial_ratio"].fillna(0.0)
    participant_qc["total_hidden_duration_ms"] = participant_qc["total_hidden_duration_ms"].fillna(0.0)
    participant_qc["bot_behavior_soft_flag_count"] = participant_qc["bot_behavior_soft_flag_count"].fillna(0)

    participant_qc["repeat_soft_flag"] = participant_qc["repeat_severe_count"] == 1
    participant_qc["hidden_ratio_flag"] = participant_qc["hidden_trial_ratio"] > config.hidden_trial_ratio_threshold
    participant_qc["hidden_duration_flag"] = participant_qc["total_hidden_duration_ms"] > config.total_hidden_duration_threshold_ms
    participant_qc["low_rt_usable_flag"] = participant_qc["prop_rt_usable"] < config.min_rt_usable_ratio
    participant_qc["min_rt_submit_ms"] = participant_qc["min_rt_submit_ms"].fillna(np.nan)

    participant_qc["hard_exclusion_flag"] = (
        participant_qc["duplicate_identity_drop"]
        | participant_qc["manual_fraud_cluster_flag"]
        | participant_qc["structural_anomaly_flag"]
        | participant_qc["all_same_response_flag"]
        | participant_qc["repeat_severe_count"].ge(2)
        | participant_qc["low_rt_usable_flag"]
    )

    soft_flag_columns = [
        "repeat_soft_flag",
        "fast_participant_flag",
        "hidden_ratio_flag",
        "hidden_duration_flag",
        "variance_low_flag",
    ]
    participant_qc["soft_flag_count"] = participant_qc[soft_flag_columns].sum(axis=1)
    participant_qc["strict_additional_exclusion_flag"] = participant_qc["soft_flag_count"] >= config.strict_soft_flag_threshold
    participant_qc["strict_exclusion_flag"] = participant_qc["hard_exclusion_flag"] | participant_qc["strict_additional_exclusion_flag"]
    return participant_qc


def _write_summary_report(output_dir: Path, participant_qc: pd.DataFrame, config: CleaningConfig) -> Path:
    counts_by_condition = participant_qc.groupby("condition", dropna=False).agg(
        total_participants=("session_id", "size"),
        hard_excluded=("hard_exclusion_flag", "sum"),
        strict_excluded=("strict_exclusion_flag", "sum"),
        suspicious_behavior_review=("suspicious_behavior_review_flag", "sum"),
        ultra_clean_additional_excluded=("ultra_clean_additional_exclusion_flag", "sum"),
        duplicate_identity_drop=("duplicate_identity_drop", "sum"),
        structural_anomaly=("structural_anomaly_flag", "sum"),
        all_same_response=("all_same_response_flag", "sum"),
        repeat_severe_ge_2=("repeat_severe_count", lambda values: int((pd.Series(values).fillna(0) >= 2).sum())),
        low_rt_usable=("low_rt_usable_flag", "sum"),
        min_rt_below_threshold=("min_rt_below_threshold_flag", "sum"),
        manual_fraud_flag=("manual_fraud_cluster_flag", "sum"),
    )

    summary_lines = [
        "# Human aggression-rating cleaning summary",
        "",
        "This report reflects a conservative implementation of the requested rules.",
        "Duplicate identities are clustered when participants share the same non-empty name or the same non-empty phone number.",
        "Optional manual fraud-cluster exclusions are only applied when a manual session-id CSV is provided.",
        "Low variance remains a soft flag unless it is combined with other soft flags in the strict-cleaned version.",
        "",
        f"Total participants in wide master: {len(participant_qc)}",
        f"Hard-cleaned participants retained: {(~participant_qc['hard_exclusion_flag']).sum()}",
        f"Strict-cleaned participants retained: {(~participant_qc['strict_exclusion_flag']).sum()}",
        f"Suspicious-behavior review rows: {int(participant_qc['suspicious_behavior_review_flag'].sum())}",
        f"Ultra-clean retained before manual final-exclusion file is applied: {int((~participant_qc['strict_exclusion_flag'] & ~participant_qc['min_rt_below_threshold_flag'] & ~participant_qc['ultra_clean_additional_exclusion_flag']).sum())}",
        "",
        "## Counts by condition",
        counts_by_condition.to_string(),
        "",
        "## Exclusion totals",
        f"Duplicate identity drops: {int(participant_qc['duplicate_identity_drop'].sum())}",
        f"Participants in duplicate-name clusters: {int(participant_qc['duplicate_name_flag'].sum())}",
        f"Participants in duplicate-phone clusters: {int(participant_qc['duplicate_phone_flag'].sum())}",
        f"Structural anomalies: {int(participant_qc['structural_anomaly_flag'].sum())}",
        f"All-same responses: {int(participant_qc['all_same_response_flag'].sum())}",
        f"Repeat severe count >= 2: {int((participant_qc['repeat_severe_count'].fillna(0) >= 2).sum())}",
        f"Low RT usable ratio: {int(participant_qc['low_rt_usable_flag'].sum())}",
        f"Minimum rt_submit_ms below threshold ({config.min_allowed_rt_ms} ms), used only for final_clean exports: {int(participant_qc['min_rt_below_threshold_flag'].sum())}",
        f"Manual fraud flags: {int(participant_qc['manual_fraud_cluster_flag'].sum())}",
        f"Strict-only additional exclusions: {int((participant_qc['strict_exclusion_flag'] & ~participant_qc['hard_exclusion_flag']).sum())}",
        f"Behavioral hard flags: {int(participant_qc['bot_behavior_hard_flag'].sum())}",
        f"Behavioral soft-flagged rows: {int((participant_qc['bot_behavior_soft_flag_count'] > 0).sum())}",
        f"Ultra-clean additional exclusions: {int(participant_qc['ultra_clean_additional_exclusion_flag'].sum())}",
        "",
        "## Soft-flag totals",
        f"Repeat severe count == 1: {int(participant_qc['repeat_soft_flag'].sum())}",
        f"Fast participant flag: {int(participant_qc['fast_participant_flag'].sum())}",
        f"Hidden ratio flag: {int(participant_qc['hidden_ratio_flag'].sum())}",
        f"Hidden duration flag: {int(participant_qc['hidden_duration_flag'].sum())}",
        f"Low variance flag: {int(participant_qc['variance_low_flag'].sum())}",
        f"Pattern cycle flag: {int(participant_qc['pattern_cycle_flag'].sum())}",
        f"Pattern bigram flag: {int(participant_qc['pattern_bigram_flag'].sum())}",
        f"Pattern low-complexity flag: {int(participant_qc['pattern_low_complexity_flag'].sum())}",
        f"Pattern monotonic-walk flag: {int(participant_qc['pattern_template_walk_flag'].sum())}",
        f"Pseudo-random flag: {int(participant_qc['pseudo_random_flag'].sum())}",
        f"Batch high-similarity flag: {int(participant_qc['batch_high_similarity_flag'].sum())}",
        f"Batch exact-match flag: {int(participant_qc['batch_exact_match_flag'].sum())}",
        f"Batch cluster flag: {int(participant_qc['batch_cluster_flag'].sum())}",
    ]

    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report_path


def run_cleaning_pipeline(
    wide_path: Path,
    long_path: Path,
    output_dir: Path,
    manual_fraud_path: Optional[Path] = None,
    manual_final_exclusion_path: Optional[Path] = None,
    config: Optional[CleaningConfig] = None,
) -> dict[str, Any]:
    config = config or CleaningConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_wide_df = _read_csv(wide_path)
    raw_long_df = _read_csv(long_path)
    wide_df = _prepare_wide_table(raw_wide_df)
    long_df = _prepare_long_table(raw_long_df, config)

    structural_metrics = _build_structural_metrics(long_df, config)
    long_with_repeats, repeat_metrics = _build_repeat_metrics(long_df, config)
    response_metrics = _build_condition_variance_flags(_build_response_metrics(long_df), config)
    rt_metrics, usable_trials, fast_trial_flags = _build_rt_metrics(long_df, config)
    behavioral_metrics = _flag_behavioral_bot_metrics(
        _build_behavioral_sequence_metrics(long_df, config).merge(
            _build_consensus_and_peer_metrics(long_df, config),
            on=["session_id", "condition"],
            how="outer",
        ),
        config,
    )
    long_enriched = long_with_repeats.merge(fast_trial_flags, on=["session_id", "trial_index_num"], how="left")
    long_enriched["fast_trial_flag"] = long_enriched["fast_trial_flag"].fillna(False)

    manual_fraud_flags = _load_manual_fraud_flags(manual_fraud_path)
    manual_final_exclusion_flags = _load_manual_final_exclusion_flags(manual_final_exclusion_path)
    participant_qc = _merge_participant_master(
        wide_df=wide_df,
        structural_metrics=structural_metrics,
        response_metrics=response_metrics,
        repeat_metrics=repeat_metrics,
        rt_metrics=rt_metrics,
        behavioral_metrics=behavioral_metrics,
        manual_fraud_flags=manual_fraud_flags,
        config=config,
    )

    participant_qc_path = output_dir / "participant_qc_master.csv"
    participant_qc.to_csv(participant_qc_path, index=False, encoding="utf-8-sig")

    hard_cleaned = participant_qc.loc[~participant_qc["hard_exclusion_flag"]].copy()
    hard_cleaned_path = output_dir / "participants_hard_cleaned.csv"
    hard_cleaned.to_csv(hard_cleaned_path, index=False, encoding="utf-8-sig")

    strict_cleaned = participant_qc.loc[~participant_qc["strict_exclusion_flag"]].copy()
    strict_cleaned_path = output_dir / "participants_strict_cleaned.csv"
    strict_cleaned.to_csv(strict_cleaned_path, index=False, encoding="utf-8-sig")

    suspicious_review = participant_qc.loc[participant_qc["suspicious_behavior_review_flag"]].copy()
    suspicious_review_path = output_dir / "suspicious_participants_review.csv"
    suspicious_review.to_csv(suspicious_review_path, index=False, encoding="utf-8-sig")

    final_clean_participants = strict_cleaned.loc[~strict_cleaned["min_rt_below_threshold_flag"]].copy()
    final_clean_participants = final_clean_participants.merge(manual_final_exclusion_flags, on="session_id", how="left")
    final_clean_participants["manual_final_exclusion_flag"] = (
        final_clean_participants["manual_final_exclusion_flag"].astype("boolean").fillna(False).astype(bool)
    )
    final_clean_participants = final_clean_participants.loc[~final_clean_participants["manual_final_exclusion_flag"]].copy()
    final_clean_session_ids = final_clean_participants["session_id"].astype(str)
    final_clean_wide = raw_wide_df.loc[raw_wide_df["session_id"].astype(str).isin(final_clean_session_ids)].copy()
    final_clean_long = raw_long_df.loc[raw_long_df["session_id"].astype(str).isin(final_clean_session_ids)].copy()

    final_clean_wide_csv_path = output_dir / "final_clean_wide.csv"
    final_clean_wide_xlsx_path = output_dir / "final_clean_wide.xlsx"
    final_clean_long_csv_path = output_dir / "final_clean_long.csv"
    final_clean_long_xlsx_path = output_dir / "final_clean_long.xlsx"

    final_clean_wide.to_csv(final_clean_wide_csv_path, index=False, encoding="utf-8-sig")
    final_clean_wide.to_excel(final_clean_wide_xlsx_path, index=False)
    final_clean_long.to_csv(final_clean_long_csv_path, index=False, encoding="utf-8-sig")
    final_clean_long.to_excel(final_clean_long_xlsx_path, index=False)

    ultra_clean_participants = final_clean_participants.loc[~final_clean_participants["ultra_clean_additional_exclusion_flag"]].copy()
    ultra_clean_session_ids = ultra_clean_participants["session_id"].astype(str)
    ultra_clean_wide = raw_wide_df.loc[raw_wide_df["session_id"].astype(str).isin(ultra_clean_session_ids)].copy()
    ultra_clean_long = raw_long_df.loc[raw_long_df["session_id"].astype(str).isin(ultra_clean_session_ids)].copy()

    ultra_cleaned_path = output_dir / "participants_ultra_cleaned.csv"
    ultra_cleaned_path_xlsx = output_dir / "participants_ultra_cleaned.xlsx"
    ultra_clean_wide_csv_path = output_dir / "final_ultra_clean_wide.csv"
    ultra_clean_wide_xlsx_path = output_dir / "final_ultra_clean_wide.xlsx"
    ultra_clean_long_csv_path = output_dir / "final_ultra_clean_long.csv"
    ultra_clean_long_xlsx_path = output_dir / "final_ultra_clean_long.xlsx"

    ultra_clean_participants.to_csv(ultra_cleaned_path, index=False, encoding="utf-8-sig")
    ultra_clean_participants.to_excel(ultra_cleaned_path_xlsx, index=False)
    ultra_clean_wide.to_csv(ultra_clean_wide_csv_path, index=False, encoding="utf-8-sig")
    ultra_clean_wide.to_excel(ultra_clean_wide_xlsx_path, index=False)
    ultra_clean_long.to_csv(ultra_clean_long_csv_path, index=False, encoding="utf-8-sig")
    ultra_clean_long.to_excel(ultra_clean_long_xlsx_path, index=False)

    long_enriched_path = output_dir / "trials_enriched.csv"
    long_enriched.to_csv(long_enriched_path, index=False, encoding="utf-8-sig")

    report_path = _write_summary_report(output_dir, participant_qc, config)

    return {
        "participant_qc_path": participant_qc_path,
        "hard_cleaned_path": hard_cleaned_path,
        "strict_cleaned_path": strict_cleaned_path,
        "suspicious_review_path": suspicious_review_path,
        "final_clean_wide_csv_path": final_clean_wide_csv_path,
        "final_clean_wide_xlsx_path": final_clean_wide_xlsx_path,
        "final_clean_long_csv_path": final_clean_long_csv_path,
        "final_clean_long_xlsx_path": final_clean_long_xlsx_path,
        "ultra_cleaned_path": ultra_cleaned_path,
        "ultra_cleaned_path_xlsx": ultra_cleaned_path_xlsx,
        "ultra_clean_wide_csv_path": ultra_clean_wide_csv_path,
        "ultra_clean_wide_xlsx_path": ultra_clean_wide_xlsx_path,
        "ultra_clean_long_csv_path": ultra_clean_long_csv_path,
        "ultra_clean_long_xlsx_path": ultra_clean_long_xlsx_path,
        "long_enriched_path": long_enriched_path,
        "summary_report_path": report_path,
        "participant_qc": participant_qc,
        "long_enriched": long_enriched,
        "usable_trials": usable_trials,
        "final_clean_wide": final_clean_wide,
        "final_clean_long": final_clean_long,
        "final_clean_participants": final_clean_participants,
        "ultra_clean_participants": ultra_clean_participants,
        "ultra_clean_wide": ultra_clean_wide,
        "ultra_clean_long": ultra_clean_long,
    }
