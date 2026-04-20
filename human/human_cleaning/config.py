from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for the human aggression-rating cleaning pipeline."""

    expected_trial_count: int = 80
    expected_repeat_count: int = 5
    min_repeat_gap: int = 25
    extreme_long_rt_ms: int = 180_000
    min_rt_usable_ratio: float = 0.80
    min_allowed_rt_ms: int = 800
    hidden_trial_ratio_threshold: float = 0.05
    total_hidden_duration_threshold_ms: int = 120_000
    fast_trial_quantile: float = 0.05
    fast_participant_quantile: float = 0.05
    fast_participant_extreme_quantile: float = 0.01
    low_variance_quantile: float = 0.05
    low_variance_extreme_quantile: float = 0.01
    strict_soft_flag_threshold: int = 2
    sequence_cycle_min_lag: int = 2
    sequence_cycle_max_lag: int = 8
    pattern_cycle_match_soft_threshold: float = 0.70
    pattern_cycle_match_hard_threshold: float = 0.85
    pattern_bigram_abs_floor: float = 0.18
    pattern_monotonic_run_abs_floor: float = 0.30
    low_complexity_quantile: float = 0.05
    low_complexity_extreme_quantile: float = 0.01
    high_entropy_quantile: float = 0.95
    low_consensus_corr_quantile: float = 0.05
    low_consensus_corr_extreme_quantile: float = 0.01
    peer_similarity_quantile: float = 0.995
    peer_exact_match_quantile: float = 0.995
    peer_similarity_abs_floor: float = 0.90
    peer_exact_match_abs_floor: float = 0.80
    peer_clone_hard_exact_match_threshold: float = 0.90
    peer_similarity_min_shared_items: int = 40
    peer_similarity_cluster_min_neighbors: int = 2
    ultra_clean_soft_flag_threshold: int = 2
    repeat_severe_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "attack_3pt": 2.0,
            "attack_7pt_likert": 4.0,
            "attack_slider_0_100": 35.0,
        }
    )
    numeric_trial_columns: tuple[str, ...] = (
        "trial_index",
        "response_value",
        "repeat_source_trial_index",
        "rt_first_ms",
        "rt_submit_ms",
        "revisit_count",
        "backtrack_count",
        "page_hidden_count",
        "page_hidden_duration_ms",
    )
    wide_identity_columns: tuple[str, ...] = (
        "session_id",
        "final_participant_number",
        "condition",
        "name",
        "phone",
        "is_internal_student",
        "student_id",
        "alipay_account",
        "gender",
        "age_range",
        "education",
        "income_range",
    )


DEFAULT_WIDE_PATH = Path("attack_completed_participants_wide.csv")
DEFAULT_LONG_PATH = Path("attack_completed_trials_long.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MANUAL_FRAUD_PATH = Path("manual_fraud_session_ids.csv")
DEFAULT_MANUAL_FINAL_EXCLUSION_PATH = Path("manual_final_session_ids.csv")
