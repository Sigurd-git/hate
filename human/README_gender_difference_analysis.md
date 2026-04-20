# Human gender-difference analysis

This analysis mirrors the earlier LLM group-swap workflow, but uses cleaned human ratings and splits results by the three rating conditions:

- `attack_3pt`
- `attack_7pt_likert`
- `attack_slider_0_100`

## Default input

The script reads:

- `outputs/final_clean_long.csv`

By default it excludes repeat trials (`is_repeat_trial == TRUE`).

## Pairing assumption

Within each condition, `template_id` is treated as the shared underlying sentence template and `shown_version` is treated as the male vs female surface form of that same template.

The sentence-level comparison therefore works on **template-level mean human ratings**:

- female template mean minus male template mean
- one paired row per `condition + template_id`

## Run

```bash
cd /home/sigurd/.openclaw/workspaces/engineer/projects/hate/human
uv run python analyze_gender_difference_pipeline.py
```

To analyze a different cleaned long file or output to a custom folder:

```bash
uv run python analyze_gender_difference_pipeline.py \
  --input-long-path outputs_slider35/final_clean_long.csv \
  --output-dir outputs_slider35/final_clean_long_gender_difference_analysis
```

## Outputs

The script writes a new directory named after the input stem, for example:

- `outputs/final_clean_long_gender_difference_analysis/`

Key outputs include:

- `template_version_scores.csv`
- `same_sentence_gender_differences.csv`
- `gender_score_long.csv`
- `topic_rankings.csv`
- `stats_overall.csv`
- `stats_directionality.csv`
- `stats_level1.csv`
- `human_gender_difference_analysis.xlsx`
- `analysis_summary.md`
- `figures/fig1_paired_scores.png`
- `figures/fig2_difference_distribution.png`
- `figures/fig3_directionality.png`
- `figures/fig4_level1_forest.png`

The `figures/panels/` subdirectory also stores panel-level exports.
