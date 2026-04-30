# Paper-style Results outline (ΔF-M as the single spine)

> Replaces the 1a / 1b / 2a thesis-style split.
> Five thematic sections, each integrates LLM and human evidence.
> Three-point rating is the anchor; 7-point and slider serve as
> measurement-condition probes, not parallel main analyses.

---

## R1. Setting the panel: nine LLMs differ in basic Chinese hate detection, motivating a minimal-pair gender-swap design

**Claim.** Before testing gender bias, we establish that the nine LLMs
differ substantially in Chinese hate-detection capability, and that
natural-language hate corpora confound target gender with content
extremity. Both observations motivate the minimal-pair gender-swap
design used throughout the rest of the paper.

**Evidence.**
- Bilingual hate-detection benchmark: F1 / Brier on 9 LLMs, zero-shot
  + CoT, English vs Chinese.
- Cross-language variance asymmetry: Chinese-side SD ≈ 3–4× English-side
  SD; variance ratio 10–15× (Bartlett, Levene reject).
- F1–Brier decoupling (e.g. Llama overconfident; Qwen better calibrated).
- HateXplain natural-data check: items targeting men receive *higher*
  attack ratings than items targeting women, but male-target items are
  far fewer and more extreme — confounded.

**Function in paper.** Two-paragraph preliminary. Justifies (a) why
nine specific LLMs are kept on the panel, (b) why we cannot read gender
bias off natural corpora and must use minimal pairs.

**Figures / tables.** F1+Brier 2×2 panel (existing); HateXplain
gender-target mean (existing); cross-language slope (existing).

---

## R2. Across all twelve rater groups, female-targeted items receive higher attack ratings (direction layer)

**Claim.** The female-minus-male attack-rating gap (ΔF-M) is positive
for every rater group we measured: nine LLMs × three scales = 27 cells,
and three human groups (overall, male, female participants) × three
scales = 9 cells. The direction is shared.

**Evidence — LLM side.**
- 27 of 27 cells with mean ΔF-M > 0.
- Cohen's d_z range 0.29 (DeepSeek-R1 / 3-point) to 1.02 (Gemma / slider).
- BH-FDR adjusted q < 10⁻⁷ in all 27 cells.

**Evidence — human side.**
- Human overall: ΔF-M_norm = 0.060 / 0.043 / 0.036 (3pt / 7pt / slider).
- Female participants: 0.090 / 0.066 / 0.056.
- Male participants: 0.030 / 0.021 / 0.017.
- Item-level direction breakdown: 69%–74% of items rated higher in the
  female version under each scale (overall human).

**Function in paper.** Establishes the existence of the effect *as a
shared phenomenon*, in the simplest possible language: "everyone, on
average, gives the female version a higher attack rating".

**Figures / tables.**
- Forest plot: 12 raters × 3 scales of d_z with 95% CI (combined).
- Companion table: mean_male, mean_female, mean_delta_norm, d_z,
  female-higher %, BH-q for all 36 cells.

---

## R3. The strength of ΔF-M is highly heterogeneous, and whether an LLM looks "more extreme" or "more moderate" depends on which human group is the reference (magnitude layer)

**Claim.** Direction is shared, but effect size is not. On the
3-point anchor scale, the nine LLMs span d_z = 0.29–0.65, the three
human groups span d_z = 0.22–0.66, and *the same LLM can look more
moderate or more extreme depending on which human group is used as
reference*. This is the paper's central comparative finding.

**Evidence (3-point anchor).**

| Reference group | d_z | Models above (more biased) | Models below (more moderate) |
|---|---|---|---|
| Male participants | 0.22 | All 9 | none |
| Overall humans | 0.62 | GPT-5.1 (0.65) | DeepSeek-R1 / V3.2 / GLM / Llama / Gemma / Claude / Qwen / Kimi |
| Female participants | 0.66 | none | All 9 |

Three-tier classification of LLMs at 3-point:

- **More moderate than overall humans (d_z ≤ 0.41):** DeepSeek-R1 0.29,
  DeepSeek-V3.2 0.29, GLM 0.30, Llama 0.38, Gemma 0.41.
- **Comparable to overall humans (d_z 0.50–0.57):** Claude 0.50,
  Qwen 0.55, Kimi 0.57.
- **At the female-participant ceiling (d_z 0.65):** GPT-5.1.
- **Above all human reference groups:** none on 3-point. (Compare R5,
  where the slider scale changes this.)

**Reframing for the reader.** "Are LLMs more or less biased than
humans?" has no single answer. Against male participants every LLM is
more biased; against female participants no LLM is. Against the overall
human aggregate, LLMs split roughly 5 / 3 / 1 between moderating,
matching, and saturating.

**Function in paper.** This is the section of the paper. The descriptive
heatmap is the headline figure.

**Figures / tables.**
- **Headline figure**: 12-rater × 3-scale d_z heatmap
  (`artifacts/human_model_dz_descriptive_comparison/fig1_dz_descriptive_heatmap.png`).
- Companion: model-minus-human d_z gap heatmap
  (`fig2_model_minus_human_dz_gap_heatmap.png`).
- Companion descriptive table: mean_male, mean_female, mean_delta,
  d_z, female_higher_pct for all 12 × 3 cells.

---

## R4. The gap concentrates on sexualization and appearance dimensions for both LLMs and humans (content-structure layer)

**Claim.** Both LLMs and human raters localize the bias to the same
content axes: sexualization-related and appearance-related attacks.
The bias is not diffuse; it is shaped.

**Evidence.**
- LLM dimension-level d_z heatmap: 9 models × 3 scales × 10 first-level
  domains. The two consistently darkest columns are 性化攻击
  (sexualization) and 外貌形象攻击 (appearance), across all 27 cells.
- Human dimension-level mean ΔF-M: same two columns are darkest for
  overall humans and especially for female participants. Male
  participants' bias is weaker overall but still concentrated on these
  two columns.

**Function in paper.** This section moves the paper from "LLMs and
humans are both biased and roughly comparable in size" to "LLMs and
humans are biased *in the same content territory*". This rules out a
'noisy alignment' interpretation: the agreement is structural.

**Figures / tables.**
- LLM-side dimension heatmap (existing
  `artifacts/model_by_level1_by_scale_dz_heatmap.png`).
- Human-side dimension heatmap (overall / male / female panels),
  needs to be regenerated using
  `artifacts/human_model_dz_descriptive_comparison/all_rater_item_deltas.csv`.

---

## R5. Rating granularity amplifies LLM bias but dampens human bias — measurement-induced human–LLM divergence

**Claim.** As the rating scale moves from 3-point → 7-point → slider,
LLM d_z generally *increases*, while human d_z stays flat or *decreases*.
This means the human–LLM gap is smallest at 3-point and largest at
slider. "Alignment" measured under one scale does not generalize.

**Evidence.**

| Rater | 3pt d_z | 7pt d_z | Slider d_z | Direction |
|---|---|---|---|---|
| Overall humans | 0.62 | 0.59 | 0.51 | ↓ monotonic |
| Male participants | 0.22 | 0.20 | 0.17 | ↓ monotonic |
| Female participants | 0.65 | 0.70 | 0.59 | ↑ peak / ↓ |
| Claude-4.5 | 0.50 | 0.69 | 0.75 | ↑ monotonic |
| DeepSeek-R1 | 0.29 | 0.42 | 0.51 | ↑ monotonic |
| DeepSeek-V3.2 | 0.29 | 0.37 | 0.46 | ↑ monotonic |
| GLM-4.6 | 0.30 | 0.35 | 0.42 | ↑ monotonic |
| Gemma-4-31B | 0.41 | 0.77 | 1.02 | ↑↑ extreme |
| GPT-5.1 | 0.65 | 0.84 | 0.71 | ↑ peak |
| Kimi-K2 | 0.57 | 0.51 | 0.59 | ≈ flat |
| Llama-4-Mav. | 0.38 | 0.48 | 0.41 | ≈ flat / peak |
| Qwen-2.5-72B | 0.55 | 0.58 | 0.69 | ↑ monotonic |

- Most LLMs amplify with finer scales; humans on average do not.
- Headline contrast: at 3-point, 5/9 LLMs are *below* overall humans;
  at slider, 5/9 LLMs are *above* overall humans (Claude, Gemma,
  GPT-5.1, Qwen, DeepSeek-R1).
- 2×3 RM-ANOVA on item-level normalized ΔF-M (participant gender ×
  scale, items as units): participant-gender main effect F(1,370)=111.17,
  p<.001, η²_G=.044; scale main effect F(2,740)=9.44, GG-corrected
  p<.001, η²_G=.007; interaction F(2,740)=1.79, p=.169 (n.s.).
- **Post-hoc (TODO)** — pairwise comparisons of (gender × scale) cells
  with Holm correction; expected to show that the gender main effect is
  significant within every scale and that the scale effect is small but
  asymmetric across rater groups.

**Function in paper.** This is the paper's strongest mechanistic
finding, because it shows that the apparent human–LLM agreement is
*measurement-conditional*. Coarse scales hide LLM amplification; fine
scales expose it.

**Figures / tables.**
- Interaction plot: x = scale (3pt / 7pt / slider), y = d_z, lines per
  rater (humans bold, LLMs grey, with selected models highlighted).
- ANOVA + post-hoc table.

---

## R6. Aggregate alignment does not transfer to item-level alignment — and the "most human-like" model is not stable across scale or human reference (item-level / ICC layer)

**Claim.** Even when LLM and human aggregate d_z values match (R3),
*which specific items* receive the largest gender gap is only weakly
shared. Item-level ICC is in the poor range across all 27 LLM × scale
pairings, and the top-1 "most human-like" model switches with both the
human reference group and the rating scale.

**Evidence.**
- 12-rater item-level ICC matrix on each scale.
- Top-1 ICC by (human reference × scale): five different models occupy
  the nine slots — no globally most-similar model exists.
- All ICCs remain in poor range (≤ 0.22).
- Cross-scale item-level Spearman ρ on ΔF-M:
  - Within humans (item-level): ρ = 0.04–0.13 across 9 (group × pair).
  - Within models (item-level): median ρ ≈ 0.08, range −0.10 to +0.31.
  - Aggregate level (10-domain vectors): ρ = 0.57–0.85 LLM-side; ρ =
    0.58–0.86 human-side.
- Conclusion: ΔF-M is a population-level / structural-level phenomenon
  shared between LLMs and humans, but no model reliably reproduces the
  *exact* item-by-item pattern of gender-bias localization.

**Function in paper.** Demoted from headline to a supplementary
diagnostic. Demonstrates that:
1. Reporting only ICC would mask the magnitude story (responding to
   the teacher's feedback that "ICC alone cannot reveal whether the
   gender-difference structure is consistent").
2. The aggregate / dimension / item three-tier conclusion is itself a
   finding: alignment is real at population level, illusory at item
   level.

**Figures / tables.**
- 12 × 12 item-level ICC matrix (3-point only as headline; 7pt and
  slider versions in supplementary).
- Aggregate vs item-level Spearman comparison strip plot.

---

## R7. Synthesis: human–LLM alignment on gender-targeted attack ratings is conditional on (rater reference, rating scale, level of analysis)

This becomes the closing paragraph of Results or the opening of Discussion.

- Shared: direction (R2), content structure (R4).
- Conditional: magnitude depends on which human group (R3); LLM–human
  divergence depends on rating granularity (R5); item-level localization
  is weak and rater-/scale-specific (R6).
- Therefore, no single number captures human–LLM alignment, and ICC
  alone — the most common alignment metric in the LLM-evaluation
  literature — would be misleading here.

---

## Outstanding analyses to run before writing

1. **Post-hoc on the 2×3 RM-ANOVA (R5).** Pairwise comparisons of the
   six (gender × scale) cells with Holm correction. Likely already in
   `human_model_dz_descriptive_comparison` or `analyze_gender_difference_pipeline`
   — verify or compute.

2. **Human-side dimension heatmap (R4).** Generate from
   `all_rater_item_deltas.csv` to mirror the existing
   `model_by_level1_by_scale_dz_heatmap`. Three panels: overall, male,
   female.

3. **12-rater item-level ICC matrix on 3-point (R6).** If the existing
   `human_model_delta_similarity*` files cover only model-vs-overall,
   extend to all 12 × 12 (or 12 × 9 model–human) on each scale.

4. **R5 interaction plot.** Build the d_z × scale line plot with
   12 lines, anchored on 3-point.

5. **R2 forest plot.** A single 12 × 3 d_z forest with CIs, sorted by
   3-point d_z within human/LLM blocks.

---

## How this addresses each teacher feedback point

| Teacher comment | Addressed in |
|---|---|
| 1. Make 1b's design intent explicit | R1 bridge paragraph |
| 2. Don't only report ICC | R3 (descriptive d_z) is headline; R6 (ICC) is supplementary |
| 3. Add the 12-rater descriptive d_z statistics | R3 (already produced by Codex) |
| 4. Post-hoc after 2×3 ANOVA | R5 outstanding analysis #1 |
| 5. Direction and effect size matter; do not over-interpret tiny effects | R2 (direction), R3 (magnitude); η²_G values reported in R5 |
| 6. Core story is human vs LLM (consistent? more moderate? more extreme? changes by scale?) | R3 reframing + R5 measurement-induced divergence |
| 7. Pick the cleanest level; don't force item-level if noisy | Aggregate / dimension lead; item-level ICC demoted to R6 |
| 8. Central descriptive page | R3 with `fig1_dz_descriptive_heatmap` |
