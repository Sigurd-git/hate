# SYY-style comments on abstract, introduction, and research background

Scope: `slides/paper.tex`, abstract through Section 2.

## Overall diagnosis

The current structure is closer to CHB after separating `Introduction` from `Research background and measurement framework`. The main remaining SYY-style risk is not length, but whether each paragraph makes a necessary step in the chain:

`online hate context -> Chinese gendered attacks -> LLMs as evaluators -> natural-corpus limitation -> gender-reversal minimal pairs -> decomposed human-LLM similarity`.

## Major comments

1. **Abstract opening**

   Previous version began with LLMs evaluating offensive speech, then moved quickly to gendered attack ratings. The link between offensive-speech evaluation and subjective social judgement was implicit.

   SYY-style diagnosis: the key variable `target gender` appears before the reader has been told why this task requires integrating social-category cues.

   Revision made: the abstract now first states that offensive-speech evaluation is a subjective social-judgement task where ratings may depend on hostile content and social-category cues.

2. **Chinese abstract wording**

   Previous version used `生产级大语言模型`. This is understandable but not stable as an academic term in Chinese.

   SYY-style diagnosis: wording is too close to product language and may distract from the scientific problem.

   Revision made: changed to `主流大语言模型`.

3. **Introduction design overview**

   The Introduction now contains study design information, including 371 pairs, 9 LLMs, 779 sessions, and three response formats.

   SYY-style comment: this is acceptable because each method detail corresponds to a gap, but it should remain a design overview. Full construction details belong in Methods.

4. **Section 2 ending**

   Previous version ended with implications for model selection and future model development.

   SYY-style diagnosis: this is a contribution/discussion-level claim, not a background-section conclusion.

   Revision made: the Section 2 ending now only explains why the paper analyzes direction, magnitude, domain structure, scale response, and item-level correspondence.

## Sentence-level points still worth watching

1. `The response format is not a neutral detail in this comparison.`

   This sentence is rhetorically effective, but if the advisor prefers a more conservative style, revise to:

   `Response format is part of the measurement design in this comparison.`

2. `judgement structure`

   This is acceptable if the paper operationalizes it through specific indicators. Avoid using it later as if it means an internal cognitive mechanism.

3. `alignment`

   Keep using `alignment` only after specifying the level: direction, magnitude, domain structure, scale response, or item-level correspondence.

