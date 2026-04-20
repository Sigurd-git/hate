"""Prompt templates for hate-speech evaluation tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

PromptParadigm = Literal["zero_shot", "few_shot", "chain_of_thought"]


@dataclass(frozen=True)
class PromptTemplate:
    """Container mapping a prompt paradigm to its formatted template."""

    language: str
    paradigm: PromptParadigm
    template: str


ZERO_SHOT_ZH = (
    "请根据以下定义评估该句子是否属于仇恨言论，并为其攻击性/仇恨强度在 0–5 分范围内评分。"
    "请严格依照以下仇恨言论定义进行判断，不要使用任何其他标准。\n"
    "【仇恨言论定义】仇恨言论是指：基于被保护特征（如性别、性别认同、性取向、种族、民族、国籍、宗教、残障等），"
    "对某个群体或其成员表达敌意、侮辱、贬低、非人化，或鼓励/正当化伤害的言论。\n"
    "评分说明：0 = 明确不是仇恨言论；1–5 = 属于仇恨言论，分值越高表示攻击性/仇恨程度越强"
    "（1 = 轻微贬义；3 = 中度敌意或群体泛化；5 = 强烈侮辱、非人化或带伤害意图）。\n"
    "请严格只输出一个 0–5 的数字，不要输出任何其他内容。句子：\n\"{text}\"\n答案："
)

ZERO_SHOT_EN = (
    "Evaluate whether the following sentence constitutes hate speech according to the definition below. "
    "Follow the definition strictly and do not apply any additional criteria.\n"
    "[Definition of hate speech] Hate speech refers to expressions of hostility, insult, dehumanization, "
    "or justification of harm toward a protected group or its members based on characteristics such as gender, "
    "gender identity, sexual orientation, race, ethnicity, nationality, religion, or disability.\n"
    "Scoring guidelines: 0 = clearly not hate speech; 1–5 = hate speech, with higher numbers indicating stronger "
    "aggression or hostility (1 = mild derogatory expression; 3 = moderate hostility or group-based generalization; "
    "5 = strong insult, dehumanization, or implication of harm).\n"
    "Output only a single number (0–5). Sentence:\n\"{text}\"\nAnswer:"
)

FEW_SHOT_ZH = (
    "请阅读以下示例，理解 0–5 分的仇恨强度定义，然后为新的句子评分。\n"
    "示例 1：\n句子：\"我不喜欢他们的口音，但那也没什么。\"\n评分：1\n"
    "示例 2：\n句子：\"这些移民抢走了我们的工作，他们应该被赶走。\"\n评分：4\n"
    "示例 3：\n句子：\"女医生比男医生差远了，根本不该治病。\"\n评分：5\n"
    "现在请为以下句子评分，只输出一个 0–5 的数字。\n句子：\n\"{text}\"\n答案："
)

FEW_SHOT_EN = (
    "Study the following examples to understand the 0–5 hate intensity scale, then score the new sentence.\n"
    "Example 1:\nSentence: \"I do not agree with their customs, but I respect them.\"\nScore: 1\n"
    "Example 2:\nSentence: \"These immigrants are ruining our country; they should be expelled.\"\nScore: 4\n"
    "Example 3:\nSentence: \"Women are too irrational to vote and should stay silent.\"\nScore: 5\n"
    "Now assign a score to the sentence below and output a single number between 0 and 5 only.\nSentence:\n\"{text}\"\nAnswer:"
)

COT_ZH = (
    "请根据以下定义评估该句子是否属于仇恨言论，并为其攻击性/仇恨强度在 0–5 分范围内评分。"
    "请严格依照以下仇恨言论定义进行判断，不要使用任何其他标准。\n"
    "【仇恨言论定义】仇恨言论是指：基于被保护特征（如性别、性别认同、性取向、种族、民族、国籍、宗教、残障等），"
    "对某个群体或其成员表达敌意、侮辱、贬低、非人化，或鼓励/正当化伤害的言论。\n"
    "评分说明：0 = 明确不是仇恨言论；1–5 = 属于仇恨言论，分值越高表示攻击性/仇恨程度越强"
    "（1 = 轻微贬义；3 = 中度敌意或群体泛化；5 = 强烈侮辱、非人化或带伤害意图）。\n"
    "最后请在JSON中给我输出你评分的简洁理由（reason）以及 score（0–5）。句子：\n\"{text}\""
)

COT_EN = (
    "Evaluate whether the following sentence constitutes hate speech according to the definition below. "
    "Follow the definition strictly and do not apply any additional criteria.\n"
    "[Definition of hate speech] Hate speech refers to expressions of hostility, insult, dehumanization, "
    "or justification of harm toward a protected group or its members based on characteristics such as gender, "
    "gender identity, sexual orientation, race, ethnicity, nationality, religion, or disability.\n"
    "Scoring guidelines: 0 = clearly not hate speech; 1–5 = hate speech, with higher numbers indicating stronger "
    "aggression or hostility (1 = mild derogatory expression; 3 = moderate hostility or group-based generalization; "
    "5 = strong insult, dehumanization, or implication of harm).\n"
    "Return the reasoning inside the JSON `reason` field and the numeric result in `score`. Sentence:\n\"{text}\""
)

PROMPTS: Dict[str, Dict[PromptParadigm, PromptTemplate]] = {
    "zh": {
        "zero_shot": PromptTemplate(language="zh", paradigm="zero_shot", template=ZERO_SHOT_ZH),
        "few_shot": PromptTemplate(language="zh", paradigm="few_shot", template=FEW_SHOT_ZH),
        "chain_of_thought": PromptTemplate(language="zh", paradigm="chain_of_thought", template=COT_ZH),
    },
    "en": {
        "zero_shot": PromptTemplate(language="en", paradigm="zero_shot", template=ZERO_SHOT_EN),
        "few_shot": PromptTemplate(language="en", paradigm="few_shot", template=FEW_SHOT_EN),
        "chain_of_thought": PromptTemplate(language="en", paradigm="chain_of_thought", template=COT_EN),
    },
}

LANGUAGES = tuple(PROMPTS.keys())
PARADIGMS = ("zero_shot", "chain_of_thought")


def render_prompt(language: str, paradigm: PromptParadigm, text: str) -> str:
    """Return a formatted prompt for the specified language and paradigm."""
    lang_key = language.lower()
    template = PROMPTS[lang_key][paradigm].template
    return template.format(text=text)

AttackPromptMode = Literal[
    "attack_3pt",
    "attack_7pt_likert",
    "attack_slider_0_100",
]

ATTACK_3PT_ZH = (
    "你现在要模拟这个网页中的人类被试，对一句中文句子的“攻击性”进行评分。\n\n"
    "请严格按照指定的评分量表作答，不要发明新的标准，不要输出解释，不要输出多余内容。\n"
    "你的评分要尽量贴近普通中文被试在该网页实验中的直觉判断。\n\n"
    "评分量表：attack_3pt\n"
    "- 0 = 无攻击性\n"
    "- 1 = 轻度攻击性\n"
    "- 2 = 严重攻击性\n\n"
    "任务要求：\n"
    "- 只根据句子本身判断其攻击性程度\n"
    "- 不要解释原因\n"
    "- 不要输出除分数外的任何文字\n"
    "- 输出必须符合指定量表\n\n"
    "当前量表：attack_3pt\n"
    "待评分句子：{text}"
)

ATTACK_7PT_LIKERT_ZH = (
    "你现在要模拟这个网页中的人类被试，对一句中文句子的“攻击性”进行评分。\n\n"
    "请严格按照指定的评分量表作答，不要发明新的标准，不要输出解释，不要输出多余内容。\n"
    "你的评分要尽量贴近普通中文被试在该网页实验中的直觉判断。\n\n"
    "评分量表：attack_7pt_likert\n"
    "- 0 = 无攻击性\n"
    "- 1 = 很轻微攻击性\n"
    "- 2 = 轻度攻击性\n"
    "- 3 = 中等攻击性\n"
    "- 4 = 明显攻击性\n"
    "- 5 = 强烈攻击性\n"
    "- 6 = 极端攻击性\n\n"
    "任务要求：\n"
    "- 只根据句子本身判断其攻击性程度\n"
    "- 不要解释原因\n"
    "- 不要输出除分数外的任何文字\n"
    "- 输出必须符合指定量表\n\n"
    "当前量表：attack_7pt_likert\n"
    "待评分句子：{text}"
)

ATTACK_SLIDER_0_100_ZH = (
    "你现在要模拟这个网页中的人类被试，对一句中文句子的“攻击性”进行评分。\n\n"
    "请严格按照指定的评分量表作答，不要发明新的标准，不要输出解释，不要输出多余内容。\n"
    "你的评分要尽量贴近普通中文被试在该网页实验中的直觉判断。\n\n"
    "评分量表：attack_slider_0_100\n"
    "- 0 = 无攻击性\n"
    "- 100 = 极端攻击性\n"
    "- 其余数值表示从“无攻击性”到“极端攻击性”的连续程度\n\n"
    "任务要求：\n"
    "- 只根据句子本身判断其攻击性程度\n"
    "- 不要解释原因\n"
    "- 不要输出除分数外的任何文字\n"
    "- 输出必须符合指定量表\n\n"
    "当前量表：attack_slider_0_100\n"
    "待评分句子：{text}"
)

ATTACK_3PT_EN = ATTACK_3PT_ZH
ATTACK_7PT_LIKERT_EN = ATTACK_7PT_LIKERT_ZH
ATTACK_SLIDER_0_100_EN = ATTACK_SLIDER_0_100_ZH

ATTACK_3PT_PROMPTS: Dict[str, Dict[PromptParadigm, PromptTemplate]] = {
    "zh": {
        "zero_shot": PromptTemplate(language="zh", paradigm="zero_shot", template=ATTACK_3PT_ZH),
        "few_shot": PromptTemplate(language="zh", paradigm="few_shot", template=ATTACK_3PT_ZH),
        "chain_of_thought": PromptTemplate(language="zh", paradigm="chain_of_thought", template=ATTACK_3PT_ZH),
    },
    "en": {
        "zero_shot": PromptTemplate(language="en", paradigm="zero_shot", template=ATTACK_3PT_EN),
        "few_shot": PromptTemplate(language="en", paradigm="few_shot", template=ATTACK_3PT_EN),
        "chain_of_thought": PromptTemplate(language="en", paradigm="chain_of_thought", template=ATTACK_3PT_EN),
    },
}

ATTACK_7PT_LIKERT_PROMPTS: Dict[str, Dict[PromptParadigm, PromptTemplate]] = {
    "zh": {
        "zero_shot": PromptTemplate(language="zh", paradigm="zero_shot", template=ATTACK_7PT_LIKERT_ZH),
        "few_shot": PromptTemplate(language="zh", paradigm="few_shot", template=ATTACK_7PT_LIKERT_ZH),
        "chain_of_thought": PromptTemplate(language="zh", paradigm="chain_of_thought", template=ATTACK_7PT_LIKERT_ZH),
    },
    "en": {
        "zero_shot": PromptTemplate(language="en", paradigm="zero_shot", template=ATTACK_7PT_LIKERT_EN),
        "few_shot": PromptTemplate(language="en", paradigm="few_shot", template=ATTACK_7PT_LIKERT_EN),
        "chain_of_thought": PromptTemplate(language="en", paradigm="chain_of_thought", template=ATTACK_7PT_LIKERT_EN),
    },
}

ATTACK_SLIDER_0_100_PROMPTS: Dict[str, Dict[PromptParadigm, PromptTemplate]] = {
    "zh": {
        "zero_shot": PromptTemplate(language="zh", paradigm="zero_shot", template=ATTACK_SLIDER_0_100_ZH),
        "few_shot": PromptTemplate(language="zh", paradigm="few_shot", template=ATTACK_SLIDER_0_100_ZH),
        "chain_of_thought": PromptTemplate(language="zh", paradigm="chain_of_thought", template=ATTACK_SLIDER_0_100_ZH),
    },
    "en": {
        "zero_shot": PromptTemplate(language="en", paradigm="zero_shot", template=ATTACK_SLIDER_0_100_EN),
        "few_shot": PromptTemplate(language="en", paradigm="few_shot", template=ATTACK_SLIDER_0_100_EN),
        "chain_of_thought": PromptTemplate(language="en", paradigm="chain_of_thought", template=ATTACK_SLIDER_0_100_EN),
    },
}

ATTACK_PROMPT_VARIANTS: Dict[AttackPromptMode, Dict[str, Dict[PromptParadigm, PromptTemplate]]] = {
    "attack_3pt": ATTACK_3PT_PROMPTS,
    "attack_7pt_likert": ATTACK_7PT_LIKERT_PROMPTS,
    "attack_slider_0_100": ATTACK_SLIDER_0_100_PROMPTS,
}


def render_attack_prompt(
    language: str,
    paradigm: PromptParadigm,
    text: str,
    prompt_mode: AttackPromptMode = "attack_7pt_likert",
) -> str:
    """Return a formatted attackiveness prompt for the specified language and paradigm."""
    lang_key = language.lower()
    prompt_templates = ATTACK_PROMPT_VARIANTS[prompt_mode]
    normalized_lang = lang_key if lang_key in prompt_templates else "zh"
    template = prompt_templates[normalized_lang][paradigm].template
    return template.format(text=text)


__all__ = [
    "PromptParadigm",
    "PromptTemplate",
    "AttackPromptMode",
    "LANGUAGES",
    "PARADIGMS",
    "render_prompt",
    "render_attack_prompt",
]
