"""M3-Bench answering and judging prompts.

The judge prompt, the control-loop system prompt and the final-round hint are copied verbatim from
the official M3-Agent implementation
(https://github.com/ByteDance-Seed/m3-agent, Apache-2.0, Copyright 2025 Bytedance Ltd.):
``mmagent/prompts.py::prompt_agent_verify_answer_referencing`` and ``m3_agent/control.py``.
They are kept verbatim so scores stay comparable with the paper; confirm that a deviation is
acceptable before changing them.

Only change: ``CONTROL_INSTRUCTION`` drops the two sentences about looking up character IDs and
names. The official memory graph assigns ``<character_i>`` entities to faces/voices; our video
memory has no such entities, and keeping those sentences would push the model into searches that
can never hit.
"""

from __future__ import annotations

# mmagent/prompts.py::prompt_agent_verify_answer_referencing (verbatim)
JUDGE_PROMPT = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
\t•\tDo not require exact wording or matching structure.
\t•\tSemantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
\t•\tOnly return "Yes" or "No", with no additional explanation or formatting.

Input fields:
\t•\tquestion: the question asked
\t•\tground_truth_answer: the correct answer
\t•\tagent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
\t•\tquestion: {question}
\t•\tground_truth_answer: {ground_truth_answer}
\t•\tagent_answer: {agent_answer}

Output ('Yes' or 'No'):"""  # noqa: E501

# m3_agent/control.py::system_prompt (verbatim)
CONTROL_SYSTEM_PROMPT = (
    "You are given a question and some relevant knowledge. Your task is to reason about "
    "whether the provided knowledge is sufficient to answer the question. If it is "
    "sufficient, output [Answer] followed by the answer. If it is not sufficient, output "
    "[Search] and generate a query that will be encoded into embeddings for a vector "
    "similarity search. The query will help retrieve additional information from a "
    "memory bank.\n\nQuestion: {question}"
)

# m3_agent/control.py::instruction, minus the two character-ID sentences (see module doc)
CONTROL_INSTRUCTION = """

Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query that would help retrieve the missing information. The search {content} needs to be different from the previous.
If the answer can be derived from the provided knowledge, the {content} is the specific answer to the question."""  # noqa: E501

# m3_agent/control.py final-round suffix (verbatim)
FINAL_ROUND_HINT = (
    "\n(The Action of this round must be [Answer]. If there is insufficient information, "
    "you can make reasonable guesses.)"
)

# m3_agent/control.py suffix for an empty search (verbatim)
EMPTY_SEARCH_HINT = (
    "\n(The search result is empty. Please try searching from another perspective.)"
)

# m3_agent/control.py::pattern (re.DOTALL, applied after the last </think>)
ACTION_PATTERN = r"Action: \[(.*)\].*Content: (.*)"


def judge_prompt(question: str, ground_truth: str, agent_answer: str) -> str:
    return JUDGE_PROMPT.format(
        question=question,
        ground_truth_answer=ground_truth,
        agent_answer=agent_answer,
    )
