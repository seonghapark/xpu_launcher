# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenizer helper for gemma SFT.

The base ``google/gemma-7b`` tokenizer does not ship with a chat template
(only the ``-it`` variant does). We install the standard Gemma chat template
so downstream code -- SFT tokenization here and inference later -- can rely
on ``apply_chat_template``.
"""

from transformers import AutoTokenizer, PreTrainedTokenizerBase


# The chat template used by gemma-7b-it. Kept in sync so that models fine-tuned
# with this template are directly compatible with existing gemma serving code.
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'assistant') %}"
    "{% set role = 'model' %}"
    "{% else %}"
    "{% set role = message['role'] %}"
    "{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + "
    "'<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\n' }}"
    "{% endif %}"
)


def build_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Gemma has EOS but no dedicated PAD token; reuse EOS for right-padding.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SFT uses right-padding (labels align with input_ids).
    tokenizer.padding_side = "right"

    if tokenizer.chat_template is None:
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE

    return tokenizer
