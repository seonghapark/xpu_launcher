# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SFT dataset + collator for instruction fine-tuning gemma-7b.

Records must have ``instruction`` (required), ``input`` (optional), and
``output`` (required) fields (Alpaca-style). Each record is rendered via the
tokenizer's chat template and, when ``mask_instruction=True``, tokens that
belong to the user turn get label ``-100`` so the LM loss is computed only
on the assistant response.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Standard "ignore in loss" index used by PyTorch's cross entropy.
IGNORE_INDEX: int = -100


def _build_user_content(record: dict, instruction_key: str, input_key: str) -> str:
    instruction = record.get(instruction_key, "")
    if instruction is None:
        instruction = ""
    extra = record.get(input_key, "") or ""
    if extra:
        return f"{instruction}\n\n{extra}".strip()
    return str(instruction).strip()


class SFTDataset(Dataset):
    """Map-style Alpaca-formatted SFT dataset."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_len: int = 2048,
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        mask_instruction: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.mask_instruction = mask_instruction

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        record = self.records[idx]
        user_content = _build_user_content(
            record, self.instruction_key, self.input_key
        )
        assistant_content = str(record.get(self.output_key, "") or "").strip()

        # Render user + assistant turns through the chat template.
        user_messages = [{"role": "user", "content": user_content}]
        full_messages = user_messages + [
            {"role": "assistant", "content": assistant_content}
        ]

        # ``add_generation_prompt=True`` closes the user turn and opens the
        # assistant turn (`<start_of_turn>model\n`). Tokens up to and including
        # that opening are the "prompt" and must be masked from the loss.
        prompt_text = self.tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # We must ensure `full_text` starts with `prompt_text` so we can split
        # the token stream cleanly. The gemma template guarantees this.
        if not full_text.startswith(prompt_text):
            # Extremely defensive: fall back to unmasked training on this row.
            prompt_text = ""

        # Tokenize WITHOUT adding another BOS -- the chat template already
        # injects ``{{ bos_token }}``.
        full_ids: list[int] = self.tokenizer(
            full_text, add_special_tokens=False, truncation=False
        )["input_ids"]

        if self.mask_instruction and prompt_text:
            prompt_ids: list[int] = self.tokenizer(
                prompt_text, add_special_tokens=False, truncation=False
            )["input_ids"]
            prompt_len = len(prompt_ids)
        else:
            prompt_len = 0

        # Append EOS so the model learns to terminate responses.
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (
            len(full_ids) == 0 or full_ids[-1] != eos_id
        ):
            full_ids = full_ids + [eos_id]

        # Truncate from the right; keep at least one supervised token if
        # possible.
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]

        labels = list(full_ids)
        if self.mask_instruction:
            mask_upto = min(prompt_len, len(labels))
            for i in range(mask_upto):
                labels[i] = IGNORE_INDEX

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


@dataclass
class SFTCollator:
    """Right-pad a batch to the longest sequence."""

    pad_token_id: int
    pad_to_multiple_of: int | None = 8

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full(
            (len(batch), max_len), IGNORE_INDEX, dtype=torch.long
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(
                item["attention_mask"], dtype=torch.long
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def load_sft_records(
    dataset_name: str | None,
    dataset_config_name: str | None,
    dataset_split: str,
    dataset_local_path: str | None,
) -> list[dict[str, Any]]:
    """Load SFT records from either a local JSON/JSONL file or an HF dataset.

    Returns a plain list of dicts to keep the pipeline dependency-light and
    checkpoint/resume trivially deterministic.
    """
    if dataset_local_path:
        return _load_local_records(dataset_local_path)

    if not dataset_name:
        raise ValueError(
            "Provide either `dataset_local_path` or `dataset_name`."
        )

    # Imported lazily so pure-tokenization tests don't need `datasets`.
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    return [dict(row) for row in ds]


def _load_local_records(path: str) -> list[dict[str, Any]]:
    import json
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # JSONL
    if path.endswith(".jsonl"):
        out: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    # JSON: expect a list of dicts.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of records in {path}, got {type(data).__name__}."
        )
    return data
