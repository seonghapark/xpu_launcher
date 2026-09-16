from dataclasses import dataclass
from typing import Literal

from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer

from .sptoken import SPTokenizer


class EZPZTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        backend: Literal[
            "sptoken",
            "sentencepiece",
            "sp",
            "spm",
            "hf",
            "huggingface",
        ] = "hf"

    def __init__(self, config: Config, *, tokenizer_path: str):
        super().__init__()
        backend = config.backend.lower().strip()

        if backend in {"sptoken", "sentencepiece", "sp", "spm"}:
            self._tokenizer = SPTokenizer(tokenizer_path)
        elif backend in {"hf", "huggingface"}:
            self._tokenizer = HuggingFaceTokenizer.Config().build(
                tokenizer_path=tokenizer_path
            )
        else:
            raise ValueError(
                f"Unknown tokenizer backend '{config.backend}'. "
                "Choose one of: sptoken, sentencepiece, sp, spm, hf, huggingface"
            )

        self.eos_id = getattr(self._tokenizer, "eos_id", None)

    def encode(self, *args, **kwargs) -> list[int]:
        if isinstance(self._tokenizer, SPTokenizer):
            if "add_bos" in kwargs:
                kwargs["bos"] = kwargs.pop("add_bos")
            if "add_eos" in kwargs:
                kwargs["eos"] = kwargs.pop("add_eos")
        return self._tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        return self._tokenizer.decode(*args, **kwargs)

    def get_vocab_size(self) -> int:
        if hasattr(self._tokenizer, "get_vocab_size"):
            return self._tokenizer.get_vocab_size()
        return int(getattr(self._tokenizer, "vocab_size"))
