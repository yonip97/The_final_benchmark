"""
``mistralai/Ministral-3-14B-Instruct-2512`` — Transformers per model card (Mistral3 + MistralCommonBackend).
https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

**Device:** optional ``cuda_device_ids`` (physical indices) with ``device_map="auto"`` + ``max_memory``; else full-machine auto.
"""

from __future__ import annotations

import os

import torch
from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration, MistralCommonBackend

from .cuda_placement import max_memory_for_device_ids


class Ministral3LocalModel:
    MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512"

    def __init__(
        self,
        token: str | None = None,
        local_device: str = "cuda",
        cuda_device_ids: tuple[int, ...] | None = None,
    ) -> None:
        if local_device not in ("cpu", "cuda"):
            raise ValueError("local_device must be 'cpu' or 'cuda'")
        self.model_id = self.MODEL_ID
        self._local_device = local_device
        self._cuda_device_ids = cuda_device_ids
        self._token = token or os.environ.get("HUGGINGFACE_TOKEN")
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        use_cuda = self._local_device == "cuda" and torch.cuda.is_available()
        self._tokenizer = MistralCommonBackend.from_pretrained(self.MODEL_ID, token=self._token)
        if use_cuda:
            kw: dict = {
                "device_map": "auto",
                "quantization_config": FineGrainedFP8Config(dequantize=True),
                "token": self._token,
            }
            if self._cuda_device_ids:
                kw["max_memory"] = max_memory_for_device_ids(self._cuda_device_ids)
            self._model = Mistral3ForConditionalGeneration.from_pretrained(self.MODEL_ID, **kw).eval()
        else:
            self._model = Mistral3ForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                token=self._token,
            ).eval()

    def infer_with_usage(self, prompt: str, max_new_tokens: int, temperature: float, **kwargs) -> tuple[str, int, int]:
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        device = next(self._model.parameters()).device

        tokenized = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(device)
        input_len = tokenized["input_ids"].shape[-1]
        gen_kw: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        with torch.inference_mode():
            out = self._model.generate(**tokenized, **gen_kw)[0]
        text = self._tokenizer.decode(out[input_len:], skip_special_tokens=True)
        return (text.strip(), 0, 0)
