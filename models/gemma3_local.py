"""
``google/gemma-3-12b-it`` — Gemma3ForConditionalGeneration + AutoProcessor per model card.
https://huggingface.co/google/gemma-3-12b-it

**Device:** optional ``cuda_device_ids`` (physical indices) with ``device_map="auto"`` + ``max_memory``; else full-machine auto.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .cuda_placement import max_memory_for_device_ids


class Gemma3LocalModel:
    MODEL_ID = "google/gemma-3-12b-it"

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
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        use_cuda = self._local_device == "cuda" and torch.cuda.is_available()
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=self._token)
        if use_cuda:
            kw: dict = {
                "dtype": torch.bfloat16,
                "device_map": "auto",
                "token": self._token,
            }
            if self._cuda_device_ids:
                kw["max_memory"] = max_memory_for_device_ids(self._cuda_device_ids)
            self._model = Gemma3ForConditionalGeneration.from_pretrained(self.MODEL_ID, **kw).eval()
        else:
            self._model = Gemma3ForConditionalGeneration.from_pretrained(
                self.MODEL_ID, dtype=torch.bfloat16, token=self._token
            ).eval()

    def infer_with_usage(self, prompt: str, max_new_tokens: int, temperature: float, **kwargs) -> tuple[str, int, int]:
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        device = next(self._model.parameters()).device
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]
        gen_kw: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        with torch.inference_mode():
            gen = self._model.generate(**inputs, **gen_kw)[0]
        text = self._processor.decode(gen[input_len:], skip_special_tokens=True)
        return (text.strip(), 0, 0)
