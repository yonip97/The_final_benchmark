"""
``mistralai/Ministral-3-14B-Instruct-2512`` — Transformers per model card (Mistral3 + MistralCommonBackend).
https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

**Device:** constructor ``local_device`` (``"cpu"`` | ``"cuda"``). CUDA uses ``device_map="auto"`` on visible GPUs; set ``CUDA_VISIBLE_DEVICES`` to pin. Parallel workers set that per process when ``local_device=="cuda"``.
"""

from __future__ import annotations

import os

import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend,FineGrainedFP8Config


class Ministral3LocalModel:
    MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512"

    def __init__(self, token: str | None = None, local_device: str = "cuda") -> None:
        if local_device not in ("cpu", "cuda"):
            raise ValueError("local_device must be 'cpu' or 'cuda'")
        self.model_id = self.MODEL_ID
        self._local_device = local_device
        self._token = token or os.environ.get("HUGGINGFACE_TOKEN")
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
       # tok_kw = {"token": self._token}
        use_cuda = self._local_device == "cuda" and torch.cuda.is_available()
        self._tokenizer = MistralCommonBackend.from_pretrained(self.MODEL_ID,token = self._token)
        if use_cuda:
            self._model = Mistral3ForConditionalGeneration.from_pretrained(
                self.MODEL_ID, device_map="auto",quantization_config=FineGrainedFP8Config(dequantize=True),token=self._token
            ).eval()
        else:
            self._model = Mistral3ForConditionalGeneration.from_pretrained(
                self.MODEL_ID,quantization_config=FineGrainedFP8Config(dequantize=True),token=self._token
            ).eval()

    def infer_with_usage(self, prompt: str, max_new_tokens:int,temperature:float, **kwargs) -> tuple[str, int, int]:
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
            "temperature":temperature,
        }
        with torch.inference_mode():
            out = self._model.generate(**tokenized, **gen_kw)[0]
        text = self._tokenizer.decode(out[input_len:], skip_special_tokens=True)
        return (text.strip(), 0, 0)
