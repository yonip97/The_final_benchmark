import os

from transformers import pipeline


class HuggingFaceModel:
    """
    Hugging Face model inference (Qwen, Mistral, Llama, etc.) via transformers.
    Uses the model's chat template when available (instruct/chat models); otherwise
    plain text completion. Runs on GPU if available.
    """

    def __init__(
        self,
        model_id: str,
        device: int | str | None = None,
        token: str | None = None,
        use_chat_template: bool = True,
        **pipeline_kwargs,
    ):
        self.model_id = model_id
        self.device = device
        self._token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.use_chat_template = use_chat_template
        self._pipeline_kwargs = pipeline_kwargs
        self._pipe = None

    def _get_pipeline(self):
        if self._pipe is None:
            self._pipe = pipeline(
                "text-generation",
                model=self.model_id,
                device=self.device,
                token=self._token,
                **self._pipeline_kwargs,
            )
        return self._pipe

    def create_config(self, prompt: str, **kwargs) -> dict:
        _ = prompt
        pipe = self._get_pipeline()
        return {
            "max_new_tokens": kwargs.get("max_new_tokens", 2000),
            "do_sample": kwargs.get("do_sample", False),
            "pad_token_id": pipe.tokenizer.eos_token_id,
            "return_full_text": False,
        }

    def _format_prompt(self, prompt: str) -> tuple[str, str]:
        """
        Format prompt with chat template if the tokenizer has one; return (input_string, prefix_to_strip).
        prefix_to_strip is the exact string that will appear before the model's reply so we can extract it.
        """
        pipe = self._get_pipeline()
        tokenizer = pipe.tokenizer
        if not self.use_chat_template or getattr(tokenizer, "chat_template", None) is None:
            return prompt, prompt
        try:
            # One user turn; add_generation_prompt adds the assistant turn start
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not formatted or not isinstance(formatted, str):
                return prompt, prompt
            # We'll need to strip this prefix from the model output to get only the reply
            return formatted, formatted
        except Exception:
            return prompt, prompt

    def infer_with_usage(self, prompt: str, max_new_tokens: int = 2000, **kwargs) -> tuple[str, int, int]:
        pipe = self._get_pipeline()
        formatted, _ = self._format_prompt(prompt)
        if "max_new_tokens" not in kwargs:
            kwargs = {**kwargs, "max_new_tokens": max_new_tokens}
        out = pipe(formatted, **self.create_config(prompt, **kwargs))
        if not out or not isinstance(out, list):
            return ("", 0, 0)
        text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
        return (text.strip(), 0, 0)
