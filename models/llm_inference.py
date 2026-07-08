"""
KubeSage LLM Generator
======================
Manages local text generation via HuggingFace transformers.
Tailored for low-resource CPU execution with models in the 1B-4B parameter range 
(specifically HuggingFaceTB/SmolLM2-1.7B-Instruct).
"""

import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.config import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """
    LLM generator using HuggingFace transformers pipeline.

    Loads a small instruction-tuned model for CPU inference.
    Generates structured JSON incident reports from RAG prompts.

    Attributes:
        model: Loaded HuggingFace causal LM.
        tokenizer: Associated tokenizer.
        model_name: Name of the loaded model.
        device: Device string (cpu/cuda).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the LLM generator.

        Args:
            model_name: HuggingFace model ID. Defaults to config.
            device: Device for inference. Defaults to config.
        """
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.device = device or settings.LLM_DEVICE

        logger.info(f"Loading LLM: {self.model_name} (device={self.device})")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load model with CPU optimizations
        # Use float16 as primary dtype for CPU (halves memory, minimal quality loss)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        # Try float16 first (CPU-friendly), fall back to float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs,
            )
        except Exception as e:
            logger.warning(f"float16 load failed: {e}. Trying float32...")
            load_kwargs["torch_dtype"] = torch.float32
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kwargs,
                )
            except Exception as e2:
                logger.error(f"Both float16 and float32 loads failed: {e2}")
                raise RuntimeError(
                    f"Failed to load model {self.model_name}: {e2}"
                ) from e2

        # Move to device
        if self.device == "cuda":
            self.model = self.model.cuda()
            logger.info(f"Model loaded on GPU ({torch.cuda.get_device_name(0)})")
        else:
            self.model = self.model.cpu()
            # Apply dynamic quantization only if configured
            if settings.LLM_LOAD_IN_8BIT:
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                    logger.info("Applied dynamic quantization (LLM_LOAD_IN_8BIT=true)")
                except Exception as e:
                    logger.warning(f"Dynamic quantization failed (model may not support it): {e}")

        self.model.eval()

        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"LLM loaded: {params_m:.0f}M parameters | Device: {self.device}")

    # -------------------------------------------------------------------
    # Chat Prompt Building
    # -------------------------------------------------------------------

    def _build_chat_prompt(self, system: str, user: str) -> str:
        """
        Build a chat-format prompt using the model's native template.

        Uses tokenizer.apply_chat_template() for correct BOS tokens,
        special token placement, and model-specific formatting.
        Falls back to ChatML if the tokenizer lacks a chat_template.

        Args:
            system: System prompt string.
            user: User prompt string.

        Returns:
            Formatted prompt string (with add_generation_prompt=True).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError, KeyError):
            # Fallback to ChatML if tokenizer has no chat_template
            logger.debug(
                "apply_chat_template unavailable; falling back to ChatML"
            )
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    # -------------------------------------------------------------------
    # Core generation (private, shared by generate and generate_json)
    # -------------------------------------------------------------------

    def _generate_impl(
        self,
        prompt: str,
        prefix: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Core generation logic shared by generate() and generate_json().

        Args:
            prompt: Full tokenized prompt string (including chat template).
            prefix: String to prepend to the decoded output.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text (prefix + decoded new tokens, stripped).
        """
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.time() - start_time
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = prefix + self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        num_tokens = len(generated_ids)
        tps = num_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"LLM: {num_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

        return response.strip()

    # -------------------------------------------------------------------
    # Public generation methods
    # -------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate free-form text using the loaded LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User-level query with context.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        prompt = self._build_chat_prompt(system_prompt, user_prompt)
        return self._generate_impl(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_start: str = "{",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a JSON response by pre-filling the assistant response.

        Forces structured JSON output by starting the assistant's
        response with json_start (default: '{'). The returned string
        has the prefix prepended for complete JSON parsing.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User-level query with context.
            json_start: String to pre-fill the assistant response.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text including the pre-filled prefix.
        """
        prompt = self._build_chat_prompt(system_prompt, user_prompt) + json_start
        return self._generate_impl(prompt, prefix=json_start, max_tokens=max_tokens, temperature=temperature)

    def generate_batch(
        self,
        prompts: list[dict[str, str]],
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> list[str]:
        """
        Generate text for multiple prompts sequentially.

        Args:
            prompts: List of dicts with 'system' and 'user' keys.
            max_tokens: Max tokens per generation.
            json_mode: If True, use generate_json (pre-fills '{').

        Returns:
            List of generated text responses.
        """
        method = self.generate_json if json_mode else self.generate
        responses: list[str] = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating {i+1}/{len(prompts)}...")
            responses.append(method(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"],
                max_tokens=max_tokens,
            ))
        return responses

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return hasattr(self, "model") and self.model is not None
