"""Adaptive Vector Steering for Qwen2.5-Omni."""
import numpy as np
import torch

from src.systems.generation.avs import DenseHiddenStateController
from src.systems.generation.logits_process import _nested_omni_forward
from .qwen2_5_omni import Qwen2_5OmniSystem


def _qwen_forward(model, inputs):
    outputs, _ = _nested_omni_forward(
        model,
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs


class AVSSystem(Qwen2_5OmniSystem):
    def __init__(self, config):
        super().__init__(config)
        self.avs = DenseHiddenStateController(self.model, self.model_config)

    def _negative_inputs(self, texts: list[str], audios: list[np.ndarray]) -> dict:
        if self.avs.negative == "none":
            prompts = [self.format_prompt(text, audio_exist=False) for text in texts]
            audio_neg = None
        elif self.avs.negative == "silence":
            prompts = [self.format_prompt(text, audio_exist=True) for text in texts]
            audio_neg = [np.zeros_like(x) for x in audios]
        else:
            raise NotImplementedError
        return self.processor(
            text=prompts,
            audio=audio_neg,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        ).to(self.device)

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Currently no batch inference"

        prompts = [self.format_prompt(text) for text in texts]
        inputs = self.processor(
            audio=audios,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        ).to(self.device)
        inputs_neg = self._negative_inputs(texts, audios)

        steering_vectors = self.avs.compute_steering_vectors(inputs, inputs_neg, forward_fn=_qwen_forward)
        self.avs.register(steering_vectors)
        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        finally:
            self.avs.remove()

        output_ids = output_ids[:, inputs["input_ids"].size(1):]
        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        return {"prediction": prediction}
