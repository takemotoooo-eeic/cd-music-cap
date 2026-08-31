"""Adaptive Vector Steering for Audio Flamingo 3."""
import numpy as np
import torch

from src.systems.generation.avs import DenseHiddenStateController
from .audio_flamingo_3 import AudioFlamingo3System


class AVSSystem(AudioFlamingo3System):
    def __init__(self, config):
        super().__init__(config)
        self.avs = DenseHiddenStateController(self.model, self.model_config)

    def _negative_inputs(self, texts: list[str], audios: list) -> dict:
        if self.avs.negative == "none":
            audio_neg = None
        elif self.avs.negative == "silence":
            audio_neg = np.zeros_like(audios[0])
        else:
            raise NotImplementedError
        conversation = self.format_conversation(texts[0], audio_path_or_array=audio_neg)
        return self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)

    @torch.inference_mode()
    def inference(self, audios: list, texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Batch size 1 recommended for AVS"

        conversation = self.format_conversation(texts[0], ids[0])
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.device)
        inputs_neg = self._negative_inputs(texts, audios)

        steering_vectors = self.avs.compute_steering_vectors(inputs, inputs_neg)
        self.avs.register(steering_vectors)
        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        finally:
            self.avs.remove()

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        return {"prediction": prediction}
