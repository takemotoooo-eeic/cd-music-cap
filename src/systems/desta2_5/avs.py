"""Adaptive Vector Steering for DeSTA2.5."""
import numpy as np
import torch

from src.systems.generation.avs import DenseHiddenStateController
from .desta2_5 import Desta2_5System


class AVSSystem(Desta2_5System):
    def __init__(self, config):
        super().__init__(config)
        self.avs = DenseHiddenStateController(self.model, self.model_config)

    def _negative_inputs(
        self,
        audios: list[np.ndarray],
        texts: list[str],
    ) -> dict:
        if self.avs.negative == "none":
            audio_neg, transcription_neg = None, None
            prompts_neg = [self.format_prompt(None, None, text) for text in texts]
        elif self.avs.negative == "silence":
            audio_neg = [np.zeros_like(x) for x in audios]
            transcription_neg = [" "] * len(audio_neg)
            prompts_neg = [
                self.format_prompt(audio, transcription, text)
                for audio, transcription, text in zip(audio_neg, transcription_neg, texts)
            ]
        else:
            raise NotImplementedError
        return self.processor(
            audio=audio_neg,
            transcription=transcription_neg,
            text=prompts_neg,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> dict:
        assert len(texts) == 1, "Currently no batch inference"

        transcriptions = self.model.prepare_transcriptions(audios, [None])
        prompts = [
            self.format_prompt(audio, transcription, text)
            for audio, transcription, text in zip(audios, transcriptions, texts)
        ]
        inputs = self.processor(
            audio=audios,
            transcription=transcriptions,
            text=prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        inputs_neg = self._negative_inputs(audios, texts)

        steering_vectors = self.avs.compute_steering_vectors(inputs, inputs_neg)
        self.avs.register(steering_vectors)
        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            self.avs.remove()

        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        return {"prediction": prediction}
