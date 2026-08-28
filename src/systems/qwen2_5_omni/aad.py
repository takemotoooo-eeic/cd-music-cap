"""
Modified from official implementation of https://arxiv.org/pdf/2506.07233
"""
import numpy as np
import torch
import torch.nn.functional as F
from src.systems.generation.logits_process import AADLogitsProcessor
from copy import deepcopy
from .qwen2_5_omni import Qwen2_5OmniSystem


class AADBeam(object):
    def __init__(self, data: dict, data_neg: dict, system_name: str):
        self.system_name = system_name
        self.generated = []
        self.data = data
        self.data_neg = data_neg

    def add_token(self, new_token):
        if isinstance(new_token, int):
            new_token = torch.LongTensor([new_token]).to(self.data["input_ids"].device)
        self.generated.append(new_token)
        self.data["input_ids"] = torch.cat([self.data["input_ids"], new_token.unsqueeze(-1)], dim=-1)
        self.data_neg["input_ids"] = torch.cat([self.data_neg["input_ids"], new_token.unsqueeze(-1)], dim=-1)

        if "attention_mask" in self.data:
            new_mask = torch.ones((self.data["attention_mask"].shape[0], 1), device=new_token.device, dtype=self.data["attention_mask"].dtype)
            self.data["attention_mask"] = torch.cat([self.data["attention_mask"], new_mask], dim=-1)
            new_mask = torch.ones((self.data_neg["attention_mask"].shape[0], 1), device=new_token.device, dtype=self.data_neg["attention_mask"].dtype)
            self.data_neg["attention_mask"] = torch.cat([self.data_neg["attention_mask"], new_mask], dim=-1)

    def copy(self) -> "AADBeam":
        beam = AADBeam(deepcopy(self.data), deepcopy(self.data_neg), self.system_name)
        beam.generated = deepcopy(self.generated)
        return beam

class AADSystem(Qwen2_5OmniSystem):
    def __init__(self, config):
        super().__init__(config)
        aad_config = self.model_config["aad"]
        self.alpha = aad_config.get("alpha", 0.5)
        self.threshold = aad_config.get("threshold", -1)
        self.negative = aad_config.get("negative", "none")
        if self.negative not in ("none", "silence"):
            raise ValueError(f"aad.negative must be 'none' or 'silence', got {self.negative!r}")
    
    def prepare_logits_processor(self, texts, audios) -> AADLogitsProcessor:
        if self.negative == "none":
            prompts = [self.format_prompt(text, audio_exist=False) for text in texts]
            audio_neg = None
        elif self.negative == "silence":
            prompts = [self.format_prompt(text, audio_exist=True) for text in texts]
            audio_neg = [np.zeros_like(x) for x in audios]
        else:
            raise NotImplementedError

        inputs_without_audio = self.processor(
            text=prompts,
            audio=audio_neg,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        ).to(self.device)

        logits_processor = AADLogitsProcessor(
            self.model, 
            inputs_without_audio,
            alpha=self.alpha, 
            threshold=self.threshold
        )
        
        return logits_processor

    @torch.inference_mode()
    def inference(self, audios: list[np.ndarray], texts: list[str], ids: list[str], max_new_tokens: int = 512) -> str:
        assert len(texts) == 1, "Currently no batch inference"
        
        prompts = [self.format_prompt(text) for text in texts]
        # prepare logits processor
        aad_logits_processor = self.prepare_logits_processor(texts, audios)
        
        # Prepare inputs
        inputs = self.processor(audio=audios, text=prompts, return_tensors="pt", padding=True, sampling_rate=16000).to(self.device)
        # text-only generation
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            logits_processor=[aad_logits_processor]
        )
        output_ids = output_ids[:, inputs["input_ids"].size(1):]

        prediction = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # clean up
        step_all, step_applied = aad_logits_processor.call_cnt, aad_logits_processor.apply_cnt

        del aad_logits_processor
        torch.cuda.empty_cache()

        return {
            "prediction": prediction,
            "steps": (step_all, step_applied)
        }

    def get_beam(self, audio_input, text_input) -> AADBeam:
        audios = [audio_input]
        prompts = [self.format_prompt(text_input)]
        inputs = self.processor(
            audio=audios,
            text=prompts,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        ).to(self.device)

        if self.negative == "none":
            prompts_neg = [self.format_prompt(text_input, audio_exist=False)]
            audio_neg = None
        elif self.negative == "silence":
            prompts_neg = [self.format_prompt(text_input, audio_exist=True)]
            audio_neg = [np.zeros_like(audio_input)]
        else:
            raise NotImplementedError

        inputs_neg = self.processor(
            audio=audio_neg,
            text=prompts_neg,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000
        ).to(self.device)

        return AADBeam(inputs, inputs_neg, system_name=self.config["system_name"])
