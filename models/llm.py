from dataclasses import dataclass
from typing import Optional, Literal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CONFIG

ModelName = Literal["llama", "qwen", "wiroai"]


@dataclass
class LLMConfig:
    model_name: ModelName = "llama"
    max_new_tokens: int = 200
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.2


class LLM:
    def __init__(self, config:Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.device = CONFIG.device
    
        if self.config.model_name == "llama":
            model_path = CONFIG.model_paths.llama_path
        elif self.config.model_name == "qwen":
            model_path = CONFIG.model_paths.qwen_path
        elif self.config.model_name == "wiroai":
            model_path = CONFIG.model_paths.wiroai_path
        else:
            raise ValueError(f"Unsopported model_name: {self.config.model_name}")
        
        print(f"[LLM] Loading model: {self.config.model_name} from {model_path}")

        if CONFIG.dtype == "float16":
            torch_dtype = torch.float16
        elif CONFIG.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code = True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch_dtype,
            device_map = "cuda:0",
            trust_remote_code = True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    @torch.inference_mode()
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        "Single function will work for both Llama and Qwen"
        
        if system_prompt:
            full_prompt = f"<system>{system_prompt}</system\n\n<user>{prompt}</user>"
        else:
            full_prompt = prompt
        
        inputs = self.tokenizer(
            full_prompt,
            return_tensors = "pt",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens = self.config.max_new_tokens,
            do_sample = self.config.temperature > 0,
            temperature = self.config.temperature,
            top_p = self.config.top_p,
            repetition_penalty = self.config.repetition_penalty,
            pad_token_id = self.tokenizer.eos_token_id,
        )

        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens = True,
        )


        return generated.strip()
