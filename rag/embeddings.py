from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from config import CONFIG


@dataclass
class EmbeddingConfig:
    model_path: str = CONFIG.model_paths.embedding_model_path
    device: str = CONFIG.device


class EmbeddingModel:
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.device = self.config.device

        print(f"[Embedding] Loading embedding model from: {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code = True,
            local_files_only = True,
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code = True,
            local_files_only = True,
        ).to(self.device)

        print(f"[Embedding] Model device: {self.model.device}")

        self.model.eval()
    
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding = True,
                truncation = True,
                return_tensors = "pt",
                max_length = 512,
            ).to(self.device)

            model_output = self.model(**encoded)

            # Mean Pooling
            # last_hidden_state shape: (batch, seq_len, hidden_size)
            last_hidden_state = model_output.last_hidden_state
            attention_mask = encoded["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts

            # L2 normalize
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
