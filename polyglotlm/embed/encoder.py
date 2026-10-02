"""Sentence encoders: mean-pooled LLM hidden states, an optional projection, L2 normalisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import torch

from polyglotlm.modeling import load_model
from polyglotlm.text import nfc
from polyglotlm.tokenizer import load_tokenizer

ENCODER_CONFIG = "encoder.json"
PROJECTION_FILE = "projection.pt"


class TextEncoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> torch.Tensor: ...


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class LMEncoder:
    """Mean of the last hidden layer over real tokens, projected and L2-normalised."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        projection: torch.nn.Linear | None = None,
        max_length: int = 128,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_length = max_length
        self.batch_size = batch_size
        hidden = model.config.hidden_size
        self.dim = projection.out_features if projection is not None else hidden

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        pad = self.tokenizer.pad_token_id
        pad = self.tokenizer.eos_token_id if pad is None else pad
        rows = [
            self.tokenizer.encode(nfc(t), add_special_tokens=False)[: self.max_length] or [pad]
            for t in texts
        ]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        device = self.model.device
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Differentiable embeddings (used while distilling)."""
        inputs = self._tokenize(texts)
        hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1].float()
        pooled = mean_pool(hidden, inputs["attention_mask"])
        if self.projection is not None:
            pooled = self.projection(pooled)
        return torch.nn.functional.normalize(pooled, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            chunks = [
                self.forward(texts[i : i + self.batch_size])
                for i in range(0, len(texts), self.batch_size)
            ]
        finally:
            self.model.train(was_training)
        return torch.cat(chunks).cpu() if chunks else torch.empty(0, self.dim)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out_dir))
        self.tokenizer.save_pretrained(str(out_dir))
        if self.projection is not None:
            torch.save(self.projection.state_dict(), out_dir / PROJECTION_FILE)
        config = {
            "pooling": "mean",
            "normalize": True,
            "max_length": self.max_length,
            "dim": self.dim,
            "projection": self.projection is not None,
        }
        (out_dir / ENCODER_CONFIG).write_text(json.dumps(config, indent=2) + "\n", "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LMEncoder:
        path = Path(path)
        model = load_model(path).eval()
        config_path = path / ENCODER_CONFIG
        config = json.loads(config_path.read_text("utf-8")) if config_path.exists() else {}
        projection = None
        if config.get("projection"):
            state = torch.load(path / PROJECTION_FILE, map_location="cpu", weights_only=True)
            projection = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
            projection.load_state_dict(state)
            projection.to(model.device)
        return cls(model, load_tokenizer(path), projection, config.get("max_length", 128))


class SentenceTransformerEncoder:
    """An off-the-shelf English teacher (`pip install sentence-transformers`)."""

    def __init__(self, name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"teacher {name!r} needs the `sentence-transformers` package"
            ) from exc
        self._model = SentenceTransformer(name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        vectors = self._model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return vectors.float().cpu()
