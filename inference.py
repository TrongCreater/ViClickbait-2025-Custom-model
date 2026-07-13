from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from config import APP_CONFIG
from model import ClickbaitDetector


class ClickbaitPredictor:
    def __init__(self, hf_token: str | None = None) -> None:
        self.device = torch.device("cpu")
        self.hf_token = hf_token or None

        self.model_config = self._download_json(
            APP_CONFIG.model_config_filename
        )
        self.source_vocab = self._download_json(
            APP_CONFIG.source_vocab_filename
        )
        self.category_vocab = self._download_json(
            APP_CONFIG.category_vocab_filename
        )

        self.source_vocab = {
            str(key): int(value)
            for key, value in self.source_vocab.items()
        }
        self.category_vocab = {
            str(key): int(value)
            for key, value in self.category_vocab.items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["phobert_name"],
            token=self.hf_token,
            use_fast=False,
        )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.model = self._load_model()

    def _download_file(self, filename: str) -> str:
        return hf_hub_download(
            repo_id=APP_CONFIG.hf_repo_id,
            filename=filename,
            repo_type="model",
            token=self.hf_token,
        )

    def _download_json(self, filename: str) -> dict[str, Any]:
        path = self._download_file(filename)
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Checkpoint phải là dict, nhận được {type(checkpoint)}"
            )

        for key in ("model_state", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate

        # Checkpoint deploy có thể chỉ chứa state_dict.
        if checkpoint and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in checkpoint.items()
        ):
            return checkpoint

        raise KeyError(
            "Không tìm thấy model_state/model_state_dict/state_dict."
        )

    def _load_checkpoint(self, path: str) -> dict[str, Any]:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except RuntimeError as error:
            # Hỗ trợ checkpoint cũ từng bị gắn location tag không hợp lệ.
            if "tagged with device" not in str(error):
                raise
            return torch.load(
                path,
                map_location=lambda storage, location: storage,
                weights_only=True,
            )

    def _load_model(self) -> ClickbaitDetector:
        cfg = self.model_config

        num_sources = int(
            cfg.get("num_sources", len(self.source_vocab))
        )
        
        num_categories = int(
            cfg.get("num_categories", len(self.category_vocab))
        )
        
        if num_sources != len(self.source_vocab):
            raise ValueError(
                "Số source không khớp: "
                f"config={num_sources}, "
                f"source_vocab={len(self.source_vocab)}"
            )
        
        if num_categories != len(self.category_vocab):
            raise ValueError(
                "Số category không khớp: "
                f"config={num_categories}, "
                f"category_vocab={len(self.category_vocab)}"
            )
        
        model = ClickbaitDetector(
            num_sources=num_sources,
            num_categories=num_categories,
            phobert_name=cfg["phobert_name"],
            cnn_channels=int(cfg["cnn_channels"]),
            hidden_dim=int(cfg["hidden_dim"]),
            meta_embed_dim=int(cfg["meta_embed_dim"]),
            dropout=float(cfg["dropout"]),
            freeze_phobert_layers=int(
                cfg.get("freeze_phobert_layers", 6)
            ),
        )

        checkpoint_path = self._download_file(
            APP_CONFIG.checkpoint_filename
        )
        checkpoint = self._load_checkpoint(checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)

        # Hỗ trợ checkpoint được lưu từ DataParallel.
        cleaned_state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(cleaned_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def get_sources(self) -> list[str]:
        return [
            name
            for name, _ in sorted(
                self.source_vocab.items(),
                key=lambda item: item[1],
            )
        ]

    def get_categories(self) -> list[str]:
        return [
            name
            for name, _ in sorted(
                self.category_vocab.items(),
                key=lambda item: item[1],
            )
        ]

    def _prepare_image(
        self,
        image: Image.Image | None,
    ) -> torch.Tensor:
        if image is None:
            # Giữ đúng fallback của Dataset lúc train.
            return torch.zeros((3, 224, 224), dtype=torch.float32)

        try:
            image = image.convert("RGB")
            return self.image_transform(image)
        except Exception:
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    @torch.inference_mode()
    def predict(
        self,
        title: str,
        lead: str = "",
        image: Image.Image | None = None,
        source: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        lead = lead.strip()

        if not title:
            raise ValueError("Tiêu đề không được để trống.")

        # Giữ đúng preprocessing lúc train.
        text_input = (
            f"{title} {self.tokenizer.sep_token} {lead}"
        )

        encoded = self.tokenizer(
            text_input,
            add_special_tokens=True,
            max_length=int(self.model_config["max_len"]),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        image_tensor = self._prepare_image(image).unsqueeze(0)

        source_id = self.source_vocab.get(str(source), 0)
        category_id = self.category_vocab.get(str(category), 0)

        logits, attention = self.model(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            images=image_tensor.to(self.device),
            source_ids=torch.tensor(
                [source_id],
                dtype=torch.long,
                device=self.device,
            ),
            category_ids=torch.tensor(
                [category_id],
                dtype=torch.long,
                device=self.device,
            ),
        )

        probability = torch.sigmoid(logits)[0].item()
        predicted_id = int(probability >= APP_CONFIG.threshold)

        return {
            "label_id": predicted_id,
            "label": (
                "clickbait"
                if predicted_id == 1
                else "non-clickbait"
            ),
            "clickbait_probability": float(probability),
            "non_clickbait_probability": float(1.0 - probability),
            "source_id": int(source_id),
            "category_id": int(category_id),
            "attention_shape": list(attention.shape),
        }
