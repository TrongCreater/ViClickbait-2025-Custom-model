import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    # Thay bằng model repository thật trên Hugging Face.
    hf_repo_id: str = os.getenv(
        "HF_REPO_ID",
        "TrongCreater05/PhoBertCustom_ClickBait2025",
    )

    checkpoint_filename: str = "clickbait_detector_deploy.pt"
    source_vocab_filename: str = "source_vocab.json"
    category_vocab_filename: str = "category_vocab.json"
    model_config_filename: str = "model_config.json"

    threshold: float = 0.5


APP_CONFIG = AppConfig()
