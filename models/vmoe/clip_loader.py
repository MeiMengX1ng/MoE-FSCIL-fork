import os
import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from transformers import CLIPVisionModel
from transformers.utils import logging as transformers_logging

DEFAULT_CLIP_REPO = "openai/clip-vit-base-patch16"
CLIP_REQUIRED_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def _load_clip_from_pretrained(path_or_repo):
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        return CLIPVisionModel.from_pretrained(path_or_repo)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)


def get_clip_encoder_layers(model):
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None and hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
        return vision_model.encoder.layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return model.encoder.layers
    available = ", ".join(sorted(name for name in dir(model) if not name.startswith("_"))[:50])
    raise AttributeError(
        f"Unable to locate CLIP encoder layers on {type(model).__name__}. Available attributes: {available}"
    )


def _download_file(url, destination, max_redirects=5):
    current_url = url
    for _ in range(max_redirects + 1):
        request = Request(current_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request) as response, open(destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            current_url = urljoin(current_url, location)
    raise RuntimeError(f"Too many redirects while downloading {url}")


def _prepare_clip_local_snapshot(repo_id):
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    cache_root = Path(os.getenv("HF_FALLBACK_CACHE", Path.home() / ".cache" / "moe-fscil" / "hf"))
    repo_dir = cache_root / repo_id.replace("/", "--")
    repo_dir.mkdir(parents=True, exist_ok=True)

    for filename in CLIP_REQUIRED_FILES:
        destination = repo_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        _download_file(f"{endpoint}/{repo_id}/resolve/main/{filename}", destination)

    return str(repo_dir)


def load_clip_vision_backbone(backbone_name_or_path=None):
    backbone_name_or_path = backbone_name_or_path or os.getenv("CLIP_VIT_B16_PATH", DEFAULT_CLIP_REPO)
    try:
        return _load_clip_from_pretrained(backbone_name_or_path)
    except OSError as exc:
        if os.path.isdir(backbone_name_or_path):
            raise
        fallback_path = _prepare_clip_local_snapshot(backbone_name_or_path)
        try:
            return _load_clip_from_pretrained(fallback_path)
        except OSError:
            raise exc
