import torch
from torch import nn
import os
from pathlib import Path
import transformers


_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_CLIP_CACHE_DIR = "models--openai--clip-vit-base-patch32"


def _load_pretrained_quietly(load_fn, *args, **kwargs):
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        return load_fn(*args, **kwargs)
    finally:
        transformers.logging.set_verbosity(verbosity)


def _local_clip_safetensors_dir():
    cache_roots = []
    for env_name in (
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "CLIP_CACHE_DIR",
    ):
        value = os.environ.get(env_name)
        if value:
            cache_roots.append(Path(value))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")

    seen = set()
    candidates = []
    for root in cache_roots:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)
        snapshots = root / _CLIP_CACHE_DIR / "snapshots"
        candidates.extend(snapshots.glob("*/model.safetensors"))

    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].parent


def _load_clip_text_model():
    safetensors_dir = _local_clip_safetensors_dir()
    if safetensors_dir is not None:
        full_config = _load_pretrained_quietly(
            transformers.CLIPConfig.from_pretrained,
            _CLIP_MODEL_NAME,
            local_files_only=True
        )
        return _load_pretrained_quietly(
            transformers.CLIPTextModel.from_pretrained,
            str(safetensors_dir),
            config=full_config.text_config,
            use_safetensors=True,
            local_files_only=True
        )

    try:
        return _load_pretrained_quietly(
            transformers.CLIPTextModel.from_pretrained,
            _CLIP_MODEL_NAME,
            use_safetensors=True
        )
    except (OSError, ValueError):
        return _load_pretrained_quietly(
            transformers.CLIPTextModel.from_pretrained,
            _CLIP_MODEL_NAME
        )


class ClipTokenizer:

    def __init__(self):
        super().__init__()
        self.tokenizer = _load_pretrained_quietly(
            transformers.CLIPTokenizer.from_pretrained,
            _CLIP_MODEL_NAME
        )

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            padding="longest",
            return_tensors="pt"
        )["input_ids"]


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = _load_clip_text_model()

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
