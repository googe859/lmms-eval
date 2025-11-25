#!/usr/bin/env python
"""
Minimal VQA inference entry point for Qwen3-VL.

Usage example:
    python scripts/run_infer.py \
        --checkpoint ./checkpoints/Qwen3-VL-4B-Instruct \
        --image ./cookbooks/assets/demo.jpeg \
        --question "Describe the image."
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


merged_feats: list[tuple[str, torch.Tensor]] = []
DEFAULT_KEEP_RATIO = 0.10
KEEP_RATIO = DEFAULT_KEEP_RATIO


def _normalize_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 2:  # (seq, hidden)
        return hidden.unsqueeze(0)
    if hidden.ndim == 3:  # (batch, seq, hidden)
        return hidden
    if hidden.ndim == 4:  # (batch, channels, h, w)
        b, c, h, w = hidden.shape
        return hidden.view(b, c, -1).transpose(1, 2)
    raise ValueError(f"Unsupported hidden shape: {hidden.shape}")



def register_merger_hooks(model, max_blocks=None):
    handles = []

    def capture(module_name):
        def hook(_, __, output):
            merged_feats.append((module_name, output.detach().cpu()))

        return hook

    handles.append(model.model.visual.merger.register_forward_hook(capture("merger_final")))

    deep_list = model.model.visual.deepstack_merger_list
    indexes = model.model.visual.deepstack_visual_indexes
    limit = len(indexes) if max_blocks is None else min(len(indexes), max_blocks)
    for idx in range(limit):
        block_idx = indexes[idx]
        handles.append(
            deep_list[idx].register_forward_hook(capture(f"deepstack_block_{block_idx:02d}"))
        )

    return handles


def run_representation_pass(model, inputs, max_blocks, pass_name: str) -> list[tuple[str, torch.Tensor]]:
    print(f"Running representation pass ({pass_name}) ...")
    merged_feats.clear()
    handles = register_merger_hooks(model, max_blocks)
    with torch.inference_mode():
        model(**inputs, output_hidden_states=True, return_dict=True)
    for handle in handles:
        handle.remove()
    captured = [(label, feat.clone()) for label, feat in merged_feats]
    merged_feats.clear()
    return captured


def run_generation(model, processor, inputs, max_new_tokens, tag: str) -> str:
    print(f"Running generation ({tag}) ...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return answer


def install_language_pruning_hook(
    model,
    keep_indices: torch.Tensor,
    vision_len: int,
) -> tuple[torch.utils.hooks.RemovableHandle, dict[str, int]]:
    """Register a hook that trims visual tokens right before entering the language model."""
    keep_indices = keep_indices.clone()
    state: dict[str, int] = {"applied": 0, "kept": int(keep_indices.numel()), "total": int(vision_len)}

    def lm_prune_hook(_, args, kwargs):
        inputs_embeds: torch.Tensor | None = kwargs.get("inputs_embeds")
        visual_pos_masks: torch.Tensor | None = kwargs.get("visual_pos_masks")
        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
        if (
            inputs_embeds is None
            or visual_pos_masks is None
            or deepstack_visual_embeds is None
            or state["applied"] > 0
        ):
            return args, kwargs

        device = inputs_embeds.device
        keep_idx = keep_indices.to(device)
        mask = visual_pos_masks.to(device)
        if mask.shape[0] != 1:
            raise NotImplementedError("Token pruning currently supports batch size 1 only.")
        vision_positions = torch.nonzero(mask[0], as_tuple=False).squeeze(-1)
        if vision_positions.numel() != vision_len:
            raise RuntimeError(
                f"Expected {vision_len} visual tokens, but found {vision_positions.numel()} in language inputs."
            )

        seq_mask = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=device)
        keep_flags = torch.zeros(vision_len, dtype=torch.bool, device=device)
        keep_flags[keep_idx] = True
        drop_positions = vision_positions[~keep_flags]
        seq_mask[drop_positions] = False

        kwargs["inputs_embeds"] = inputs_embeds[:, seq_mask, :].contiguous()

        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            kwargs["attention_mask"] = attention_mask[:, seq_mask]
        elif isinstance(attention_mask, dict):
            new_mask = {}
            for key, value in attention_mask.items():
                if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == seq_mask.shape[0]:
                    new_mask[key] = value[..., seq_mask]
                else:
                    new_mask[key] = value
            kwargs["attention_mask"] = new_mask

        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            kwargs["position_ids"] = position_ids[..., seq_mask]

        cache_position = kwargs.get("cache_position")
        if cache_position is not None and cache_position.shape[-1] == seq_mask.shape[0]:
            kwargs["cache_position"] = cache_position[seq_mask]

        kwargs["visual_pos_masks"] = mask[:, seq_mask]

        pruned_deepstack: list[torch.Tensor] = []
        for embed in deepstack_visual_embeds:
            pruned_deepstack.append(embed.to(device).index_select(0, keep_idx))
        kwargs["deepstack_visual_embeds"] = pruned_deepstack

        state["applied"] += 1
        return args, kwargs

    handle = model.model.language_model.register_forward_pre_hook(
        lm_prune_hook,
        with_kwargs=True,
    )
    return handle, state


def visualize_features(
    feats: list[tuple[str, torch.Tensor]],
    out_dir: Path,
    orig_image: Image.Image,
    model_image: Image.Image,
    t: int,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> None:
    merged_h = max(1, grid_h // merge_size)
    merged_w = max(1, grid_w // merge_size)
    vision_len = t * merged_h * merged_w
    orig_arr = np.array(orig_image, dtype=np.float32)
    model_arr = np.array(model_image, dtype=np.float32)

    final_feat = None
    for label, feat in feats:
        if label == "merger_final":
            final_feat = feat.to(torch.float32)
            break
    if final_feat is None:
        raise RuntimeError("Failed to capture final merged features.")

    final_grid = final_feat[:vision_len].view(t, merged_h, merged_w, -1)
    final_norms = final_grid.norm(dim=-1)
    flat_norms = final_norms.reshape(-1)
    keep_k = max(1, int(flat_norms.numel() * KEEP_RATIO))
    topk = torch.topk(flat_norms, keep_k, largest=True)
    keep_indices = topk.indices
    keep_mask = torch.zeros_like(flat_norms, dtype=torch.bool)
    keep_mask.scatter_(0, topk.indices, True)
    keep_mask_grid = keep_mask.view(t, merged_h, merged_w)
    mask_vis = keep_mask_grid.any(dim=0).float()

    keep_indices = keep_indices.to(torch.long)
    kept = int(keep_indices.numel())
    total = int(flat_norms.numel())
    print(f"Keeping top {kept}/{total} tokens ({kept / total:.1%}) from final merger.")

    for idx, (label, feat) in enumerate(feats):
        feat = feat.to(torch.float32)
        token_norm = feat[:vision_len].norm(dim=-1).view(t, merged_h, merged_w)
        grid = token_norm.mean(0) if t > 1 else token_norm.squeeze(0)
        grid_np = grid.cpu().numpy()
        min_val, max_val = np.percentile(grid_np, [1, 99])
        grid_clipped = np.clip(grid_np, min_val, max_val)
        grid_norm = (grid_clipped - min_val) / (max_val - min_val + 1e-6)
        grid_display = np.power(grid_norm, 0.8)
        selected_display = grid_display * mask_vis.cpu().numpy()

        cmap = plt.get_cmap("inferno")
        heat_color = (cmap(selected_display)[..., :3] * 255).astype("uint8")
        heat = Image.fromarray(heat_color).resize(model_image.size, Image.BILINEAR)
        heat_arr = np.array(heat, dtype=np.float32)
        overlay_arr = np.clip(0.6 * model_arr + 0.4 * heat_arr, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(grid_display, cmap="inferno")
        axes[0].set_title(f"{label}")
        axes[0].axis("off")

        axes[1].imshow(overlay_arr.astype(np.uint8))
        axes[1].set_title("Top 10% Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{idx:02d}_{label}_merger.png", bbox_inches="tight")
        plt.close(fig)
    return keep_indices, keep_mask_grid.clone()
    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VQA inference with Qwen3-VL.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/Qwen3-VL-4B-Instruct",
        help="Model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image used for VQA.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation length for the answer.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force running inference on CPU.",
    )
    parser.add_argument(
        "--max-early-blocks",
        type=int,
        default=4,
        help="Number of early vision blocks to capture (-1 for all).",
    )
    parser.add_argument(
        "--fixed-image-size",
        type=int,
        default=1024,
        help="Resize input image to (size, size) before feeding the model (-1 to disable).",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=DEFAULT_KEEP_RATIO,
        help="Fraction of visual tokens to retain when running the sparse pass (0-1].",
    )

    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def main() -> None:
    out_dir = Path("visualizations_merger")
    out_dir.mkdir(exist_ok=True)
    args = parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep-ratio must be within (0, 1].")
    global KEEP_RATIO
    KEEP_RATIO = args.keep_ratio

    device_map = "auto"
    torch_dtype = "auto"
    if args.cpu_only:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print(f"Loading model from {args.checkpoint} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    max_blocks = None if args.max_early_blocks < 0 else args.max_early_blocks

    orig_image = load_image(args.image)
    model_image = orig_image
  
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": model_image},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    baseline_feats = run_representation_pass(model, inputs, max_blocks, "baseline")
    baseline_answer = run_generation(model, processor, inputs, args.max_new_tokens, "Baseline")
    print("\nQuestion:", args.question)
    print("Baseline Answer:", baseline_answer)

    t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
    merge_size = model.model.visual.spatial_merge_size
    keep_indices, keep_mask_grid = visualize_features(
        baseline_feats,
        out_dir,
        orig_image,
        model_image,
        t,
        grid_h,
        grid_w,
        merge_size,
    )

    merged_h = max(1, grid_h // merge_size)
    merged_w = max(1, grid_w // merge_size)
    vision_len = t * merged_h * merged_w


















    lm_hook, prune_state = install_language_pruning_hook(
        model=model,
        keep_indices=keep_indices,
        vision_len=vision_len,
    )
    try:
        sparse_answer = run_generation(model, processor, inputs, args.max_new_tokens, "Sparse")
    finally:
        lm_hook.remove()

    print("Sparse Answer:", sparse_answer)
    print(
        f"Sparse pass kept {prune_state['kept']}/{prune_state['total']} visual tokens "
        f"({prune_state['kept'] / prune_state['total']:.1%})."
    )
    print("Sparse visualization skipped (reserved for token modification experiments).")

    






if __name__ == "__main__":
    main()
