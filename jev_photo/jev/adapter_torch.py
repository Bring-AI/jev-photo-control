"""Qwen3.5 (transformers / CUDA) cache, position and LM-head details.

PyTorch port of jev-visual's MLX Qwen35Adapter. Qwen3.5 interleaves Gated
DeltaNet (conv + recurrent state) with full attention (KV). Branching the shared
prefix therefore copies BOTH kinds of state; forking only KV would be wrong.
"""
import copy
import time
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache

MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass
class Timings:
    preprocessing_ms: float = 0
    vision_ms: float = 0
    prefill_ms: float = 0
    scoring_ms: float = 0
    cache_fork_ms: float = 0
    language_forward_calls: int = 0
    vision_forward_calls: int = 0

    def dict(self):
        return asdict(self)


class Qwen35TorchAdapter:
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, torch.device(device)
        self.tokenizer = processor.tokenizer
        self.core = model.model  # Qwen3_5Model: .visual, .language_model
        self.timings = Timings()

    def sync(self):
        torch.cuda.synchronize(self.device)

    def reset(self):
        self.sync()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.timings = Timings()

    def peak_memory_gb(self):
        return torch.cuda.max_memory_allocated(self.device) / 1e9

    def prepare(self, prompt, image, extra_ids=None):
        t = time.perf_counter()
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        if extra_ids:
            extra = torch.tensor([extra_ids], dtype=inputs["input_ids"].dtype)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(extra, dtype=inputs["mm_token_type_ids"].dtype)], dim=1)
        inputs = inputs.to(self.device)
        self.timings.preprocessing_ms += (time.perf_counter() - t) * 1000
        return inputs

    def project(self, hidden, picks):
        """Project ONLY requested positions through the unmodified LM head (float32 out)."""
        selected = torch.cat([hidden[i, torch.tensor(row, device=hidden.device)] for i, row in enumerate(picks)])
        logits = self.model.lm_head(selected).float()
        return list(torch.split(logits, [len(row) for row in picks]))

    @torch.inference_mode()
    def prefill(self, inputs, picks=None):
        ids = inputs["input_ids"]
        t = time.perf_counter()
        positions, _ = self.core.get_rope_index(ids, inputs["mm_token_type_ids"],
                                                image_grid_thw=inputs.get("image_grid_thw"))
        embeds = self.core.get_input_embeddings()(ids)
        if inputs.get("pixel_values") is not None:
            image_embeds = self.core.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                        return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds).to(embeds.device, embeds.dtype)
            mask, _ = self.core.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=image_embeds)
            embeds = embeds.masked_scatter(mask, image_embeds)
            self.timings.vision_forward_calls += 1
        self.sync()
        self.timings.vision_ms += (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        cache = DynamicCache(config=self.model.config)
        out = self.core.language_model(inputs_embeds=embeds, position_ids=positions,
                                       past_key_values=cache, use_cache=True)
        logits = self.project(out.last_hidden_state, picks) if picks else None
        self.sync()
        self.timings.prefill_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1
        return cache, int(positions.max().item()) + 1, logits

    def fork(self, cache, count):
        t = time.perf_counter()
        branches = copy.deepcopy(cache)
        # reorder_cache index-selects every KV tensor AND every conv/recurrent state.
        branches.reorder_cache(torch.zeros(count, dtype=torch.long, device=self.device))
        self.sync()
        self.timings.cache_fork_ms += (time.perf_counter() - t) * 1000
        return branches

    @torch.inference_mode()
    def suffix(self, cache, next_position, rows, picks):
        branches = self.fork(cache, len(rows))
        t = time.perf_counter()
        length = max(map(len, rows))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = torch.tensor([row + [pad] * (length - len(row)) for row in rows], device=self.device)
        positions = (torch.arange(length, device=self.device) + next_position).view(1, 1, -1).expand(3, len(rows), -1)
        out = self.core.language_model(inputs_embeds=self.core.get_input_embeddings()(ids), position_ids=positions,
                                       past_key_values=branches, use_cache=True)
        logits = self.project(out.last_hidden_state, picks)
        self.sync()
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1
        # Right padding cannot affect earlier causal outputs; padded branch
        # states are discarded here and never continued.
        return logits


def load_adapter(model_path=None, device="cuda:0", dtype="bfloat16"):
    source = model_path or MODEL_ID
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    if model.config.model_type != "qwen3_5":
        raise ValueError(f"No verified adapter for {model.config.model_type}; only qwen3_5")
    return Qwen35TorchAdapter(model, processor, device), source
