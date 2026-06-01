"""Compatibility helpers for Nemotron-H generation cache handling."""

from __future__ import annotations

import functools
import sys

import torch


def patch_nemotron_h_cache_compat(model) -> bool:
    """Patch the loaded Nemotron-H module so generate() can use cache correctly.

    This bridges the cache name mismatch introduced by newer transformers
    (cache_params vs past_key_values) and fixes the Mamba convolution cache
    shape for the fast path.
    """

    target_model, module = _resolve_nemotron_h_model(model)
    if target_model is None or module is None:
        return False

    cache_cls = getattr(module, "HybridMambaAttentionDynamicCache", None)
    if cache_cls is None:
        return False

    if not getattr(cache_cls, "_llm_infer_cache_patch", False):
        _patch_cache_class(cache_cls)

    _patch_model_generation(target_model)
    return True


def _resolve_nemotron_h_model(model):
    candidates = [model]
    seen = set()

    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        module = sys.modules.get(candidate.__class__.__module__)
        if module is not None and hasattr(module, "HybridMambaAttentionDynamicCache"):
            if hasattr(candidate, "prepare_inputs_for_generation") and hasattr(candidate, "forward"):
                return candidate, module

        for attr_name in ("base_model", "model"):
            child = getattr(candidate, attr_name, None)
            if child is not None:
                candidates.append(child)

    return None, None


def _patch_cache_class(cache_cls):
    orig_init = cache_cls.__init__
    orig_update_conv_state = cache_cls.update_conv_state
    orig_get_seq_length = cache_cls.get_seq_length

    @functools.wraps(orig_init)
    def patched_init(self, config, batch_size, dtype=torch.float16, device=None):
        orig_init(self, config, batch_size, dtype=dtype, device=device)

        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        conv_dim = intermediate_size + 2 * config.n_groups * config.ssm_state_size
        conv_kernel_size = config.conv_kernel
        pattern = config.hybrid_override_pattern

        for layer_idx in range(config.num_hidden_layers):
            if pattern[layer_idx] == "M":
                if (
                    self.conv_states[layer_idx].ndim != 3
                    or self.conv_states[layer_idx].shape[1] != conv_dim
                    or self.conv_states[layer_idx].shape[2] != conv_kernel_size
                ):
                    self.conv_states[layer_idx] = torch.zeros(
                        batch_size, conv_dim, conv_kernel_size, device=device, dtype=dtype
                    )

    def patched_update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        _device = self.conv_states[layer_idx].device
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(_device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            if new_conv_state.dim() == 2:
                self.conv_states[layer_idx][:, :, -1] = new_conv_state.to(_device)
            else:
                self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(_device)
        return self.conv_states[layer_idx]

    def patched_get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) == 0:
            return 0
        if layer_idx not in self.transformer_layers and self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        if layer_idx >= len(self.key_cache):
            return 0
        cache = self.key_cache[layer_idx]
        if cache.numel() == 0 or cache.shape[-1] == 0:
            return 0
        return cache.shape[-2]

    cache_cls.__init__ = patched_init
    cache_cls.update_conv_state = patched_update_conv_state
    cache_cls.get_seq_length = patched_get_seq_length
    cache_cls._llm_infer_cache_patch = True
    cache_cls._llm_infer_orig_init = orig_init
    cache_cls._llm_infer_orig_update_conv_state = orig_update_conv_state
    cache_cls._llm_infer_orig_get_seq_length = orig_get_seq_length


def _patch_model_generation(model):
    if getattr(model, "_llm_infer_generation_patch", False):
        return

    orig_prepare = model.prepare_inputs_for_generation
    orig_forward = model.forward

    @functools.wraps(orig_prepare)
    def patched_prepare(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, position_ids=None, use_cache=True, **kwargs):
        cache_params = kwargs.pop("cache_params", None)
        if past_key_values is None and cache_params is not None:
            past_key_values = cache_params

        if past_key_values is not None and cache_position is None:
            try:
                past_length = int(past_key_values.get_seq_length())
            except Exception:
                past_length = 0
            if past_length >= input_ids.shape[1]:
                cache_position = torch.tensor([past_length], device=input_ids.device)
            else:
                cache_position = torch.arange(past_length, input_ids.shape[1], device=input_ids.device)

        return orig_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

    @functools.wraps(orig_forward)
    def patched_forward(*args, **kwargs):
        if "past_key_values" in kwargs and "cache_params" not in kwargs:
            kwargs["cache_params"] = kwargs.pop("past_key_values")
        return orig_forward(*args, **kwargs)

    model.prepare_inputs_for_generation = patched_prepare
    model.forward = patched_forward
    model._llm_infer_generation_patch = True
