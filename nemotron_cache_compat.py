"""Compatibility helpers for Nemotron-H generation cache handling."""

from __future__ import annotations

import functools
import sys

import torch


class _StateList(list):
    @property
    def device(self):
        for tensor in self:
            if getattr(tensor, "numel", lambda: 0)() > 0:
                return tensor.device
        if self:
            return self[0].device
        return torch.device("cpu")


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

    _patch_model_module(module)
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
        self.conv_states = _StateList(self.conv_states)
        self.ssm_states = _StateList(self.ssm_states)

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

        candidate_layers = []
        if layer_idx is not None and 0 <= layer_idx < len(self.key_cache):
            candidate_layers.append(layer_idx)
        candidate_layers.extend(i for i in range(len(self.key_cache)) if i not in candidate_layers)

        for candidate_layer in candidate_layers:
            cache = self.key_cache[candidate_layer]
            if cache.numel() > 0 and cache.shape[-1] != 0:
                return cache.shape[-2]
        return 0

    cache_cls.__init__ = patched_init
    cache_cls.update_conv_state = patched_update_conv_state
    cache_cls.get_seq_length = patched_get_seq_length
    cache_cls._llm_infer_cache_patch = True
    cache_cls._llm_infer_orig_init = orig_init
    cache_cls._llm_infer_orig_update_conv_state = orig_update_conv_state
    cache_cls._llm_infer_orig_get_seq_length = orig_get_seq_length


def _patch_model_module(module):
    block_cls = getattr(module, "NemotronHBlock", None)
    if block_cls is not None and not getattr(block_cls, "_llm_infer_block_patch", False):
        _patch_block_class(block_cls)

    moe_cls = getattr(module, "NemotronHMOE", None)
    if moe_cls is not None and not getattr(moe_cls, "_llm_infer_moe_patch", False):
        _patch_moe_class(moe_cls)

    causal_lm_cls = getattr(module, "NemotronHForCausalLM", None)
    if causal_lm_cls is not None and not getattr(causal_lm_cls, "_llm_infer_logits_patch", False):
        _patch_causal_lm_class(causal_lm_cls, module)


def _patch_block_class(block_cls):
    orig_forward = block_cls.forward

    @functools.wraps(orig_forward)
    def patched_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            if self.block_type == "mamba":
                hidden_states = self.mixer(
                    hidden_states, cache_params=cache_params, cache_position=cache_position
                )
            elif self.block_type == "attention":
                hidden_states = self.mixer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=cache_params,
                    cache_position=cache_position,
                    use_cache=cache_params is not None,
                )[0]
            elif self.block_type in ["mlp", "moe"]:
                hidden_states = self.mixer(hidden_states)
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")

            return residual + hidden_states

    block_cls.forward = patched_forward
    block_cls._llm_infer_block_patch = True
    block_cls._llm_infer_orig_forward = orig_forward


def _patch_moe_class(moe_cls):
    orig_moe = moe_cls.moe

    def patched_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        active_experts = torch.unique(topk_indices).tolist()

        for expert_idx in active_experts:
            token_indices, weight_indices = torch.where(topk_indices == expert_idx)
            if token_indices.numel() == 0:
                continue

            expert_output = self.experts[expert_idx](hidden_states[token_indices])
            expert_weights = topk_weights[token_indices, weight_indices].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, expert_output * expert_weights)

        return final_hidden_states.to(hidden_states.dtype)

    moe_cls.moe = patched_moe
    moe_cls._llm_infer_moe_patch = True
    moe_cls._llm_infer_orig_moe = orig_moe


def _patch_causal_lm_class(causal_lm_cls, module):
    output_cls = getattr(module, "NemotronHCausalLMOutput")
    orig_forward = causal_lm_cls.forward

    @functools.wraps(orig_forward)
    def patched_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        cache_params=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_cache=None,
        cache_position=None,
        attention_mask=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]

        logits_hidden_states = hidden_states
        logits_to_keep = kwargs.get("logits_to_keep", None)
        if labels is None and logits_to_keep is not None and logits_to_keep != 0:
            logits_hidden_states = hidden_states[:, -int(logits_to_keep):, :]

        logits = self.lm_head(logits_hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = module.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return output_cls(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    causal_lm_cls.forward = patched_forward
    causal_lm_cls._llm_infer_logits_patch = True
    causal_lm_cls._llm_infer_orig_forward = orig_forward


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
