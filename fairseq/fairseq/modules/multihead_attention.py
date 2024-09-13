# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
import numpy as np
import random


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.head_mask = None
        self.head_mask_empty = None
        self.head_mask_skip = None
        self.head_mask_select = None
        self.is_generative = None
        
        self._apply_gates = False
        self.gate = ConcreteGate([1,self.num_heads,1,1]) 
        self.reg_coeff = None
        self.confidence = torch.zeros(self.num_heads)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        global_start = time.time()
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # skip entire attention layer
        if self.head_mask_skip and self.head_mask_empty:
            bmm_start_1 = time.time()
            attn = torch.zeros_like(query, memory_format=torch.contiguous_format)
            bmm_end_1 = time.time()
            attn = self.out_proj(attn)
            attn_weights: Optional[Tensor] = None

            global_end = time.time()
            overall_time = global_end - global_start
            bmm_time = (bmm_end_1 - bmm_start_1)
            if self.is_generative:
                print('overall time is %.9f s, bmm time is %.9f s, rate is %.3f%%' %
                      (overall_time, bmm_time, bmm_time / overall_time * 100))

            return attn, attn_weights

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            and self.head_mask is None
            and not self._apply_gates
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        # select attention head and compute
        expand_q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        expand_k = k.transpose(1, 2).view(bsz, self.num_heads, self.head_dim, src_len)
        weight_list = []
        if self.head_mask is not None:
            flat_head_mask = self.head_mask.view(-1)
        bmm_start_1 = time.time()
        if self.head_mask_select and self.head_mask is not None:
            attn_weights = torch.zeros(bsz, self.num_heads, tgt_len, src_len, device=query.device)
            unpruned_index = (flat_head_mask > 0.5).nonzero(as_tuple=True)[0]
            unpruned_q = expand_q[:, unpruned_index, :, :].reshape(bsz * unpruned_index.size()[0], tgt_len, self.head_dim)
            unpruned_k = expand_k[:, unpruned_index, :, :].reshape(bsz * unpruned_index.size()[0], self.head_dim, src_len)
            unpruned_attn = torch.bmm(unpruned_q, unpruned_k).reshape(bsz, unpruned_index.size()[0], tgt_len, src_len)
            attn_weights[:, unpruned_index, :, :] = unpruned_attn
            attn_weights = attn_weights.reshape(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        bmm_end_1 = time.time()

        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        # for each head, add max weights for each token and record
        attn_probs = self.dropout_module(attn_weights)

        if self.head_mask is not None and not self.head_mask_select:
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len) * self.head_mask
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        if self.head_mask is None:
            attn_confidence = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_confidence, _ = torch.max(attn_confidence, dim=3)
            attn_confidence = torch.mean(attn_confidence, dim=[0, 2])
            attn_confidence = attn_confidence.view(self.num_heads)
            self.confidence = torch.add(self.confidence, attn_confidence.cpu())
            if self._apply_gates:
                attn_probs = self.gate(attn_probs.view(bsz, self.num_heads, tgt_len, src_len))
                attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        assert v is not None

        # select attention head and compute attn * v
        expand_attn = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
        expand_v = v.view(bsz, self.num_heads, src_len, self.head_dim)
        weight_list = []
        bmm_start_2 = time.time()
        if self.head_mask_select and self.head_mask is not None:
            attn = torch.zeros(bsz, self.num_heads, tgt_len, self.head_dim, device=query.device)
            unpruned_index = (flat_head_mask > 0.5).nonzero(as_tuple=True)[0]
            unpruned_attn = expand_attn[:, unpruned_index, :, :].reshape(bsz * unpruned_index.size()[0], tgt_len, src_len)
            unpruned_v = expand_v[:, unpruned_index, :, :].reshape(bsz * unpruned_index.size()[0], src_len, self.head_dim)
            unpruned_attn = torch.bmm(unpruned_attn, unpruned_v).reshape(bsz, unpruned_index.size()[0], tgt_len, self.head_dim)
            attn[:, unpruned_index, :, :] = unpruned_attn
            attn = attn.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        else:
            attn = torch.bmm(attn_probs, v)
        bmm_end_2 = time.time()

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        global_end = time.time()
        overall_time = global_end - global_start
        bmm_time = (bmm_end_1 - bmm_start_1) + (bmm_end_2 - bmm_start_2)
        if self.is_generative:
            print('overall time is %.9f s, bmm time is %.9f s, rate is %.3f%%' %
                  (overall_time, bmm_time, bmm_time / overall_time * 100))

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def apply_masks(self, head_mask, head_skip=False, head_select=False, is_generative=False):
        self.head_mask = head_mask
        self.is_generative = is_generative
        self.head_mask_empty = bool(torch.sum(self.head_mask) == 0)
        self.head_mask_skip = head_skip
        self.head_mask_select = head_select

    def apply_gates(self, reg_coeff):
        if not self._apply_gates:
            self._apply_gates = True
            self.reg_coeff = reg_coeff

    def get_penalty(self):
        reg = 0.0
        if self._apply_gates:
            reg = self.gate.get_penalty(self.reg_coeff)
        return reg

    def get_penalty_p1(self):
        reg = 0.0
        if self._apply_gates:
            reg = self.gate.get_penalty_p1()
        return reg

    def get_penalty_p0(self):
        reg = 0.0
        if self._apply_gates:
            reg = self.gate.get_penalty_p0()
        return reg

    def get_penalty_concentrator(self):
        reg = 0.0
        if self._apply_gates:
            reg = self.gate.get_penalty_concentrator()
        return reg

    def get_penalty_pnb(self):
        reg = 0.0
        if self._apply_gates:
            reg = self.gate.get_penalty_pnb()
        return reg

    def get_gate_values(self, is_concrete):
        gate_values = None
        if self.gate is not None:
            if is_concrete:
                gate_values = self.gate.get_gates(False).flatten()
            elif self.gate.gate_values is not None:
                gate_values = self.gate.gate_values.flatten()
        return gate_values

    def get_loga(self):
        loga = None
        if self.gate is not None:
            loga = self.gate.get_loga().flatten()
        return loga

    def get_confidence(self):
        conf = None
        if self.confidence is not None:
            conf = self.confidence
        return conf

    def set_loga(self, loga):
        self.gate.set_loga(loga)
    
    def remove_gates(self):
        self._apply_gates = False

    def reset_confidence(self):
        self.confidence = torch.zeros(self.num_heads)

    def remove_masks(self):
        self.head_mask = None


class ConcreteGate(nn.Module):
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    """

    def __init__(self, shape, temperature=0.33, stretch_limits=(-0.1, 1.1), eps=1e-6):
        super(ConcreteGate, self).__init__()
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.log_a = nn.Parameter(torch.empty(shape))
        self.shape = shape
        self.gate_values = None
        nn.init.xavier_uniform_(self.log_a)

    def forward(self, values, is_train=None):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        is_train = self.training if is_train is None else is_train
        gates = self.get_gates(is_train)
        if is_train:
            self.gate_values = gates
        return values * gates

    def get_gates(self, is_train):
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits
        if is_train:
            shape = self.log_a.size()
            noise = (1 - 2*self.eps) * torch.rand(shape).to(self.log_a.device) + self.eps
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = torch.sigmoid(self.log_a)
        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)
        return clipped_concrete

    def get_loga(self):
        """ return the log_a variables"""
        return self.log_a

    def set_loga(self, loga):
        # with torch.no_grad():
        self.log_a = nn.Parameter(loga.view(self.shape))

    def get_penalty(self, reg_coeff):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
        p_open = torch.sigmoid(self.log_a - self.temperature * np.log(-low / high))
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        total_reg = reg_coeff * torch.sum(p_open)

        return total_reg

    def get_penalty_p1(self):
        """
        Computes penalty p1
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_1 = torch.sigmoid(self.log_a - self.temperature * np.log((1 - low) / (high - 1)))
        p_1 = torch.clamp(p_1, self.eps, 1.0 - self.eps)

        total_reg = torch.sum(p_1)

        return total_reg

    def get_penalty_p0(self):
        """
        Computes penalty p0
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_0 = torch.sigmoid(self.temperature * np.log(-low / high) - self.log_a)
        p_0 = torch.clamp(p_0, self.eps, 1.0 - self.eps)

        total_reg = torch.sum(p_0)

        return total_reg

    def get_penalty_concentrator(self):
        """
        Computes penalty concentrator
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_0 = torch.sigmoid(self.temperature * np.log(-low / high) - self.log_a)
        p_0 = torch.clamp(p_0, self.eps, 1.0 - self.eps)

        total_reg = 1 - torch.prod(p_0)

        return total_reg

    def get_penalty_pnb(self):
        """
        Computes penalty pnb
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_0 = torch.sigmoid(self.temperature * np.log(-low / high) - self.log_a)
        p_1 = torch.sigmoid(self.log_a - self.temperature * np.log((1 - low) / (high - 1)))
        p_not_binary = 1 - (p_1 + p_0)
        p_not_binary = torch.clamp(p_not_binary, self.eps, 1.0 - self.eps)

        total_reg = torch.sum(p_not_binary)

        return total_reg

    def get_sparsity_rate(self):
        """ Computes the fraction of gates which are now active (non-zero) """
        is_nonzero = self.get_gates(False) == 0.0
        return torch.mean(is_nonzero.float())