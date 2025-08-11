from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from transformers import DebertaModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedLinear


@dataclass
class DebertaHRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


class DebertaHRMConfig(BaseModel):
    model_name: str = "microsoft/deberta-v3-base"
    hidden_size: int = 768
    num_labels: int = 4

    H_cycles: int = 1
    L_cycles: int = 2

    H_layers: int = 1
    L_layers: int = 1

    # Transformer config (matching HRM)
    expansion: float = 4.0
    num_heads: int = 12

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    forward_dtype: str = "bfloat16"


class DebertaHRMBlock(nn.Module):
    """HRM reasoning block - copied from original HRM implementation"""
    def __init__(self, config: DebertaHRMConfig) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class DebertaHRMReasoningModule(nn.Module):
    """HRM reasoning module - copied from original HRM implementation"""
    def __init__(self, layers: List[DebertaHRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class DebertaHRM(nn.Module):
    """DeBERTa + HRM hybrid model"""
    def __init__(self, config: DebertaHRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Full DeBERTa model as embed
        self.deberta = DebertaModel.from_pretrained(config.model_name)

        # Get sequence length from DeBERTa's position embeddings
        self.seq_len = self.deberta.config.max_position_embeddings

        # Rotary embeddings for HRM layers
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_heads,
            max_position_embeddings=self.seq_len,
            base=self.config.rope_theta
        )

        # HRM reasoning modules (trainable)
        self.H_level = DebertaHRMReasoningModule(
            layers=[DebertaHRMBlock(self.config) for _i in range(self.config.H_layers)]
        )
        self.L_level = DebertaHRMReasoningModule(
            layers=[DebertaHRMBlock(self.config) for _i in range(self.config.L_layers)]
        )

        # Classification head
        self.q_head = CastedLinear(self.config.hidden_size, self.config.num_labels, bias=True)

        # Initial states (trainable)
        self.H_init = nn.Parameter(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        )
        self.L_init = nn.Parameter(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        )

        # Initialize Q head for faster learning
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5) # pyright: ignore

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:


        deberta_outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        input_embeddings = deberta_outputs.last_hidden_state.to(self.forward_dtype)

        batch_size, seq_len = input_embeddings.shape[:2]

        # Initialize hidden states
        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Sequence info for positional encoding
        seq_info = dict(cos_sin=self.rotary_emb())

        # HRM reasoning cycles (following original implementation)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        # Final step with gradients
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Classification using [CLS] token (first token)
        logits = self.q_head(z_H[:, 0]).to(torch.float32)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)
