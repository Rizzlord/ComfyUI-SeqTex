# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from functools import cache
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import WanTransformer3DModel
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import \
    WanTimeTextImageEmbedding
from diffusers.utils import (USE_PEFT_BACKEND, logging, scale_lora_layers,
                             unscale_lora_layers)
from einops import rearrange, repeat
from peft import LoraConfig


class WanT2TexAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        geometry_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if geometry_embedding is not None:
            # add-type geometry embedding
            if True:
                if isinstance(geometry_embedding, Tuple):
                    query = query + geometry_embedding[0]
                    key = key + geometry_embedding[1]
                else:
                    query = query + geometry_embedding
                    key = key + geometry_embedding
            else:
                # mul-type geometry embedding
                if isinstance(geometry_embedding, Tuple):
                    query = query * (1 + geometry_embedding[0])
                    key = key * (1 + geometry_embedding[1])
                else:
                    query = query * (1 + geometry_embedding)
                    key = key * (1 + geometry_embedding)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2) # [B, F*H*W, 2C] -> [B, H, F*H*W, 2C//H]
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            
            if isinstance(rotary_emb, Tuple):
                query = apply_rotary_emb(query, rotary_emb[0])
                key = apply_rotary_emb(key, rotary_emb[1])
            else:
                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanTimeTaskTextImageEmbedding(WanTimeTextImageEmbedding):
    def __init__(
        self,
        original_model,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        randomly_init: bool = False,
    ):
        super(WanTimeTaskTextImageEmbedding, self).__init__(dim, time_freq_dim, time_proj_dim, text_embed_dim, image_embed_dim)
        if not randomly_init:
            self.load_state_dict(original_model.state_dict(), strict=True)
        # cond_proj = nn.Linear(512, original_model.timesteps_proj.num_channels, bias=False)
        # setattr(self.time_embedder, "cond_proj", cond_proj)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        # time_cond: Optional[torch.Tensor] = None,
    ):
        B = timestep.shape[0]
        timestep = rearrange(timestep, "B F -> (B F)")
        timestep = self.timesteps_proj(timestep)
        timestep = rearrange(timestep, "(B F) D -> B F D", B=B)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0, addtional_qk_geo: bool = False
    ):
        super().__init__()

        if addtional_qk_geo: # to add PE to geometry embedding
            attention_head_dim = attention_head_dim * 2
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor, uv_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        _, _, uv_num_frames, uv_height, uv_width = uv_hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        uppf, upph, uppw = uv_num_frames // p_t, uv_height // p_h, uv_width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        uv_freqs_f = freqs[0][ppf:ppf+uppf].view(uppf, 1, 1, -1).expand(uppf, upph, uppw, -1)
        uv_freqs_h = freqs[1][:upph].view(1, upph, 1, -1).expand(uppf, upph, uppw, -1)
        uv_freqs_w = freqs[2][:uppw].view(1, 1, uppw, -1).expand(uppf, upph, uppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        uv_freqs = torch.cat([uv_freqs_f, uv_freqs_h, uv_freqs_w], dim=-1).reshape(1, 1, uppf * upph * uppw, -1)
        return torch.cat([freqs, uv_freqs], dim=-2)

class WanT2TexTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        addtional_qk_geo: bool = False,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanT2TexAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanT2TexAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.geometry_caster = nn.Linear(dim, dim)
        nn.init.zeros_(self.geometry_caster.weight.data)
        nn.init.zeros_(self.geometry_caster.bias.data)

        self.attnuv = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanT2TexAttnProcessor2_0(),
        )
        self.normuv2 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.scale_shift_table_uv = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.ffnuv = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        geometry_embedding: Optional[torch.Tensor] = None,
        token_shape: Optional[Tuple[int, int, int, int, int, int]] = None,
    ) -> torch.Tensor:
        post_patch_num_frames, post_patch_height, post_patch_width, post_uv_num_frames, post_uv_height, post_uv_width = token_shape
        mv_temb, uv_temb = temb[:, :post_patch_num_frames], temb[:, post_patch_num_frames:]
        mv_temb = repeat(mv_temb, "B F N D -> B N (F H W) D", H=post_patch_height, W=post_patch_width)
        uv_temb = repeat(uv_temb, "B F N D -> B N (F H W) D", H=post_uv_height, W=post_uv_width)
        dit_ssg = rearrange(self.scale_shift_table, "1 N D -> 1 N 1 D") + mv_temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = torch.unbind(dit_ssg, dim=1)
        dit_ssg_uv = rearrange(self.scale_shift_table_uv, "1 N D -> 1 N 1 D") + uv_temb.float()
        shift_msa_uv, scale_msa_uv, gate_msa_uv, c_shift_msa_uv, c_scale_msa_uv, c_gate_msa_uv = torch.unbind(dit_ssg_uv, dim=1)

        geometry_embedding = self.geometry_caster(geometry_embedding)

        n_mv, n_uv = post_patch_num_frames * post_patch_height * post_patch_width, post_uv_num_frames * post_uv_height * post_uv_width
        assert hidden_states.shape[1] == n_mv + n_uv, f"hidden_states shape {hidden_states.shape} is not equal to {n_mv + n_uv}"
        mv_hidden_states, uv_hidden_states = hidden_states[:, :n_mv], hidden_states[:, n_mv:]



        # 1. Self-attention
        mv_norm_hidden_states = (self.norm1(mv_hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(mv_hidden_states)
        uv_norm_hidden_states = (self.norm1(uv_hidden_states.float()) * (1 + scale_msa_uv) + shift_msa_uv).type_as(uv_hidden_states)


        mv_attn_output = self.attn1(hidden_states=mv_norm_hidden_states, rotary_emb=rotary_emb[:, :, :n_mv], attention_mask=attn_bias, geometry_embedding=geometry_embedding[:, :n_mv])
        mv_hidden_states = (mv_hidden_states.float() + mv_attn_output * gate_msa).type_as(mv_hidden_states)
        uv_attn_output = self.attnuv(hidden_states=uv_norm_hidden_states, encoder_hidden_states=torch.cat([mv_hidden_states, uv_norm_hidden_states], dim=1), 
                                      rotary_emb=(rotary_emb[:, :, n_mv:], rotary_emb), geometry_embedding=(geometry_embedding[:, n_mv:], geometry_embedding))
        

        uv_hidden_states = (uv_hidden_states.float() + uv_attn_output * gate_msa_uv).type_as(uv_hidden_states)


      



        # 2. Cross-attention
        mv_norm_hidden_states = self.norm2(mv_hidden_states.float()).type_as(mv_hidden_states)
        uv_norm_hidden_states = self.normuv2(uv_hidden_states.float()).type_as(uv_hidden_states)
        attn_output = self.attn2(hidden_states=torch.cat([mv_norm_hidden_states, uv_norm_hidden_states], dim=1), encoder_hidden_states=encoder_hidden_states)
        mv_attn_output, uv_attn_output = attn_output[:, :n_mv], attn_output[:, n_mv:]
        mv_hidden_states.add_(mv_attn_output)
        uv_hidden_states.add_(uv_attn_output)

        # 3. Feed-forward
        mv_norm_hidden_states = (self.norm3(mv_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            mv_hidden_states
        )
        uv_norm_hidden_states = (self.norm3(uv_hidden_states.float()) * (1 + c_scale_msa_uv) + c_shift_msa_uv).type_as(
            uv_hidden_states
        )
        ff_output = self.ffn(mv_norm_hidden_states)
        mv_hidden_states = (mv_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(mv_hidden_states)
        ff_output_uv = self.ffnuv(uv_norm_hidden_states)
        uv_hidden_states = (uv_hidden_states.float() + ff_output_uv.float() * c_gate_msa_uv).type_as(uv_hidden_states)
        hidden_states = torch.cat([mv_hidden_states, uv_hidden_states], dim=1)

        return hidden_states


class WanT2TexTransformer3DModel(WanTransformer3DModel):
    """
    3D Transformer model for T2Tex.
    """
    def __init__(self, 
                 patch_size: Tuple[int] = (1, 2, 2),
                 num_attention_heads: int = 40,
                 attention_head_dim: int = 128,
                 in_channels: int = 16,
                 out_channels: int = 16,
                 text_dim: int = 4096,
                 freq_dim: int = 256,
                 ffn_dim: int = 13824,
                 num_layers: int = 40,
                 cross_attn_norm: bool = True,
                 qk_norm: Optional[str] = "rms_norm_across_heads",
                 eps: float = 1e-6,
                 image_dim: Optional[int] = None,
                 added_kv_proj_dim: Optional[int] = None,
                 rope_max_seq_len: int = 1024,
                 **kwargs
                ):
        super(WanT2TexTransformer3DModel, self).__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=2048
        )
        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(self.rope.attention_head_dim, self.rope.patch_size, self.rope.max_seq_len)
        self.norm_patch_embedding = copy.deepcopy(self.patch_embedding)
        self.pos_patch_embedding = copy.deepcopy(self.patch_embedding)

        # 2. Condition embeddings
        inner_dim = num_attention_heads * attention_head_dim
        self.condition_embedder = WanTimeTaskTextImageEmbedding(
            original_model=self.condition_embedder,
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.num_attention_heads = num_attention_heads

        block = WanT2TexTransformerBlock(
            inner_dim, 
            ffn_dim, 
            num_attention_heads, 
            qk_norm, 
            cross_attn_norm, 
            eps, 
            added_kv_proj_dim,
        )
        self.blocks = None
        self.blocks = nn.ModuleList(
            [
                copy.deepcopy(block)
                for _ in range(num_layers)
            ]
        )
        self.scale_shift_table_uv = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        
        # 4. Auto-configure LoRA adapter for SeqTex
        self.configure_lora_adapter()

    def configure_lora_adapter(self, lora_rank: int = 128, lora_alpha: int = 64):
        """
        Configure LoRA adapter with custom settings or auto-configuration.
        
        Args:
            lora_rank (int, optional): LoRA rank parameter, default (128)
            lora_alpha (int, optional): LoRA alpha parameter, default (64)
        """
        # Get parameters from args, environment variables, or defaults
        target_modules = [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", 
            "attn1.to_out.0", "attn1.to_out.2",
            "ffn.net.0.proj", "ffn.net.2"
        ]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        
        self.add_adapter(lora_config)

    @cache
    def get_attention_bias(self, mv_length, uv_length):
        total_len = mv_length + uv_length
        attention_mask = torch.ones((total_len, total_len), dtype=torch.bool)
        uv_start = mv_length
        attention_mask[:uv_start, uv_start:] = False

        attention_mask = repeat(attention_mask, "s l -> 1 h s l", h=self.num_attention_heads)
        attention_bias = torch.ones_like(attention_mask)
        attention_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
        attention_bias = attention_bias.to("cuda").contiguous()
        return attention_bias

    def forward(
            self,
            hidden_states: Tuple[torch.Tensor, torch.Tensor],
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: Optional[torch.Tensor] = None,
            # task_cond: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            use_qk_geometry: Optional[bool] = False,

        ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                raise NotImplementedError()

        assert timestep.ndim == 2, "Use Diffusion Forcing to set seperate timestep for each frame."

        mv_hidden_states, uv_hidden_states = hidden_states

        batch_size, num_channels, num_frames, height, width = mv_hidden_states.shape
        _, _, uv_num_frames, uv_height, uv_width = uv_hidden_states.shape

        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        post_uv_num_frames = uv_num_frames // p_t
        post_uv_height = uv_height // p_h
        post_uv_width = uv_width // p_w

        rotary_emb = self.rope(mv_hidden_states, uv_hidden_states)

        # Patchify
        mv_rgb_hidden_states, mv_pos_hidden_states, mv_norm_hidden_states = torch.chunk(mv_hidden_states, 3, dim=1)
        uv_rgb_hidden_states, uv_pos_hidden_states, uv_norm_hidden_states = torch.chunk(uv_hidden_states, 3, dim=1)
        mv_geometry_embedding = self.pos_patch_embedding(mv_pos_hidden_states) + self.norm_patch_embedding(mv_norm_hidden_states)
        uv_geometry_embedding = self.pos_patch_embedding(uv_pos_hidden_states) + self.norm_patch_embedding(uv_norm_hidden_states)
        
        mv_hidden_states = self.patch_embedding(mv_rgb_hidden_states)
        uv_hidden_states = self.patch_embedding(uv_rgb_hidden_states)
        if use_qk_geometry:
            mv_geometry_embedding = mv_geometry_embedding.flatten(2).transpose(1, 2)
            uv_geometry_embedding = uv_geometry_embedding.flatten(2).transpose(1, 2) # [B, F*H*W, C]
            geometry_embedding = torch.cat([mv_geometry_embedding, uv_geometry_embedding], dim=1)
        else:
            raise NotImplementedError("please set use_qk_geometry to True")
            # geometry_embedding = None
            # mv_hidden_states = mv_hidden_states + mv_geometry_embedding
            # uv_hidden_states = uv_hidden_states + uv_geometry_embedding

        mv_hidden_states = mv_hidden_states.flatten(2).transpose(1, 2)
        uv_hidden_states = uv_hidden_states.flatten(2).transpose(1, 2) # [B, F*H*W, C]
        hidden_states = torch.cat([mv_hidden_states, uv_hidden_states], dim=1) # [B, F*H*W, C]

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        # temb [B, F, 6*D], timestep_proj [B, F, 6*D], used to be [B, 6*D]
        timestep_proj = timestep_proj.unflatten(-1, (6, -1)) # [B, F, 6*D] -> [B, F, 6, D]

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        attn_bias = None

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, 
                    attn_bias, geometry_embedding, (post_patch_num_frames, post_patch_height, post_patch_width, post_uv_num_frames, post_uv_height, post_uv_width))
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, 
                                      attn_bias=attn_bias, geometry_embedding=geometry_embedding,
                                      token_shape=(post_patch_num_frames, post_patch_height, post_patch_width, post_uv_num_frames, post_uv_height, post_uv_width))

        # 5. Output norm, projection & unpatchify
        # [B, 2, D] chunk into [B, 1, D] and [B, 1, D], D is 1536
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        mv_temb, uv_temb = temb[:, :post_patch_num_frames], temb[:, post_patch_num_frames:]
        mv_temb = repeat(mv_temb, "B F D -> B 1 (F H W) D", H=post_patch_height, W=post_patch_width)
        uv_temb = repeat(uv_temb, "B F D -> B 1 (F H W) D", H=post_uv_height, W=post_uv_width)
        shift, scale = (self.scale_shift_table.view(1, 2, 1, inner_dim) + mv_temb).chunk(2, dim=1)
        shift_uv, scale_uv = (self.scale_shift_table_uv.view(1, 2, 1, inner_dim) + uv_temb).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.squeeze(1).to(hidden_states.device)
        scale = scale.squeeze(1).to(hidden_states.device)
        shift_uv = shift_uv.squeeze(1).to(hidden_states.device)
        scale_uv = scale_uv.squeeze(1).to(hidden_states.device)

        # Unpatchify
        uv_token_length = post_uv_num_frames * post_uv_height * post_uv_width
        mv_token_length = post_patch_num_frames * post_patch_height * post_patch_width
        assert uv_token_length + mv_token_length == hidden_states.shape[1]
        uv_hidden_states = hidden_states[:, mv_token_length:]
        mv_hidden_states = hidden_states[:, :mv_token_length]

        mv_hidden_states = (self.norm_out(mv_hidden_states.float()) * (1 + scale) + shift).type_as(mv_hidden_states)
        uv_hidden_states = (self.norm_out(uv_hidden_states.float()) * (1 + scale_uv) + shift_uv).type_as(uv_hidden_states)
        mv_hidden_states = self.proj_out(mv_hidden_states)
        uv_hidden_states = self.proj_out(uv_hidden_states)

        mv_hidden_states = mv_hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        mv_hidden_states = mv_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        mv_output = mv_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        uv_hidden_states = uv_hidden_states.reshape(
            batch_size, post_uv_num_frames, post_uv_height, post_uv_width, p_t, p_h, p_w, -1
        )
        uv_hidden_states = uv_hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        uv_output = uv_hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return ((mv_output, uv_output),)

