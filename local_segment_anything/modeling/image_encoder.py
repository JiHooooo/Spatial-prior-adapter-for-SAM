# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock
from .adapter_modules import SpatialPriorModule, SpatialPriorModule_self, \
            InteractionBlock_global, injector_global, deform_inputs, \
            InteractionBlock_deformable, InteractionBlock_deformable_mlp_on_vit

from torch.nn.init import normal_
from timm.models.layers import trunc_normal_
import math
from functools import partial

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        adapter_flag: bool = False,
        adapter_type: str = 'deformable',
        prompt_flag: str = 'simple',
        **kwargs,
  
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.adapter_flag = adapter_flag
        self.adapter_type = adapter_type
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        
        if self.adapter_flag:
        
                
            if self.adapter_type == 'spa':
                self.SPM = SpatialPriorModule_self(inplanes=64, embed_dim=embed_dim, with_cp=False, n_level_list = kwargs['n_level_list'])
                
                self.n_level_ratio_list = kwargs['n_level_list']
                self.level_embed = nn.Parameter(torch.zeros(len(self.n_level_ratio_list), embed_dim))
                normal_(self.level_embed)
                self.interaction_indexes = kwargs['interaction_indexes']
                self.attn_type = kwargs['attn_type']
                if self.attn_type == 'global':
                    self.interactions = nn.Sequential(*[
                                            InteractionBlock_global(embed_dim, num_heads=1, downsample_rate=2)
                                                for i in range(len(self.interaction_indexes))
                                            ])
                    self.final_interaction = injector_global(embed_dim, num_heads=1, downsample_rate=2)
                elif self.attn_type == 'deformable':
                    self.interactions = nn.Sequential(*[
                                            InteractionBlock_deformable(embed_dim, num_heads=kwargs['deform_num_heads'], 
                                                    n_points=kwargs['n_points'],
                                                    init_values=kwargs['init_values'], drop_path=kwargs['drop_path'],
                                                    norm_layer=norm_layer,
                                                    cffn_ratio=kwargs['cffn_ratio'], deform_ratio=kwargs['deform_ratio'],
                                                    with_cp=kwargs['with_cp'], final_block=False, n_level=len(self.n_level_ratio_list))
                                                for i in range(len(self.interaction_indexes))
                                            ])
                    self.final_interaction = InteractionBlock_deformable(embed_dim, num_heads=kwargs['deform_num_heads'], 
                                                    n_points=kwargs['n_points'],
                                                    init_values=kwargs['init_values'], drop_path=kwargs['drop_path'],
                                                    norm_layer=norm_layer,
                                                    cffn_ratio=kwargs['cffn_ratio'], deform_ratio=kwargs['deform_ratio'],
                                                    with_cp=kwargs['with_cp'], final_block=True, n_level=len(self.n_level_ratio_list))
                #
                elif self.attn_type == 'deformable_vit_mlp':
                    self.interactions = nn.Sequential(*[
                                            InteractionBlock_deformable_mlp_on_vit(embed_dim, num_heads=kwargs['deform_num_heads'], 
                                                    n_points=kwargs['n_points'],
                                                    init_values=kwargs['init_values'], drop_path=kwargs['drop_path'],
                                                    norm_layer=norm_layer,
                                                    cffn_ratio=kwargs['cffn_ratio'], deform_ratio=kwargs['deform_ratio'],
                                                    with_cp=kwargs['with_cp'], final_block=False, n_level=len(self.n_level_ratio_list))
                                                for i in range(len(self.interaction_indexes))
                                            ])
                    self.final_interaction = InteractionBlock_deformable_mlp_on_vit(embed_dim, num_heads=kwargs['deform_num_heads'], 
                                                    n_points=kwargs['n_points'],
                                                    init_values=kwargs['init_values'], drop_path=kwargs['drop_path'],
                                                    norm_layer=norm_layer,
                                                    cffn_ratio=kwargs['cffn_ratio'], deform_ratio=kwargs['deform_ratio'],
                                                    with_cp=kwargs['with_cp'], final_block=False, n_level=len(self.n_level_ratio_list))
                self.SPM.apply(self._init_weights)
                self.interactions.apply(self._init_weights)
                self.final_interaction.apply(self._init_weights)

        if prompt_flag == 'simple':
            self.segmentic_token = nn.Embedding(1, out_chans)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def _add_level_embed_flex(self, c_list):
        c_list_new = [c_list[i] + self.level_embed[i] for i in range(len(c_list))]
        return c_list_new
    

    def forward_with_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if 'deformable' in self.attn_type:
            deform_inputs1, deform_inputs2 = deform_inputs(x, self.n_level_ratio_list)
    
        # SPM forward
        # c1 [bs, dim, H/4, W/4]
        # c2-c4 [bs, H/8 - H/16 - H/32, dim] 
        c2, c3, c4 = self.SPM(x)
        feature_dict = {8:c2, 16:c3, 32:c4}
        
        c_list = [feature_dict[i] for i in self.n_level_ratio_list]
        c_list = self._add_level_embed_flex(c_list)
        # [bs, N, dim]
        c = torch.cat(c_list, dim=1)
        
        #
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        bs, H, W, dim = x.shape
        for blk_index, blk in enumerate(self.blocks):
            if blk_index in self.interaction_indexes:
                
                x = x.view(bs, -1, dim)
                if 'deformable' in self.attn_type:
                    
                    x, c = self.interactions[self.interaction_indexes.index(blk_index)](x, c, deform_inputs1, deform_inputs2, H, W)
                else:
                    x, c = self.interactions[self.interaction_indexes.index(blk_index)](x, c)
                x = x.view(bs, H, W, dim)
            x = blk(x)
        
        # final part
        x = x.view(bs, -1, dim)
        if 'deformable' in self.attn_type:
            x, c  = self.final_interaction(x, c,deform_inputs1, deform_inputs2, H, W)
        else:
            x = self.final_interaction(x, c)
        #default
        x = x.view(bs, H, W, dim)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

        # # test
        # x = x.view(bs, H, W, dim).permute(0,3,1,2)
        # return x, c
    def standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # test
        # x = x.permute(0,3,1,2)
        # return x

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter_flag:
            if self.adapter_type == 'spa':
                return self.forward_with_adapter(x)
            else:
                raise TypeError('the adapter type should be spa but got %s'%(self.adapter_type))
        else:
            return self.standard_forward(x)


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size if window_size == 0 else (window_size, window_size),
            )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor, return_attn_flag=False) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        if return_attn_flag:
            x, attn = self.attn(x, return_attn_flag)
        else:
            x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        if return_attn_flag:
            return x, attn
        else:
            return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor, return_attn_flag=False) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        if return_attn_flag:
            return x, attn
        else:
            return x

class Attention_lora(Attention):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        lora_dim: int = 4,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
                         rel_pos_zero_init=rel_pos_zero_init, input_size=input_size)
        self.dim = dim
        self.lora_dim = lora_dim
        self.lora_a_q = nn.Linear(dim, self.lora_dim, bias=False)
        self.lora_b_q = nn.Linear(self.lora_dim, dim, bias=False)
        self.lora_a_v = nn.Linear(dim, self.lora_dim, bias=False)
        self.lora_b_v = nn.Linear(self.lora_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, return_attn_flag=False) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        # lora part
        qkv = self.qkv(x) # (B, N, N , 3*dim)
        new_q = self.lora_b_q(self.lora_a_q(x))
        new_v = self.lora_b_v(self.lora_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        if return_attn_flag:
            return x, attn
        else:
            return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
