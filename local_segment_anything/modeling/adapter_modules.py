import logging
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as cp

# from ops.modules import MSDeformAttn
from timm.models.layers import DropPath
from typing import Type

_logger = logging.getLogger(__name__)
import math

# for MSDeformAttn
from .functions import MSDeformAttnFunction
from torch.nn.init import constant_, xavier_uniform_

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).\
            view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,
                                            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = int(embedding_dim // downsample_rate)
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



class InteractionBlock_deformable(nn.Module):
    def __init__(self, embed_dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop_path=0., cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, with_cp=False, final_block = True, n_level=3):
        super().__init__()
        
        self.injector = Injector(embed_dim, n_levels=n_level, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        if not final_block:
            self.extractor = None
        else:
            self.extractor = extractor_deform(embed_dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                norm_layer=norm_layer, deform_ratio=deform_ratio,
                                cffn_ratio=cffn_ratio, drop_path=drop_path, with_cp=with_cp)

    def forward(self, vit_feature, adapter_feature, deform_inputs1, deform_inputs2, H, W):
        # bs, _, dim
        if self.extractor is not None:
            adapter_feature = self.extractor(adapter_feature=adapter_feature, reference_points=deform_inputs2[0],
                                vit_feature=vit_feature, spatial_shapes=deform_inputs2[1],
                                level_start_index=deform_inputs2[2], H=H, W=W)
        # bs, _, dim
        vit_feature = self.injector(query=vit_feature, reference_points=deform_inputs1[0],
                            feat=adapter_feature, spatial_shapes=deform_inputs1[1],
                            level_start_index=deform_inputs1[2], ) 
        
        return vit_feature, adapter_feature
    


class InteractionBlock_global(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1, mlp_vit=False, cff_ratio=0.5):
        super().__init__()
        if mlp_vit:
            self.injector = injector_global(embed_dim, num_heads, downsample_rate, with_cffn=True, mlp_dim=int(embed_dim*cff_ratio))
            self.extractor = extractor_global(embed_dim, num_heads, downsample_rate, with_cffn=False)
        else:
            self.injector = injector_global(embed_dim, num_heads, downsample_rate, with_cffn=False)
            self.extractor = extractor_global(embed_dim, num_heads, downsample_rate, with_cffn=True, mlp_dim=int(embed_dim*cff_ratio))

    def forward(self, vit_feature, adapter_feature):
        adapter_feature = self.extractor(vit_feature, adapter_feature)
        vit_feature = self.injector(vit_feature, adapter_feature)
        
        return vit_feature, adapter_feature

class injector_global(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1, with_cffn=True, mlp_dim=2048, mlp_activation=nn.ReLU):
        super().__init__()
        self.gamma = nn.Parameter(0 * torch.ones((embed_dim)), requires_grad=True)
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.adapter_norm = nn.LayerNorm(embed_dim)
        self.atten = Attention(embedding_dim = embed_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.with_cffn = with_cffn
        if self.with_cffn:
            self.mlp = MLPBlock(embed_dim, mlp_dim, mlp_activation)
            self.mlp_norm = nn.LayerNorm(embed_dim)
            self.drop_path = nn.Identity()

    def forward(self, vit_feature, adapter_feature):

        vit_feature_norm = self.vit_norm(vit_feature)

        adapter_feature_norm = self.adapter_norm(adapter_feature)
        
        attn = self.atten(q=vit_feature_norm, k=adapter_feature_norm, v=adapter_feature_norm)

        vit_feature = vit_feature + self.gamma * attn

        if self.with_cffn:
            vit_feature = vit_feature +  self.drop_path(self.mlp(self.mlp_norm(vit_feature)))

        return vit_feature

    
class extractor_global(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1, mlp_dim=2048, mlp_activation=nn.ReLU, with_cffn=True):
        super().__init__()
        # self.gamma = nn.Parameter(0 * torch.ones((embed_dim)), requires_grad=True)
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.adapter_norm = nn.LayerNorm(embed_dim)
        self.atten = Attention(embedding_dim = embed_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        
        self.with_cffn = with_cffn
        if self.with_cffn:
            self.mlp = MLPBlock(embed_dim, mlp_dim, mlp_activation)
            self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, vit_feature, adapter_feature):

        vit_feature_norm = self.vit_norm(vit_feature)

        adapter_feature_norm = self.adapter_norm(adapter_feature)

        attn = self.atten(q=adapter_feature_norm, k=vit_feature_norm, v=vit_feature_norm)

        adapter_feature = adapter_feature + attn

        if self.with_cffn:
            adapter_feature = adapter_feature + self.mlp(self.mlp_norm(adapter_feature))

        return adapter_feature

class extractor_deform(nn.Module):
    def __init__(self, embed_dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, mlp_act=nn.GELU):
        super().__init__()
        # self.gamma = nn.Parameter(0 * torch.ones((embed_dim)), requires_grad=True)
        self.vit_norm = norm_layer(embed_dim)
        self.adapter_norm = norm_layer(embed_dim)
        self.attn = MSDeformAttn(d_model=embed_dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        if with_cffn:
            self.mlp = MLPBlock(embed_dim, int(embed_dim * cffn_ratio), mlp_act)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.mlp_norm = norm_layer(embed_dim)
        self.with_cp = with_cp

    def forward(self, adapter_feature, reference_points, vit_feature, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):
            attn = self.attn(self.adapter_norm(query), reference_points,
                             self.vit_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
            if self.with_cffn:
                query = query + self.drop_path(self.mlp(self.mlp_norm(query)))
            return query
        
        if self.with_cp and adapter_feature.requires_grad:
            adapter_feature = cp.checkpoint(_inner_forward, adapter_feature, vit_feature)
        else:
            adapter_feature = _inner_forward(adapter_feature, vit_feature)
            
        return adapter_feature
    

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, n_level_ratio_list=[8,16,32]):
    bs, c, h, w = x.shape
    
    # spatial_shapes = torch.as_tensor([(h // 8, w // 8),
    #                                 (h // 16, w // 16),
    #                                 (h // 32, w // 32)],
    #                                 dtype=torch.long, device=x.device)
    spatial_shapes = torch.as_tensor([(h//i, w//i) for i in n_level_ratio_list],
                                    dtype=torch.long, device=x.device)

    
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # reference_points = get_reference_points([(h // 8, w // 8),
    #                                          (h // 16, w // 16),
    #                                          (h // 32, w // 32)], x.device)
    reference_points = get_reference_points([(h//i, w//i) for i in n_level_ratio_list], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
    
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query

class InteractionBlock_deformable_mlp_on_vit(nn.Module):
    def __init__(self, embed_dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop_path=0., cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, with_cp=False, final_block = True, n_level=3, deform_attn=True):
        super().__init__()
        self.deform_attn = deform_attn
        if self.deform_attn:
            self.injector = Injector(embed_dim, n_levels=n_level, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                              with_cp=with_cp, with_cffp=True, drop_path=drop_path, cffn_ratio=cffn_ratio)
        else:
            self.injector = Injector_mlp_global(embed_dim, num_heads=num_heads, downsample_rate=1/cffn_ratio, norm_layer=norm_layer,
                                                drop_path=drop_path, cffn_ratio=cffn_ratio, with_cp=True)
        if not final_block:
            self.extractor = None
        else:
            if self.deform_attn:
                self.extractor = extractor_deform(embed_dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                    norm_layer=norm_layer, deform_ratio=deform_ratio,
                                    cffn_ratio=cffn_ratio, drop_path=drop_path, with_cp=with_cp, with_cffn=False)
            else:
                self.extractor = extractor_global_no_mlp(embed_dim=embed_dim,num_heads=num_heads, downsample_rate=1/cffn_ratio)

    def forward(self, vit_feature, adapter_feature, deform_inputs1=None, deform_inputs2=None, H=None, W=None):
        # bs, _, dim
        if self.deform_attn:
            if self.extractor is not None:
                adapter_feature = self.extractor(adapter_feature=adapter_feature, reference_points=deform_inputs2[0],
                                    vit_feature=vit_feature, spatial_shapes=deform_inputs2[1],
                                    level_start_index=deform_inputs2[2], H=H, W=W)
            # bs, _, dim
            vit_feature = self.injector(query=vit_feature, reference_points=deform_inputs1[0],
                                feat=adapter_feature, spatial_shapes=deform_inputs1[1],
                                level_start_index=deform_inputs1[2], ) 
        else:
            if self.extractor is not None:
                adapter_feature = self.extractor(vit_feature, adapter_feature)
            vit_feature = self.injector(vit_feature, adapter_feature)

        return vit_feature, adapter_feature


class Injector_deform(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False,
                 cffn_ratio=0.24, drop_path=0., mlp_act=nn.GELU, with_cffp=True):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        
        if with_cffp:
            self.mlp = MLPBlock(dim, int(dim * cffn_ratio), mlp_act)
            self.mlp_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            
            return query + self.gamma * (self.drop_path(self.mlp(self.mlp_norm(attn))))
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query
    
class extractor_global_no_mlp(nn.Module):
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1,):
        super().__init__()
        # self.gamma = nn.Parameter(0 * torch.ones((embed_dim)), requires_grad=True)
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.adapter_norm = nn.LayerNorm(embed_dim)
        self.atten = Attention(embedding_dim = embed_dim, num_heads=num_heads, downsample_rate=downsample_rate)
    
    def forward(self, vit_feature, adapter_feature):

        vit_feature_norm = self.vit_norm(vit_feature)

        # adapter_feature_norm = self.adapter_norm(adapter_feature)

        attn = self.atten(q=self.adapter_norm(adapter_feature), k=vit_feature_norm, v=vit_feature_norm)

        # adapter_feature = adapter_feature + attn

        # mlp_out = self.mlp_norm(adapter_feature)

        return adapter_feature + attn

class Injector_mlp_global(nn.Module):
    def __init__(self, dim, num_heads=6, downsample_rate=1, cffn_ratio=0.25, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_activation=nn.ReLU, drop_path=0.0, init_values=0.0,
                 with_cp=True):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.mlp_norm = norm_layer(dim)
        self.attn = Attention(embedding_dim = dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.mlp = MLPBlock(dim, int(dim * cffn_ratio), mlp_activation)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, vit_feature, adapter_feature):
        
        def _inner_forward(query, feat):
            feat_norm = self.feat_norm(feat)
            attn = self.attn(q=self.query_norm(vit_feature), k=feat_norm, v=feat_norm)
            
            return query + self.gamma * (self.drop_path(self.mlp(self.mlp_norm(attn))))
        
        if self.with_cp and vit_feature.requires_grad:
            query = cp.checkpoint(_inner_forward, vit_feature, adapter_feature)
        else:
            query = _inner_forward(vit_feature, adapter_feature)
            
        return query

class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, 
                 with_cffp=False, drop_path=0.0,  mlp_activation=nn.ReLU, cffn_ratio=0.25):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.with_cffp = with_cffp
        if self.with_cffp:
            self.mlp_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.mlp = MLPBlock(dim, int(dim * cffn_ratio), mlp_activation)
            
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        
        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            
            query = query + self.gamma * attn
            if self.with_cffp:
                query = query + self.drop_path(self.mlp(self.mlp_norm(query)))
            return query
        
            # if self.with_cffp:
            #     return query + self.gamma * (self.drop_path(self.mlp(self.mlp_norm(attn))))
            # else:
            #     return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls
    

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    
class SpatialPriorModule_self(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, n_level_list=[8,16,32]):
        super().__init__()
        self.with_cp = with_cp
        self.n_level_list = n_level_list
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        if 16 in self.n_level_list or 32 in self.n_level_list:
            self.conv3 = nn.Sequential(*[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True)
            ])
        if 32 in self.n_level_list:
            self.conv4 = nn.Sequential(*[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True)
            ])
        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        if 8 in self.n_level_list:
            self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        if 16 in self.n_level_list:
            self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        if 32 in self.n_level_list:
            self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            
            bs = x.shape[0]
            dim = self.embed_dim

            c1 = self.stem(x)
            c2 = self.conv2(c1)
            if 16 in self.n_level_list or 32 in self.n_level_list:
                c3 = self.conv3(c2)
            if 32 in self.n_level_list:
                c4 = self.conv4(c3)
            # c1 = self.fc1(c1)
            if 8 in self.n_level_list:
                c2 = self.fc2(c2)
            if 16 in self.n_level_list:
                c3 = self.fc3(c3)
            if 32 in self.n_level_list:
                c4 = self.fc4(c4)
    
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            if 8 in self.n_level_list:
                c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            else:
                c2 = 0
            if 16 in self.n_level_list:
                c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            else:
                c3 = 0
            if 32 in self.n_level_list:
                c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
            else:
                c4 = 0
    
            return c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs