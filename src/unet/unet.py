"""
Code adapted from https://github.com/FlorentinBieder/PatchDDM-3D/blob/master/guided_diffusion/unet.py
"""
import math
import diffusers
import json
import os
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


from diffusers import ModelMixin, ConfigMixin, UNet3DConditionModel
from typing import Union

from abc import abstractmethod

from utils.attention import *
from utils.nn_utils import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, Union[SpatialTransformer, SpatialTransformer3D]):
                x = layer(x, context=context)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3 and self.resample_2d:
            #x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
            x = F.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        #stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        stride = 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        num_groups=32,
        resample_2d=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.num_groups = num_groups

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
            self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        num_groups=32,
        resample_2d=True,
        additive_skips=False,
        verbose=False,
        self_attention_blocks = [],#["input", "middle", "output"],
        cross_attention_blocks = [], #["input", "middle", "output"],
        cross_attention_dim = None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # main parameters
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint

        # self-attention block parameters
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # other paramters
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.num_groups = num_groups
        self.resample_2d = resample_2d
        self.additive_skips = additive_skips
        self.verbose = verbose

        # activate self-attention if required
        self.self_attention_blocks = self_attention_blocks
        self.input_blocks_SelfAtt = "input" in self.self_attention_blocks
        self.middle_block_SelfAtt = "middle" in self.self_attention_blocks
        self.output_blocks_SelfAtt = "output" in self.self_attention_blocks

        # activate cross-attention if required
        self.cross_attention_blocks = cross_attention_blocks
        self.cross_attention_dim = cross_attention_dim
        activate_CrossAtt = self.cross_attention_dim is not None
        self.input_blocks_CrossAtt = ("input" in self.cross_attention_blocks and activate_CrossAtt)
        self.middle_block_CrossAtt = ("middle" in self.cross_attention_blocks and activate_CrossAtt)
        self.output_blocks_CrossAtt = ("output" in self.cross_attention_blocks and activate_CrossAtt)

        self.config_ = self.setup_config()

        # Time embedding dimension
        time_embed_dim = model_channels * 4
        # Time embedding MLP
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Input convolution
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        ###############################################################
        # Input blocks
        ###############################################################
        self.levels_num = len(channel_mult)
        for level, mult in enumerate(channel_mult):
            num_selfatt_blocks, num_crossatt_blocks = 0, 0
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if self.input_blocks_SelfAtt:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                num_groups=self.num_groups,
                            )
                        )
                        num_selfatt_blocks += 1
                    if self.input_blocks_CrossAtt:
                        layers.append(
                            SpatialTransformer_nd(
                                dims,
                                ch,
                                n_heads=num_heads,
                                d_head=num_head_channels if num_head_channels > 0 else ch // num_heads,
                                context_dim=cross_attention_dim,
                                depth=1,
                            )
                        )
                        num_crossatt_blocks += 1
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1: # add a downsample block to all levels except the last
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, 
                            conv_resample, 
                            dims=dims, 
                            out_channels=out_ch, 
                            resample_2d=resample_2d,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
            # verbose
            print(f"down level {level}:")
            print(f"    - {num_res_blocks} ResBlocks")
            if num_selfatt_blocks:
                print(f"    - {num_selfatt_blocks} Attention blocks")
            if num_crossatt_blocks:
                print(f"    - {num_crossatt_blocks} CrossAttention blocks")
            if level != len(channel_mult) - 1:
                print("    - Downsample block")
        self.input_block_chans_bk = input_block_chans[:]

        ################################################################
        # Middle block
        ################################################################
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
            *([AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            )] if self.middle_block_SelfAtt else []
            ),
            *([SpatialTransformer_nd(
                dims,
                ch,
                n_heads=num_heads,
                d_head=num_head_channels if num_head_channels > 0 else ch // num_heads,
                context_dim=cross_attention_dim,
                depth=1,
            )] if self.middle_block_CrossAtt else []
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
            ),
        )
        # verbose
        print("bottleneck:")
        print(f"    - 1 ResBlock")
        if self.middle_block_SelfAtt :
                print(f"    - 1 Attention block")
        if self.middle_block_CrossAtt:
                print(f"    - 1 CrossAttention block")

        ####################################################################
        # Output blocks
        ####################################################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            num_selfatt_blocks, num_crossatt_blocks = 0, 0
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                mid_ch = model_channels * mult if not self.additive_skips else (
                        input_block_chans[-1] if input_block_chans else model_channels
                        )
                if ch != ich: 
                    print(f" channels don't match: {ch=: >4}, {ich=: >4}, {level=}, {i=}")
                else:
                    if self.verbose:
                        print(f" channels do    match: {ch=: >4}, {ich=: >4}, {level=}, {i=}")
                layers = [
                    ResBlock(
                        ch + ich if not self.additive_skips else ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mid_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                    )
                ]
                if ds in attention_resolutions:
                    if self.output_blocks_SelfAtt:
                        layers.append(
                            AttentionBlock(
                                mid_ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                num_groups=self.num_groups,
                            )
                        )
                        num_selfatt_blocks += 1
                    if self.output_blocks_CrossAtt:
                        layers.append(
                            SpatialTransformer_nd(
                                dims,
                                mid_ch,
                                n_heads=num_heads,
                                d_head=num_head_channels if num_head_channels > 0 else ch // num_heads,
                                context_dim=cross_attention_dim,
                                depth=1,
                            )
                        )
                        num_crossatt_blocks += 1
                ch = mid_ch
                self.output_blocks.append(TimestepEmbedSequential(*layers))
            if level and i == num_res_blocks - 1:
                #out_ch = chmodel_config_name_or_path
                ich = input_block_chans.pop()
                out_ch = model_channels * mult if not self.additive_skips else (
                        input_block_chans[-1] if input_block_chans else model_channels
                    )
                self.output_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch, 
                            conv_resample, 
                            dims=dims, 
                            out_channels=out_ch, 
                            resample_2d=resample_2d
                            )
                    )
                )
                ds //= 2
            # verbose
            print(f"up level {level}:")
            print(f"    - {num_res_blocks} ResBlocks")
            if num_selfatt_blocks:
                print(f"    - {num_selfatt_blocks} Attention blocks")
            if num_crossatt_blocks:
                print(f"    - {num_crossatt_blocks} CrossAttention blocks")
            if level:
                print("    - Upsample block")
            mid_ch = ch

        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # sanity check - make sure the unet is symmetric
        assert len(self.input_blocks) == len(self.output_blocks) + 1


    def forward(self, x, timesteps, context=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #assert x.device == self.devices[0], f"{x.device=} does not match {self.devices[0]=}"
        #assert timesteps.device == self.devices[0], f"{timesteps.device=} does not match {self.devices[0]=}"

        # Embed time steps
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Input blocks
        hs = []
        h = x
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context)
            hs.append(h)

        # Middle block
        h = self.middle_block(h, emb, context=context)

        # Output blocks with concatenation or additive skip connections
        for module in self.output_blocks:
            new_hs = hs.pop()
            if self.additive_skips:
               #print(h.shape, new_hs.shape)
                h = (h + new_hs)/2 # will break if the smallest spatial resolution is 1
            else:
                h = th.cat([h, new_hs], dim=1)
            h = module(h, emb, context=context)

        h = self.out(h)
        return h

    def setup_config(self):
        config = {"_class_name": "UNetModel", "_diffusers_version": diffusers.__version__,
            "in_channels": self.in_channels, "model_channels": self.model_channels,
            "out_channels": self.out_channels,  "num_res_blocks": self.num_res_blocks, "attention_resolutions": self.attention_resolutions,
            "dropout": self.dropout, "channel_mult": self.channel_mult, "conv_resample": self.conv_resample, "dims": self.dims,
            "use_checkpoint": self.use_checkpoint,
            "num_heads": self.num_heads, "num_head_channels": self.num_head_channels, "num_heads_upsample": self.num_heads_upsample,
            "use_scale_shift_norm": self.use_scale_shift_norm, "resblock_updown": self.resblock_updown,
            "use_new_attention_order": self.use_new_attention_order, "num_groups": self.num_groups,
            "resample_2d": self.resample_2d, "additive_skips": self.additive_skips, "verbose": self.verbose,
            "cross_attention_dim": self.cross_attention_dim,
            "self_attention_blocks": self.self_attention_blocks, "cross_attention_blocks": self.cross_attention_blocks,
        }
        return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save the configuration instance's parameters to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file to save a configuration instance's parameters.
        """
        config_ = self.setup_config()
        with open(json_file_path, "w", encoding="utf-8") as writer:
            json.dump(config_, writer, indent=2)


class MyUNet3DModel(ModelMixin, ConfigMixin):

    def __init__(
        self,
        sample_size: int = 32,
        in_channels: int = 4,
        out_channels: int = 4,
        cross_attention_dim: int = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = [int(p) for p in 32*np.array([1,2,4,8])]
        self.cross_attention_dim = cross_attention_dim
        self.cross_attention = self.cross_attention_dim is not None
        if self.cross_attention:
            down_block_types= ("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D")
            up_block_types = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D")
        else:
            down_block_types = ("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D")
            up_block_types = ("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D")

        self.unet = UNet3DConditionModel(sample_size=self.sample_size, in_channels=self.in_channels, out_channels=self.out_channels,
                                        block_out_channels=self.model_channels, \
                                        cross_attention_dim=self.cross_attention_dim,
                                        down_block_types=down_block_types,
                                        up_block_types=up_block_types)

    def forward(self, x,  timesteps, context=None):

        if self.cross_attention:
            bsz = x.shape[0]
            hidden_states = hidden_states.view(bsz, -1, self.cross_attention_dim)
        else:
            hidden_states = torch.ones((bsz, 1, self.cross_attention_dim))
        
        hidden_states = hidden_states.to(x.dtype).to(x.device)
        out = self.unet(x, timestep=timesteps, encoder_hidden_states=hidden_states).sample
        return out

